from typing import Any, Dict, Tuple, Optional
from omegaconf import DictConfig
import torch
from torch.nn import ModuleDict
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
import hydra
from blocks.modules.auto_reg_wrapper import AutoRegWrapper
from blocks.unwrapped_models.enc_dec_unwrapper import Unwrappedbart
from transformers import AutoTokenizer, BertModel
from transformer_lens import HookedTransformer
import code

class EncDecEncModel(LightningModule):
    """a wrapper connecting two sequence models with discrete bottleneck layers

    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler = None,
        **kwargs: Any,
    ) -> None:

        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=[])
        if self.hparams.trainer_accelerator == 'cpu':
            self.hparams.device = 'cpu'
        else:
            try:
                trainer_device = self.hparams.trainer_devices[0]
                self.hparams.device = 'cuda:' + str(trainer_device)
            except:
                self.hparams.device = 'cuda:0'
                print('No GPU device list found, using cuda:0')

        self.probed_model = HookedTransformer.from_pretrained(self.hparams.probed_model_name, device=self.hparams.device)

        self.activation_embedding_to_encoder = torch.nn.Linear(self.probed_model.cfg.d_model,
                                                                    self.hparams.encdec_config.d_model)
        self.encoder_to_activation_embedding = torch.nn.Linear(self.hparams.encdec_config.d_model,
                                                                    self.probed_model.cfg.d_model)
        encdec_vector_model, encoder_embedding, _, _ = Unwrappedbart(
                hydra.utils.instantiate(self.hparams.encdec_config, vocab_size=self.probed_model.tokenizer.vocab_size))
        self.enc_model = BertModel(hydra.utils.instantiate(self.hparams.enc_config))
        
        
        # initializing the discretizers
        # self.hparams.probe_discretizer_config['encoder_embedding_weight']= probe_encoder_embedding_weight
        probe_encoder_embedding = self.enc_model.get_input_embeddings()
        self.hparams.probe_discretizer_config['decoder_embedding']= None
        self.hparams.probe_discretizer_config['linear_head']= None
        self.probe_discretizer = hydra.utils.instantiate(self.hparams.probe_discretizer, {**self.hparams.probe_discretizer_config,
                                                        'encoder_embedding': probe_encoder_embedding})
        
        self.hparams.input_discretizer_config.dimensions['vocab_size'] = self.probed_model.tokenizer.vocab_size
        self.hparams.input_discretizer_config.dimensions['unembedding_dim'] = self.probed_model.tokenizer.vocab_size
        self.input_dicsretizer = hydra.utils.instantiate(self.hparams.input_discretizer, {**self.hparams.input_discretizer_config,
                                                                                        'encoder_embedding': encoder_embedding})
    
        # self.input_tokenizer = AutoTokenizer.from_pretrained(self.hparams.input_tokenizer_name)
        self.input_tokenizer = self.probed_model.tokenizer
        self.tokenizer_config = {'control_token_ids': { 'input_pad_token_id': self.input_tokenizer.pad_token_id,
                                    'output_eos_token_id': self.input_tokenizer.eos_token_id, 
                                    'output_pad_token_id': self.input_tokenizer.pad_token_id,
                                    'output_unknown_token_id': self.input_tokenizer.unk_token_id,},
                'output_prepending_ids': torch.tensor(self.input_tokenizer.bos_token_id)
            }
        self.autoreg_wrapped_encdec_model = AutoRegWrapper(encdec_vector_model, self.input_dicsretizer, 
                                                           self.probe_discretizer, config={**self.hparams.autoreg_wrapper_config,
                                                                                           **self.tokenizer_config, 'device': self.hparams.device})

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()
        # metric objects for calculating and averaging accuracy across batches
        # self.train_acc = Accuracy(task="multiclass", num_classes=10)
        # self.val_acc = Accuracy(task="multiclass", num_classes=10)
        # self.test_acc = Accuracy(task="multiclass", num_classes=10)
        # # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        # self.test_loss = MeanMetric()
        # # for tracking best so far validation accuracy
        # self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of input data. size: [batch_size, sequence_length, num_features]
        :return: A tensor of logits.
        """
        (input_token_embeds, activations_embeds, attention_mask) = x
        activations_embeds_transformed = self.activation_embedding_to_encoder(activations_embeds)
        input_token_embeds[:, 0] = activations_embeds_transformed
        symbolic_representation = self.autoreg_wrapped_encdec_model(input_embeds_enc=input_token_embeds, input_attention_mask=attention_mask,
                                            max_output_length=None)
        encoder_output = self.enc_model(inputs_embeds=symbolic_representation['quantized_vector_encoder'], attention_mask=symbolic_representation['output_attention_mask'])['last_hidden_state']
        reconstructed_x = self.encoder_to_activation_embedding(encoder_output.mean(axis=1))
        loss = torch.nn.functional.mse_loss(reconstructed_x, activations_embeds)
        return reconstructed_x, loss

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.probed_model.eval()
        # self.val_acc.reset()
        # self.val_acc_best.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        x = batch['tokens']
        with torch.no_grad():
            (probed_logits, probed_loss), cache = self.probed_model.run_with_cache(x, return_type='both')
            activations = cache[self.hparams.probed_layer]
            # to decode and compare the logits and see how the predictions look like
            # self.probed_model.tokenizer.decode(probed_logits.argmax(dim=-1)[0]) 
            # probed_model.tokenizer.decode(logits.argmax(dim=-1)[0]) 
            # randomly choosing one token at each row to be the target
            mask = torch.randint(1, x.size(1), (x.size(0),)).unsqueeze(1).expand(-1, x.size(1)).to(x)
            # masking all the tokens appearing after the target token
            mask_embeds = mask >= torch.arange(x.size(1),).unsqueeze(0).to(x)
            mask_eos = mask == torch.arange(x.size(1),).unsqueeze(0).to(x)
            # getting the embeddings of the target tokens
            activations_embeds = activations[mask_eos, :]
            filtered_tokens = x * mask_embeds + self.input_tokenizer.pad_token_id * (~mask_embeds)
            input_token_embeds = self.input_dicsretizer.encoder_embedding_from_id(filtered_tokens)

        reconstruction, reconstruction_loss = self.forward((input_token_embeds, activations_embeds, mask_embeds))
        
        loss = reconstruction_loss # + self.disc_loss_coeff 

        return loss, reconstruction

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, reconstruction = self.model_step(batch)
        self.log("train/loss", loss, on_step=True, prog_bar=True)

        # update and log metrics
        self.train_loss(loss)
        # self.train_acc(preds, targets)
        # self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("train/disc_loss", disc_loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        scheduler = self.lr_schedulers()
        # If the selected scheduler is a ReduceLROnPlateau scheduler.
        # scheduler.step()
        scheduler.step(self.trainer.callback_metrics[self.hparams.monitor])
        self.log(name=f'lr', value=scheduler._last_lr[0], sync_dist=True)

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, reconstruction = self.model_step(batch)
        self.val_loss(loss)
        # update and log metrics
        self.log("val/loss", self.val_loss, on_step=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        pass
        # acc = self.val_acc.compute()  # get current val acc
        # self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, reconstruction = self.model_step(batch)

        # update and log metrics
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.test_loss(loss)
        # self.test_acc(preds, targets)
        # self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("test/disc_loss", disc_loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        # if self.hparams.compile and stage == "fit":
        #     self.net = torch.compile(self.net)
        pass
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = hydra.utils.instantiate(self.hparams.optimizer)(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler =  hydra.utils.instantiate(self.hparams.scheduler)(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    'monitor': self.hparams['monitor'], # "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}



@hydra.main(version_base="1.3", config_path="../../configs/model", config_name="encdec_enc.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    # the encoder and decoder
    model = EncDecEncModel(**cfg)
    print(model)
    
if __name__ == "__main__":
    main()



############################ unused code for when we had a unique dictionary instead of the two embeddings ################################
# self.model_x_to_z = hydra.utils.instantiate(cfg.modules.model_x_to_z, 
#                                             **self.hparams.modules.config_x_to_z,
#                                             special_tokens_ids=self.special_tokens_ids, _recursive_ = False)
# self.model_z_to_x = hydra.utils.instantiate(self.hparams.modules.model_z_to_x, 
#                                             **self.hparams.modules.config_z_to_x,
#                                             special_tokens_ids=self.special_tokens_ids, _recursive_ = False)


# # code to check if all params are on device
# self.trainer.optimizers[0].param_groups[0]['params']
# for param in self.trainer.optimizers[0].param_groups[0]['params']:
#     if not param.device == torch.device('cuda:0'):
#         print(param.device)
#         print(param.requires_grad)
#         print(param.dtype)
#         print(param.shape)
#         # print(param)
#         print('-----------------')
