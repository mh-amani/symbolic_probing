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
        
        self.encdec_vector_model, self.encoder_embedding_weight, self.decoder_embedding_weight, \
                                self.linearhead_weight, self.linearhead_bias = Unwrappedbart(hydra.utils.instantiate(self.hparams.encdec_config))
        self.enc_model = BertModel(hydra.utils.instantiate(self.hparams.enc_config))
        probe_encoder_embedding_weight = self.enc_model.get_input_embeddings().weight

        self.probe_discretizer = hydra.utils.instantiate(self.hparams.probe_discretizer, {**self.hparams.probe_discretizer_config,
            **{'encoder_embedding_weight': probe_encoder_embedding_weight, \
                'decoder_embedding_weight': self.decoder_embedding_weight, 
                'linear_head_weight': self.linearhead_weight, 'linear_head_bias': self.linearhead_bias}} )
        self.input_dicsretizer = hydra.utils.instantiate(self.hparams.input_discretizer, 
                                                         {**self.hparams.input_discretizer_config, 
                                                          'encoder_embedding_weight': self.encoder_embedding_weight})
        
        self.input_tokenizer = AutoTokenizer.from_pretrained(self.hparams.input_tokenizer_name)
        config = {'control_token_ids': { 'input_pad_token_id': self.input_tokenizer.pad_token_id,
                                    'output_eos_token_id': self.input_tokenizer.eos_token_id, 
                                    'output_pad_token_id': self.input_tokenizer.pad_token_id,
                                    'output_unknown_token_id': self.input_tokenizer.unk_token_id,},
                'output_prepending_ids': torch.tensor(self.input_tokenizer.bos_token_id)
            }
        self.autoreg_wrapped_encdec_model = AutoRegWrapper(self.encdec_vector_model, self.input_dicsretizer, 
                                                           self.probe_discretizer, config={**config,
                                                                                           **self.hparams.autoreg_wrapper_config})

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()
        # metric objects for calculating and averaging accuracy across batches
        # self.train_acc = Accuracy(task="multiclass", num_classes=10)
        # self.val_acc = Accuracy(task="multiclass", num_classes=10)
        # self.test_acc = Accuracy(task="multiclass", num_classes=10)
        # # for averaging loss across batches
        # self.train_loss = MeanMetric()
        # self.val_loss = MeanMetric()
        # self.test_loss = MeanMetric()
        # # for tracking best so far validation accuracy
        # self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of input data. size: [batch_size, sequence_length, num_features]
        :return: A tensor of logits.
        """
        y = self.autoreg_wrapped_encdec_model(x)
        discretized_y = self.probe_discretizer(y)
        reconstructed_x = self.enc_model(discretized_y['encoder_hidden_states'])
        return y, discretized_y, reconstructed_x

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

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
        x, y = batch
        logits, disc_loss = self.forward(x)
        loss = self.criterion(logits, y) + self.disc_loss_coeff * disc_loss
        preds = torch.argmax(logits, dim=1)
        return loss, disc_loss, preds, y

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, disc_loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/disc_loss", disc_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

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
        loss, disc_loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/disc_loss", disc_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, disc_loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/disc_loss", disc_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

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

