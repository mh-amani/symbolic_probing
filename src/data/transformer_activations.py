from typing import Any, Dict, Optional, Tuple
from omegaconf import OmegaConf
import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from datasets import load_dataset
from torchvision.transforms import transforms
import hydra


class TransformerActivationsDataModule(LightningDataModule):
    """`LightningDataModule` for a dataset of text inputs and transformer activations.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    """

    def __init__(
        self,
        **kwargs: Dict[str, Any],
    ) -> None:
        """Initialize a `TransformerActivationsDataModule`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms = transforms.Compose(
            [transforms.ToTensor(),]
        )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = self.hparams.batch_size

    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        :return: The number of MNIST classes (10).
        """
        return 10

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        pass
        

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load dataset with HF datasets
        hf_dataset_configs = OmegaConf.to_container(self.hparams.dataset_config, resolve=True)
        datasets = load_dataset(self.hparams.dataset_name, **hf_dataset_configs)

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            trainset = datasets['train']
            valset = datasets['validation']
            testset = datasets['test']

            self.data_train = datasets['train']
            self.data_val = datasets['validation']
            self.data_test = datasets['test']

            # dataset = ConcatDataset(datasets=[trainset, valset, testset])
            # self.data_train, self.data_val, self.data_test = random_split(
            #     dataset=dataset,
            #     lengths=self.hparams.train_val_test_split,
            #     generator=torch.Generator().manual_seed(42),
            # )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )


def ActivationAugmentedDataset(dataset: Dataset, model_under_lens):
    """A dataset that augments the input data with transformer activations.

    :param dataset: The input dataset.
    :param model_under_lens: The model under lens.
    """
    pass



@hydra.main(version_base="1.3", config_path="../../configs/data", config_name="transformer_activations.yaml")
def main(cfg):
    datamodule = TransformerActivationsDataModule(**cfg)
    datamodule.setup()
    train_dataloader = datamodule.train_dataloader()

    # print the number of batches in the train dataloader
    print(f"Number of batches in the train dataloader: {len(train_dataloader)}")


if __name__ == "__main__":
    main()