from typing import Any, Dict, Optional, Tuple

import hydra
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
from tqdm import tqdm
import code
from omegaconf import DictConfig


class PVRDataModule(LightningDataModule):
    """`LightningDataModule` for the PVR dataset.

    PVR info

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

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    func_dict = {'sum_mod_10': lambda x: x.sum() % 10,}

    def __init__(self, **kwargs) -> None:
        self,
        # data_dir: str = "data/",
        # train_val_test_split: Tuple[int, int, int] = (55_000, 5_000, 10_000),
        # batch_size: int = 64,
        # num_workers: int = 0,
        # pin_memory: bool = False,

        """Initialize a `PVRDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        self.save_hyperparameters(logger=False)
        # code.interact(local=locals())   
        # self.dataset_parameters = dataset_parameters
        # self.data_dir: str = dataset_parameters.get('data_dir', 'data/')
        # self.train_val_test_split: Tuple[int, int, int] = dataset_parameters.get('train_val_test_split', (55_000, 5_000, 10_000))
        # self.batch_size: int = dataset_parameters.get('batch_size', 64)
        # self.num_workers: int = dataset_parameters.get('num_workers', 0)
        # self.pin_memory: bool = dataset_parameters.get('pin_memory', False)

        # self.seed: int = dataset_parameters.get('seed', 0)
        # self.pointer_size: int = dataset_parameters.get('pointer_size', 1)
        # self.agg_func = dataset_parameters.get('agg_func', lambda x: x.sum() % 10)
        # self.window_size: int = dataset_parameters.get('window_size', 3)
        # self.trim_window: bool = dataset_parameters.get('trim_window', False)

        # self.split = self.params['split']
        # assert self.split in {"train", "val", "test"}, "Unexpected split reference"

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # # data transformations
        # self.transforms = transforms.Compose(
        #     [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        # )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = kwargs['batch_size']

    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        :return: The number of possible outputs which is 10.
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

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            generator = torch.Generator().manual_seed(self.hparams.seed)
            dataset = self.__generate_data(sum(self.hparams.train_val_test_split), generator=generator) # TODO: move this to prepare_data?
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=self.hparams.train_val_test_split,
                generator=generator,
            )

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

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        # TODO: Implement
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        # TODO: Implement
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


    def __generate_data(self, n_samples:int, generator:torch.Generator):
        """Generate data for the PVR dataset.

        :param data: The data.
        :param window_size: The window size.
        :param pointer_size: The pointer size.
        :param agg_func: The aggregation function.
        :param trim_window: Whether to trim the window.
        :return: The generated data.
        """
        X = torch.randint(0, 10, (n_samples, self.hparams.pointer_size + 10 ** self.hparams.pointer_size), generator=generator)
        pointer_val = (X[:, :self.hparams.pointer_size] * 10 ** torch.arange(self.hparams.pointer_size - 1, -1, -1)).sum(dim=1)
        pointer_val = pointer_val + self.hparams.pointer_size
        y = torch.zeros(n_samples)
        agg_func = self.func_dict[self.hparams.agg_func]
        # TODO: make the implimentation more efficient
        if self.hparams.trim_window:
            for i in tqdm(range(n_samples)):
                y[i] = agg_func(X[i, pointer_val[i]:pointer_val[i] + self.hparams.window_size])
        else:
            X_cat = torch.cat([X, X[:, self.hparams.pointer_size:]], dim=1)
            for i in tqdm(range(n_samples)):
                y[i] = agg_func(X_cat[i, pointer_val[i]:pointer_val[i] + self.hparams.window_size])

        return TensorDataset(X, y.long())
        

        

# @hydra.main(version_base="1.3", config_path="../../configs/data", config_name="pvr.yaml")
# def main(cfg: DictConfig) -> Optional[float]:
#     data_module = PVRDataModule(**cfg)
#     data_module.setup()
#     dl = data_module.train_dataloader()
#     for x, y in dl:
#         print(x, y)
#         break

# if __name__ == "__main__":
#     main()
