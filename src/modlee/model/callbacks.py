from functools import partial
import pickle
from typing import Any, Optional
import numpy as np

import os
import inspect
import pandas as pd
import torch
from torch.utils.data import Dataset

import lightning.pytorch as pl
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT

import modlee
from modlee import (
    data_metafeatures,
    save_run,
    get_code_text_for_model,
    save_run_as_json,
    logging,
)
from modlee import (
    utils as modlee_utils,
    model_metafeatures as mmf,
    data_metafeatures as dmf,
)
from modlee.converter import Converter
from modlee.utils import _make_serializable

modlee_converter = Converter()

from modlee.config import TMP_DIR, MLRUNS_DIR

import mlflow
import shutil

import warnings

base_lightning_module = LightningModule()
base_lm_keys = list(LightningModule.__dict__.keys())


class ModleeCallback(Callback):
    """
    Base class for Modlee-specific callbacks.
    """

    def __init__(self) -> None:
        super().__init__()

    def get_input(self, trainer, pl_module, _dataloader=None):
        """
        Get an input (one element from a batch) from a trainer's dataloader.

        :param trainer: The trainer with the dataloader.
        :param pl_module: The model module, used for loading the data input to the correct device.
        :return: An input from the batch.
        """
        if _dataloader is None:
            _dataloader = trainer.train_dataloader
        _batch = next(iter(_dataloader))
        # NOTE - how can we generalize to different input schemes?
        # e.g. siamese network with multiple inputs
        # Right now, this makes the assumption that that only the network
        # uses only the first element
        if type(_batch) in [list, tuple]:
            # Get the number of inputs based on the model's signature
            ## Note: this broke in scenario where n_inputs=1
            ## Edge case: The dataloader returns a list of tensor of length 1 while the model expects a single tensor

            n_inputs = len(inspect.signature(pl_module.forward).parameters)
            if n_inputs == 1:
                _input = _batch[0]
                _input = _input.to(pl_module.device)
            else:
                _input = _batch[:n_inputs]
                _input = [i.to(pl_module.device) for i in _input]
        else:
            _input = _batch
            _input = [i.to(pl_module.device) for i in _input]
            # print(_batch[0].shape)
            # _batch = torch.Tensor(_batch[0])

        # try:
        #     _input = _input.to(pl_module.device)
        # except:
        #     pass
        return _input
        #return tuple(_input)


class PushServerCallback(Callback):
    """
    Callback to push run assets to the server at the end of training.
    """

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # save_run(pl_module.run_path)
        save_run_as_json(pl_module.run_path)
        return super().on_fit_end(trainer, pl_module)


class LogParamsCallback(Callback):
    """
    Callback to log parameters at the start of training.
    """

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        mlflow.log_param("batch_size", trainer.train_dataloader.batch_size)
        return super().on_train_start(trainer, pl_module)


class LogCodeTextCallback(ModleeCallback):
    """
    Callback to log the model as code and text.
    """

    def __init__(self, kwargs_to_cache={}, *args, **kwargs):
        """
        Constructor for LogCodeTextCallback.

        :param kwargs_to_cache: A dictionary of kwargs to cache in the run for rebuilding the model.
        """
        Callback.__init__(self, *args, **kwargs)
        self.kwargs_cache = kwargs_to_cache

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        # log the code text as a python file
        # self._log_code_text(trainer=trainer, pl_module=pl_module)
        return super().setup(trainer, pl_module, stage)

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        logging.info(
            "Logging model as code (model_graph.py) and text (model_graph.txt)..."
        )
        self._log_code_text(trainer=trainer, pl_module=pl_module)

        return super().on_train_start(trainer, pl_module)

    def _log_code_text(self, trainer: Trainer, pl_module: LightningModule):
        """
        Log the model as code and text. Converts the model through modlee.converter pipelines.

        :param trainer: The trainer that contains the dataloader.
        :param pl_module: The model as a module.
        """
        # _get_code_text_for_model = getattr(modlee, "get_code_text_for_model", None)
        _get_code_text_for_model = get_code_text_for_model
        code_text = ""
        # return
        if _get_code_text_for_model is not None:
            # ==== METHOD 1 ====
            # Save model as code using parsing
            code_text = get_code_text_for_model(pl_module, include_header=True)
            mlflow.log_text(code_text, "model.py")
            # Save variables required to rebuild the model
            pl_module._update_kwargs_cached()
            mlflow.log_dict(self.kwargs_cache, "cached_vars")

            # ==== METHOD 2 ====
            # Save model as code by converting to a graph through ONNX
            #### Implement the same logic for try and except error from get_input
            input_dummy = self.get_input(trainer, pl_module)
            onnx_model = modlee_converter.torch2onnx(pl_module, input_dummy=input_dummy)
            onnx_text = modlee_converter.onnx2onnx_text(onnx_model)
            mlflow.log_text(onnx_text, "model_graph.txt")
            torch_graph_code = modlee_converter.onnx_text2code(onnx_text)
            mlflow.log_text(torch_graph_code, "model_graph.py")

            # Save model size
            model_size = modlee_utils.get_model_size(pl_module, as_MB=False)
            mlflow.log_text(str(model_size), "model_size")

        else:
            logging.warning(
                "Could not access model-text converter, \
                    not logging but continuing experiment"
            )


class ModelMetafeaturesCallback(ModleeCallback):
    def __init__(
        self,
        ModelMetafeatures=mmf.ModelMetafeatures,
    ):
        super().__init__()
        self.ModelMetafeatures = ModelMetafeatures

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        logging.info("Logging model metafeatures...")
        super().on_train_start(trainer, pl_module)
        # TODO - need to select the Metafeature module based on modality, task,
        # Same with data metafeatures
        _input = self.get_input(trainer, pl_module)

        #print(pl_module.device)
        #print([i.device for i in _input])

        model_mf = self.ModelMetafeatures(pl_module, _input)
        mlflow.log_dict(
            {**model_mf.properties, 
             #**model_mf.embedding
             },
              "model_metafeatures"
        )

        #print(pl_module.device)
        #print([i.device for i in _input])


class LogONNXCallback(ModleeCallback):
    """
    Callback for logging the model in its ONNX representations.
    Deprecated, will be combined with LogCodeTextCallback.
    """

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        # self._log_onnx(trainer, pl_module)
        return super().setup(trainer, pl_module, stage)

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._log_onnx(trainer, pl_module)
        return super().on_fit_start(trainer, pl_module)

    def _log_onnx(self, trainer, pl_module):
        # train_input,_ = next(iter(trainer.train_dataloader))
        # print(train_input)
        # NOTE assumes that model input is the first output of a batch
        modlee_utils.safe_mkdir(TMP_DIR)
        data_filename = f"{TMP_DIR}/model.onnx"

        _input = self.get_input(trainer, pl_module)

        model_output = pl_module.forward(_input)
        torch.onnx.export(
            pl_module,
            # train_input,
            _input,
            data_filename,
            export_params=False,
        )
        mlflow.log_artifact(data_filename)

        pass


class LogOutputCallback(Callback):
    """
    Callback to log the output metrics for each batch.
    """

    def __init__(self, *args, **kwargs):
        """
        Constructor for LogOutputCallback.
        """
        Callback.__init__(self, *args, **kwargs)
        self.on_train_batch_end = partial(self._on_batch_end, phase="train")
        self.on_validation_batch_end = partial(self._on_batch_end, phase="val")
        self.outputs = {"train": [], "val": []}

    def _on_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        phase="train",
    ) -> None:
        """
        Helper function to log output metrics on batch end.
        Currently catches metrics formatted as '{phase}_loss'.

        :param trainer: The trainer.
        :param pl_module: The model as a module.
        :param outputs: The outputs on batch end, automatically passed by the base callback.
        :param batch: The batch, automatically passed by the base callback.
        :param batch_idx: The index of the batch, automatcally passed by te base callback.
        :param phase: The phase of training for logging, ["train", "val"]. Defaults to "train".
        """
        if trainer.is_last_batch:
            if isinstance(outputs, dict):
                for output_key, output_value in outputs.items():
                    pl_module.log(output_key, output_value)
            elif isinstance(outputs, list):
                for output_idx, output_value in outputs:
                    pl_module.log(f"{phase}_step_output_{output_idx}", output_value)
            elif outputs is not None:
                pl_module.log(f"{phase}_loss", outputs)
            self.outputs[phase].append(outputs)
        return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)


class DataMetafeaturesCallback(ModleeCallback):
    """
    Callback to calculate and log data meta-features.
    """

    def __init__(
        self,
        data_snapshot_size=1e7,
        DataMetafeatures=dmf.DataMetafeatures,
        *args,
        **kwargs,
    ):
        """
        Constructor for the data metafeature callback.

        :param data_snapshot_size: The maximum size of the cached data snapshot.
        :param DataMetafeatures: The DataMetafeatures module. If not provided, will not calculate metafeatures.
        """
        Callback.__init__(self, *args, **kwargs)
        super().__init__()
        self.data_snapshot_size = data_snapshot_size
        self.DataMetafeatures = DataMetafeatures
        self.dataloader = None

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:

        if hasattr(trainer.model, 'is_recommended_modlee_model'):
            self._log_data_mfe_recommender(trainer)

        else:
            self._log_data_metafeatures_dataloader(trainer.train_dataloader)


        self._log_output_size(trainer, pl_module)

        return super().on_train_start(trainer, pl_module)

    def _log_data_mfe_recommender(self, trainer) -> None:
        """
        Log data metafeatures with a dataloader.

        :param dataloader: The dataloader.
        """
        if trainer.model.data_mfe:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)

                logging.info(f"Logging data metafeatures with {self.DataMetafeatures} from recommender analysis")
                
                mlflow.log_dict(_make_serializable(trainer.model.data_mfe), "data_metafeatures")
        else:
            logging.warning("Cannot log data from recommender analysis, running calculations again")
            self._log_data_metafeatures_dataloader(trainer.train_dataloader)
    
    def _log_data_metafeatures(self, data, targets=[]) -> None:
        """
        Log the data metafeatures from input data and targets.
        Deprecated in favor of _log_data_metafeatures_dataloader.

        :param data: The input data.
        :param targets: The targets.
        """
        if self.DataMetafeatures:
            if isinstance(data, torch.Tensor):
                data, targets = data.numpy(), targets.numpy()
            data_metafeatures = self.DataMetafeatures(x=data, y=targets)
            mlflow.log_dict(data_metafeatures.data_metafeatures, "data_metafeatures")
        else:
            logging.warning(
                "Could not access data statistics calculation from server, \
                    not logging but continuing experiment"
            )

    def _log_data_timeseriesmetafeatures(self, data) -> None:
        """
        Log time series data meta features with a pandas dataframe.
        :param dataframe: The pandas dataframe.
        """
        if self.TimeseriesDMF:
            data_mf = self.TimeseriesDMF(data)
            mlflow.log_dict(data_mf.properties, "data_metafeatures")
        else:
            logging.warning("Cannot log data statistics, could not access from server")

    def _log_data_metafeatures_dataloader(self, dataloader) -> None:
        """
        Log data metafeatures with a dataloader.

        :param dataloader: The dataloader.
        """
        if self.DataMetafeatures:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)

                logging.info(f"Logging data metafeatures with {self.DataMetafeatures}")
                
                # Pass the dataloader to the TimeseriesDataMetafeatures
                data_mf = self.DataMetafeatures(dataloader=dataloader)  # Ensure dataloader is passed
                
                #mlflow.log_dict(data_mf.stats_rep, "stats_rep")
                data_mf_dict = {
                    **data_mf.properties,
                    **data_mf.mfe,
                }
                # attrs = data_mf_dict.keys()
                # logging.info(f"Logged data metafeatures: {','.join(attrs)}")
                mlflow.log_dict(_make_serializable(data_mf_dict), "data_metafeatures")
        else:
            logging.warning("Cannot log data statistics, could not access from server")

    def _log_output_size(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """
        Log the output size of the model.

        :param trainer: The trainer.
        :param pl_module: The model as a module.
        """

        _input = self.get_input(trainer, pl_module)
        
        try:
            if isinstance(_input, tuple):
                _output = pl_module.forward(*_input)
            else:
                _output = pl_module.forward(_input)
            output_shape = list(_output.shape[1:])
            mlflow.log_param("output_shape", output_shape)
        except:
            ### Not: Add another try statement if lenght of input is 1 and model expects a single tensor pass one single tensor
            try:
                _output = pl_module.forward(_input[0])
                output_shape = list(_output.shape[1:])
                mlflow.log_param("output_shape", output_shape)
            except:
                logging.warning(
                    "Cannot log output shape, could not pass batch through network"
                )

    def _get_data_targets(self, trainer: Trainer):
        """
        Get the data and targets from a trainer's dataloader.

        :param trainer: The trainer.
        :return: The data and targets.
        """
        _dataset = trainer.train_dataloader.dataset
        if isinstance(_dataset, list):
            data = np.array(_dataset)
        elif isinstance(_dataset, torch.utils.data.dataset.IterableDataset):
            data = list(_dataset)
        else:
            if isinstance(_dataset, torch.utils.data.Subset):
                _dataset = _dataset.dataset
            data = _dataset.data
            if isinstance(data, torch.Tensor):
                data = data.numpy()

        self._save_snapshot(data, "data")

        targets = []
        if hasattr(_dataset, "targets"):
            targets = _dataset.targets
            self._save_snapshot(targets, "targets", max_len=len(data))
        return data, targets

    def _save_snapshot(self, data, snapshot_name="data", max_len=None):
        """
        Save a snapshot of data.

        :param data: The data to save.
        :param snapshot_name: The name to save the data.
        :param max_len: The maximum length of the data.
        """
        data = self._get_snapshot(data=data, max_len=max_len)
        modlee_utils.safe_mkdir(TMP_DIR)
        data_filename = f"{TMP_DIR}/{snapshot_name}_snapshot.npy"
        np.save(data_filename, data)
        mlflow.log_artifact(data_filename)

    def _get_snapshot(self, data, max_len=None):
        """
        Get a snapshot of data.

        :param data: The data.
        :param max_len: The maximum length of the snapshot.
        :return: A snapshot of the data.
        """
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        elif not isinstance(data, np.ndarray):
            data = np.array(data)
        if max_len is None:
            data_size = data.nbytes
            # take a slice that should be no larger than 10MB
            max_len = int(
                np.min([(len(data) * self.data_snapshot_size) // data_size, len(data)])
            )
        return data[:max_len]

    def _save_snapshots_batched(self, data_snapshots):
        """
        Save batches of data snapshots.

        :param data_snapshots: A batch of data snapshots.
        """
        modlee_utils.safe_mkdir(TMP_DIR)
        for snapshot_idx, data_snapshot in enumerate(data_snapshots):
            data_filename = f"{TMP_DIR}/snapshot_{snapshot_idx}.npy"
            np.save(data_filename, data_snapshot)
            mlflow.log_artifact(data_filename)

    def _get_snapshots_batched(self, dataloader, max_len=None):
        """
        Get a batch of data snapshots.

        :param dataloader: The dataloader of the data to snapshot.
        :param max_len: The maximum length of the snapshot.
        :return: A batch of data snapshots.
        """
        # Use batch to determine how many "sub"batches to create
        _batch = next(iter(dataloader))

        data_snapshot_size = self.data_snapshot_size
        if type(_batch) in [list, tuple]:
            n_snapshots = len(_batch)
        else:
            n_snapshots = 1
        data_snapshots = [np.array([])] * n_snapshots

        # Keep appending to batches until the combined size reaches the limit
        batch_ctr = 0
        while np.sum([ds.nbytes for ds in data_snapshots]) < data_snapshot_size:
            _batch = next(iter(dataloader))

            # If there are multiple elements in the batch,
            # append to respective subbranches
            if type(_batch) in [list, tuple]:
                for batch_idx, _subbatch in enumerate(_batch):
                    if str(_subbatch.device) != "cpu":
                        _subbatch = _subbatch.cpu()
                    if data_snapshots[batch_idx].size == 0:
                        data_snapshots[batch_idx] = _subbatch.numpy()
                    else:
                        data_snapshots[batch_idx] = np.vstack(
                            [data_snapshots[batch_idx], (_subbatch.numpy())]
                        )
            else:
                if data_snapshots[0].size == 0:
                    data_snapshots[0] = _batch.numpy()
                else:
                    data_snapshots[0] = np.vstack([data_snapshots[0], (_batch.numpy())])
            batch_ctr += 1
        return data_snapshots


class LogTransformsCallback(ModleeCallback):
    """
    Logs transforms applied to the dataset, if applied with torchvision.transforms
    """

    def on_train_start(self, trainer, pl_module):
        dataset = trainer.train_dataloader.dataset
        if hasattr(dataset, "transforms"):
            mlflow.log_text(str(dataset.transform), "transforms.txt")


class LogModelCheckpointCallback(pl.callbacks.ModelCheckpoint):
    """
    Callback to log the best performing model in a training routine based on Loss value
    """

    def __init__(
        self,
        monitor="val_loss",
        filename="model_checkpoint",
        temp_dir_path=f"{TMP_DIR}/checkpoints",
        save_top_k=1,
        mode="min",
        verbose=True,
        *args,
        **kwargs,
    ):
        """
        Constructor for LogModelCheckpointCallback. Extends standard pl ModelCheckpoint callback.

        :param monitor: The metric to monitor for saving the best model checkpoint.
        :param filename: Template for checkpoint filenames.
        :param dirpath: The directory to save the temporarily generated checkpoint files.
        :param save_top_k: The number of best model checkpoints to save.
        :param mode: One of {'min', 'max'}.
        :param verbose: Whether to print verbose messages.
        """
        modlee_utils.safe_mkdir(temp_dir_path)
        super().__init__(
            filename=f"{filename}_{monitor}",
            dirpath=temp_dir_path,
            monitor=monitor,
            save_top_k=save_top_k,
            mode=mode,
            verbose=verbose,
            *args,
            **kwargs,
        )
        self.monitor = monitor
        self.temp_dir_path = temp_dir_path

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)
        if self.monitor == "val_loss":
            self.log_and_clean_checkpoint(trainer, pl_module, "val")
        else:
            self.log_and_clean_checkpoint(trainer, pl_module, "train")

    def log_and_clean_checkpoint(self, trainer, pl_module, step):
        """
        Log the latest checkpoint to MLflow and remove the local file.

        :param trainer: The PyTorch Lightning `Trainer` instance.
        :param pl_module: The LightningModule being trained.
        :param step: The step ('train' or 'val') that triggered the checkpoint.
        """
        checkpoint_path = self.best_model_path

        # Log the checkpoint file to MLflow
        if checkpoint_path:
            mlflow.log_artifact(checkpoint_path, artifact_path=f"checkpoints/{step}")
            # Get the current epoch and metric value
            current_epoch = trainer.current_epoch
            metric_value = trainer.callback_metrics.get(self.monitor)

            # Ensure the metric value is a float

            if not isinstance(metric_value, float):
                metric_value = float(metric_value)
            # Log the epoch and metric value
            mlflow.log_metrics(
                {
                    f"best_{self.monitor}_epoch": current_epoch,
                    f"best_{self.monitor}_value": metric_value,
                }
            )

    def on_fit_end(self, trainer, pl_module):
        super().on_fit_end(trainer, pl_module)

        # Cleaning up temp_directory
        if os.path.exists(self.temp_dir_path):
            shutil.rmtree(self.temp_dir_path)
        # else:
        #     print(f"Directory {self.temp_dir_path} does not exist.")

class LogModalityTaskCallback(ModleeCallback):
    """
    Logs the modality and task
    """
    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        mlflow.log_text(pl_module.modality, "modality")
        mlflow.log_text(pl_module.task, "task")
        return super().on_train_start(trainer, pl_module)

class LossLoggingCallback(pl.Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """
        Log training loss at the end of each training batch and provide guidance if output is misconfigured.
        """
        # Ensure the output is a dictionary and contains 'loss'
        if not isinstance(outputs, dict) or 'loss' not in outputs:
            logging.error("'training_step' output must be a dictionary with a 'loss' key.")
            logging.error("Example of correct 'training_step' return:\n")
            logging.error(
                """def training_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self(x)
    loss = self.criterion(y_hat, y)
    return {'loss': loss}"""
            )
        else:
            # Access the loss from the outputs of the training step
            loss = outputs["loss"]
            # Log the loss (using the trainer's logger)
            trainer.logger.log_metrics({"loss": loss.item()}, step=trainer.global_step)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """
        Log validation loss at the end of each validation batch and provide guidance if output is misconfigured.
        """
        # Ensure the output is a dictionary and contains 'val_loss'
        if not isinstance(outputs, dict) or 'val_loss' not in outputs:
            logging.error("'validation_step' output must be a dictionary with a 'val_loss' key.")
            logging.error("Example of correct 'validation_step' return:\n")
            logging.error(
                """def validation_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self(x)
    val_loss = self.criterion(y_hat, y)
    return {'val_loss': val_loss}"""
            )
        else:
            # Access the validation loss from the outputs of the validation step
            val_loss = outputs["val_loss"]
            # Log the validation loss (using the trainer's logger)
            trainer.logger.log_metrics({"val_loss": val_loss.item()}, step=trainer.global_step)


class LRSchedulerCallback(Callback):
    def __init__(self, patience=5, factor=0.1, min_lr=1e-6):
        """
        Custom callback for reducing the learning rate when a metric plateaus.

        :param patience: Number of epochs with no improvement after which LR will be reduced.
        :param factor: Factor by which the learning rate will be reduced.
        :param min_lr: Minimum learning rate after which it cannot be reduced further.
        """
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.wait = 0
        self.best_loss = float('inf')
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """
        This hook is called after the validation epoch ends. It checks whether the 
        validation loss has improved and adjusts the learning rate if necessary.
        """
        # Get the current validation loss from the trainer's logged metrics
        current_loss = trainer.callback_metrics.get("val_loss")
        
        # If validation loss is None, we cannot proceed
        if current_loss is None:
            return
        
        # If the loss improved, reset the wait counter
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1

        # If the wait counter exceeds the patience, reduce the learning rate
        if self.wait >= self.patience:
            for optimizer in trainer.optimizers:
                for param_group in optimizer.param_groups:
                    new_lr = param_group['lr'] * self.factor
                    if new_lr >= self.min_lr:
                        print(f"Reducing learning rate to {new_lr}")
                        param_group['lr'] = new_lr
            self.wait = 0  # Reset the wait counter after changing the LR

import os
import copy
import torch
from lightning.pytorch.callbacks import Callback


class CustomModelCheckpoint(Callback):
    def __init__(self, monitor="loss", save_top_k=1, mode="min", verbose=True, dirpath=None, filename="best_model"):
        """
        Custom callback for saving the model checkpoint when the monitored metric improves.

        :param monitor: Metric to monitor (e.g., "val_loss" for validation or "loss" for training).
        :param save_top_k: Number of top models to save. Currently supports saving the best one.
        :param mode: "min" for minimizing the metric, "max" for maximizing it.
        :param verbose: If True, logs messages when the model is saved.
        :param dirpath: Directory where to save the checkpoints.
        :param filename: Filename for the checkpoint.
        """
        self.monitor = monitor
        self.save_top_k = save_top_k
        self.mode = mode
        self.verbose = verbose
        self.dirpath = dirpath or "./checkpoints"
        self.filename = filename
        self.best_score = None
        self.best_model_weights = None
        self.first_pass = True

        if not os.path.exists(self.dirpath):
            os.makedirs(self.dirpath)

        # Set the comparison function based on mode
        if mode == "min":
            self.compare_fn = lambda current, best: current < best
            self.best_score = float('inf')
        elif mode == "max":
            self.compare_fn = lambda current, best: current > best
            self.best_score = -float('inf')
        else:
            raise ValueError("mode must be either 'min' or 'max'")

    def _check_and_save_model(self, trainer, pl_module, epoch_type="validation"):
        """
        Common logic to check if the model has improved and save it. 
        Handles both validation and training epochs.
        
        :param trainer: The PyTorch Lightning trainer object.
        :param pl_module: The PyTorch Lightning module (the model).
        :param epoch_type: Either "validation" or "training" to indicate which epoch type is being checked.
        """
        current_score = trainer.callback_metrics.get(self.monitor)
        if self.first_pass == False:
            # print(f"{epoch_type.capitalize()} Epoch - Monitoring {self.monitor}")
            # print('best_score: ', self.best_score)
            # print('current_score: ', current_score)

            if current_score is None:
                if self.verbose:
                    print(f"Warning: '{self.monitor}' metric is not available for checkpointing.")
                return
        else:
            #skip first pass, usually
            self.first_pass = False
            return
        
        if self.compare_fn(current_score, self.best_score):
            self.best_score = current_score
            self.best_model_weights = copy.deepcopy(pl_module.state_dict())

            # Save the checkpoint to a file
            checkpoint_path = os.path.join(self.dirpath, f"{self.filename}_{epoch_type}.ckpt")
            torch.save(self.best_model_weights, checkpoint_path)

            if self.verbose:
                print(f"Model improved. Saving model to {checkpoint_path} with {self.monitor}: {self.best_score}")

    def on_validation_epoch_end(self, trainer, pl_module):
        """
        This hook is called at the end of each validation epoch. Checks and saves the model if necessary.
        """
        # Only run the checkpointing logic if monitoring 'val_loss'
        if "val" in self.monitor:
            self._check_and_save_model(trainer, pl_module, epoch_type="validation")

    def on_train_epoch_end(self, trainer, pl_module):
        """
        This hook is called at the end of each training epoch. Checks and saves the model if necessary.
        """
        # Only run the checkpointing logic if monitoring 'loss'
        if "loss" in self.monitor and "val" not in self.monitor:
            self._check_and_save_model(trainer, pl_module, epoch_type="training")

    def on_train_end(self, trainer, pl_module):
        """
        This hook is called at the end of training. Ensures the best model is saved 
        and reinitializes the model with the best weights if necessary.
        """
        if self.best_model_weights is not None:
            # Check if the current model is the best model
            current_state_dict = pl_module.state_dict()
            is_best_model = all(torch.equal(current_state_dict[k], self.best_model_weights[k])
                                for k in current_state_dict.keys())

            if not is_best_model:
                # Reinitialize the model with the best weights
                pl_module.load_state_dict(self.best_model_weights)
                if self.verbose:
                    print("Reinitialized model with the best weights after training.")

            # Save the best model one last time
            final_path = os.path.join(self.dirpath, f"{self.filename}_final.ckpt")
            torch.save(self.best_model_weights, final_path)

            if self.verbose:
                print(f"Training completed. Best model saved to {final_path} with {self.monitor}: {self.best_score}")
        else:
            if self.verbose:
                print("No model improvement during training.")