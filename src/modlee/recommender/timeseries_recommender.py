from modlee.data_metafeatures import DataMetafeatures
from modlee.recommender import Recommender
from modlee.model import RecommendedModelFactory
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import modlee
from modlee.converter import Converter
from modlee.utils import get_model_size, typewriter_print
modlee_converter = Converter()


class TimeseriesRecommender(Recommender):
    """
    Recommender for Timeseries models.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.modality = "timeseries"
        self.MetafeatureClass = modlee.data_metafeatures.TimeseriesDataMetafeatures

    def calculate_metafeatures(self, dataloader, *args, **kwargs):
        return super().calculate_metafeatures(
            dataloader,
            data_metafeature_cls=modlee.data_metafeatures.TimeseriesDataMetafeatures,
        )

    def fit(self, dataloader, *args, **kwargs):
        """
        Fit the recommended to an Timeseries dataloader.

        :param dataloader: The dataloader, should contain Timeseries input as the first batch element.
        """
        super().fit(dataloader, *args, **kwargs)
        assert self.metafeatures is not None
        if hasattr(self, 'prediction_length'):
            self.metafeatures.update({"prediction_length": self.prediction_length})

        try:            
            self.model_text = self._get_model_text(self.metafeatures)

            if not isinstance(self.model_text, str):
                self.model_text = self.model_text.decode("utf-8")
            model = modlee_converter.onnx_text2torch(self.model_text)
            for param in model.parameters():
                try:
                    torch.nn.init.xavier_normal_(param, 1.0)
                except:
                    torch.nn.init.normal_(param)

            model_factory = RecommendedModelFactory(modality = self.modality, task = self.task, model=model, loss_fn=self.loss_fn)
            self.model = model_factory.get_model()

            self.model.data_mfe = self.metafeatures
            self.code_text = self.get_code_text()
            self.model_code = modlee_converter.onnx_text2code(self.model_text)

            self.write_file(self.model_text, "./model.txt")
            self.write_file(self.model_code, "./model.py")

        except Exception as e:
            logging.error(
                f"TimeseriesRecommender.fit failed, could  not return a recommended model, defaulting model to None"
            )
            self.model = None


class TimeseriesForecastingRecommender(TimeseriesRecommender):
    """
    Recommender for time series forecasting tasks.
    Uses cross-entropy loss.
    """

    def __init__(self, prediction_length=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task = "forecasting"
        self.loss_fn = F.cross_entropy
        # Check if prediction_length is set, raise ValueError with a professional error message
        if prediction_length is None:
            raise ValueError("recommender.fit: prediction_length must be provided when using for modality='timeseries', task='forecasting'.")
        self.prediction_length = prediction_length

