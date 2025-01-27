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


class TextRecommender(Recommender):
    """
    Recommender for Text models.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.modality = "text"
        self.MetafeatureClass = modlee.data_metafeatures.TextDataMetafeatures

    def calculate_metafeatures(self, dataloader, *args, **kwargs):
        return super().calculate_metafeatures(
            dataloader,
            data_metafeature_cls=modlee.data_metafeatures.TextDataMetafeatures,
        )

    def fit(self, dataloader, *args, **kwargs):
        """
        Fit the recommended to an Text dataloader.

        :param dataloader: The dataloader, should contain Text input as the first batch element.
        """
        super().fit(dataloader, *args, **kwargs)
        assert self.metafeatures is not None
        if hasattr(self, 'num_classes'):
            self.metafeatures.update({"num_classes": self.num_classes})
        elif hasattr(self, 'vocab_size'):
            self.metafeatures.update({"vocab_size": self.vocab_size})

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
            #Check if these are needed, might be reduntant
            self.code_text = self.get_code_text() 
            self.model_code = modlee_converter.onnx_text2code(self.model_text)

            self.write_file(self.model_text, "./model.txt")
            self.write_file(self.model_code, "./model.py")

        except Exception as e:
            logging.error(
                f"TextRecommender.fit failed, could  not return a recommended model, defaulting model to None"
            )
            self.model = None


class TextClassificationRecommender(TextRecommender):
    """
    Recommender for Text classification tasks.
    Uses cross-entropy loss.
    """

    def __init__(self, num_classes=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task = "classification"
        self.loss_fn = F.cross_entropy
        # Check if num_classes is set, raise ValueError with a professional error message
        if num_classes is None:
            raise ValueError("recommender.fit: num_classes must be provided when using for modality='text', task='classification'.")
        self.num_classes = num_classes

class TextRegressionRecommender(TextRecommender):
    """
    Recommender for Text regression tasks.
    Uses cross-entropy loss.
    """

    def __init__(self, num_classes=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task = "regression"
        self.loss_fn = F.mse_loss
        self.num_classes = 1

class TextTexttotextRecommender(TextRecommender):
    """
    Recommender for Text TextToText tasks.
    Uses cross-entropy loss.
    """

    def __init__(self, vocab_size=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task = "texttotext"
        self.loss_fn = CustomLoss()
        if vocab_size is None:
            raise ValueError("recommender.fit: vocab_size must be provided for modality='text', task='texttotext'.")
        
        self.vocab_size = vocab_size

import torch.nn as nn

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, y_out, y):
        try:
            loss = self.criterion(y_out, y)
        except:
            if not y_out.requires_grad:
                y_out.requires_grad_(True)
            loss = self.criterion(y_out.permute(0, 2, 1), y)
        return loss

