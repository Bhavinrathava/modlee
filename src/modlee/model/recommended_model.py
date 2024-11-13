from .model import *  # Assuming ModleeModel is your default parent
import torch.nn.functional as F
import torch
from torch.optim import AdamW

class RecommendedModelFactory:
    """
    A factory class to create an instance of RecommendedModel with a dynamically determined
    parent class based on the specified modality and task.
    """
    
    def __init__(self, modality, task, model, loss_fn=F.cross_entropy, *args, **kwargs):
        """
        Initialize the RecommendedModelFactory with modality and task.
        
        Parameters:
            modality (str): The modality type (e.g., "tabular", "image").
            task (str): The task type (e.g., "classification", "regression").
            model (torch.nn.Module): The model to be wrapped in the recommended model.
            loss_fn (function): The loss function to be used for training.
        """
        self.modality = modality
        self.task = task
        self.model = model
        self.loss_fn = loss_fn
        self.args = args
        self.kwargs = kwargs
        self.recommended_model = self._create_recommended_model()

    def _create_recommended_model(self):
        """
        Private method to dynamically create the recommended model class based on modality and task.
        """
        # Construct the specific parent class name, e.g., TabularClassificationModleeModel
        specific_model_class_name = f"{self.modality.capitalize()}{self.task.capitalize()}ModleeModel"
        
        # Retrieve the specific model class, defaulting to ModleeModel if not found
        ParentClass = globals().get(specific_model_class_name, ModleeModel)

        class RecommendedModel(ParentClass):
            """
            A ready-to-train ModleeModel that wraps around a recommended model from the recommender.
            Contains default functions for training.
            """
            def __init__(self, model, loss_fn, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.model = model
                self.loss_fn = loss_fn
                self.is_recommended_modlee_model = True
                self.data_mfe = None

            def forward(self, x):
                return self.model(x)

            def training_step(self, batch, batch_idx, *args, **kwargs):
                x, y = batch
                y_out = self(x)
                loss = self.loss_fn(y_out, y)
                return {"loss": loss}

            def validation_step(self, val_batch, batch_idx, *args, **kwargs):
                x, y = val_batch
                y_out = self(x)
                loss = self.loss_fn(y_out, y)
                return {"val_loss": loss}

            def configure_optimizers(self):
                """
                Configure a default AdamW optimizer with learning rate decay.
                """
                optimizer = AdamW(self.parameters(), lr=0.001)
                return optimizer

            def configure_callbacks(self):
                base_callbacks = super().configure_callbacks()
                return base_callbacks

        # Instantiate and return the RecommendedModel
        return RecommendedModel(model=self.model, loss_fn=self.loss_fn, *self.args, **self.kwargs)

    def get_model(self):
        """
        Returns the created recommended model instance.
        """
        return self.recommended_model