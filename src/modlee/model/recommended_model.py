from .model import *  # Assuming ModleeModel is your default parent
#from .text_model import *
import torch.nn.functional as F
import torch
from torch.optim import AdamW
import sys

class BaseRecommendedModel(ModleeModel):
    """
    Base class for all recommended models with common functionality.
    """
    def __init__(self, model, loss_fn=F.cross_entropy, *args, **kwargs):
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
        self.modality = modality.lower()
        self.task = task.lower()
        self.model = model
        self.loss_fn = loss_fn
        self.args = args
        self.kwargs = kwargs
        self.recommended_model = self._create_recommended_model()

    def _get_or_create_model_class(self):
        """
        Dynamically creates or retrieves a model class based on modality and task.
        The class is created in the global namespace to ensure picklability.
        """
        # Generate the class name
        class_name = f"{self.modality.capitalize()}{self.task.capitalize()}RecommendedModel"
        
        # Check if the class already exists in the global namespace
        if class_name in globals():
            return globals()[class_name]
        
        # Get the appropriate parent class based on modality and task if it exists
        parent_class_name = f"{self.modality.capitalize()}{self.task.capitalize()}ModleeModel"
        #TODO base recommended model subclasses from pl.lightening, but parent_class_name does not. 
        #might lead to bugs in the future. 
        ParentClass = globals().get(parent_class_name, BaseRecommendedModel)
        
        # Create the new class dynamically
        new_class = type(
            class_name,
            (ParentClass,),
            {
                '__module__': __name__,  # This is important for pickling
                '__doc__': f"Dynamically created recommended model for {self.modality} {self.task}."
            }
        )
        
        # Add the class to the global namespace
        globals()[class_name] = new_class
        
        return new_class

    def _create_recommended_model(self):
        """
        Create the recommended model instance based on modality and task.
        """
        ModelClass = self._get_or_create_model_class()

        return ModelClass(
                model=self.model,
                loss_fn=self.loss_fn,
                *self.args,
                **self.kwargs
            )

    def get_model(self):
        """
        Returns the created recommended model instance.
        """
        return self.recommended_model