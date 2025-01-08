import torch
from modlee.model import ModleeModel

class TextClassificationModleeModel(ModleeModel):
    """
    Subclass of ModleeModel with text-classification-specific convenience wrappers
    - Calculates data-specific data statistics
    """

    def __init__(self, *args, **kwargs):
        """
        TextClassificationModleeModel constructor.

        """
        modality = 'text'
        task = "classification"
        vars_cache = {"modality":modality,"task": task}
        ModleeModel.__init__(
            self,
            kwargs_cache=vars_cache, *args, **kwargs
        )

class TextRegressionModleeModel(ModleeModel):
    """
    Subclass of ModleeModel with text-classification-specific convenience wrappers
    - Calculates data-specific data statistics
    """

    def __init__(self, *args, **kwargs):
        """
        TextClassificationModleeModel constructor.

        """
        modality = 'text'
        task = "regression"
        vars_cache = {"modality":modality,"task": task}
        ModleeModel.__init__(
            self,
            kwargs_cache=vars_cache, *args, **kwargs
        )

class TextTextToTextModleeModel(ModleeModel):
    """
    Subclass of ModleeModel with text-classification-specific convenience wrappers
    - Calculates data-specific data statistics
    """

    def __init__(self, *args, **kwargs):
        """
        TextClassificationModleeModel constructor.

        """
        modality = 'text'
        task = "text_to_text"
        vars_cache = {"modality":modality,"task": task}
        ModleeModel.__init__(
            self,
            kwargs_cache=vars_cache, *args, **kwargs
        )