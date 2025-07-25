""" 
Modlee model for tabular. 
"""
from modlee.model import ModleeModel

class TabularClassificationModleeModel(ModleeModel):
    """
    Subclass of ModleeModel with tabular-classification-specific convenience wrappers
    - Calculates data-specific data statistics
    """

    def __init__(self, *args, **kwargs):
        """
        TabularClassificationModleeModel constructor.

        """
        modality = 'tabular'
        task = "classification"
        vars_cache = {"modality":modality,"task": task}
        ModleeModel.__init__(
            self,
            kwargs_cache=vars_cache, *args, **kwargs
        )

class TabularRegressionModleeModel(ModleeModel):
    """
    Subclass of ModleeModel with tabular-regression-specific convenience wrappers
    - Calculates data-specific data statistics
    """

    def __init__(self, *args, **kwargs):
        """
        TabularRegressionModleeModel constructor.

        """
        modality = 'tabular'
        task = "regression"
        vars_cache = {"modality":modality,"task": task}
        ModleeModel.__init__(
            self,
            kwargs_cache=vars_cache, *args, **kwargs
        )