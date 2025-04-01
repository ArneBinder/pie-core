from typing import Type

from pie_core.hf_hub_mixin import PieModelHFHubMixin, PieTaskModuleHFHubMixin
from pie_core.model import PyTorchIEModel
from pie_core.taskmodule import TaskModule


class AutoModel(PieModelHFHubMixin):

    @classmethod
    def from_config(cls, config: dict, **kwargs) -> PyTorchIEModel:
        """Build a model from a config dict."""
        config = config.copy()
        class_name = config.pop(cls.config_type_key)
        # the class name may be overridden by the kwargs
        class_name = kwargs.pop(cls.config_type_key, class_name)
        clazz = PyTorchIEModel.by_name(class_name)
        return clazz._from_config(config, **kwargs)


class AutoTaskModule(PieTaskModuleHFHubMixin):

    @classmethod
    def from_config(cls, config: dict, **kwargs) -> TaskModule:
        """Build a task module from a config dict."""
        config = config.copy()
        class_name = config.pop(cls.config_type_key)
        # the class name may be overridden by the kwargs
        class_name = kwargs.pop(cls.config_type_key, class_name)
        clazz: Type[TaskModule] = TaskModule.by_name(class_name)
        return clazz._from_config(config, **kwargs)
