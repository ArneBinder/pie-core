from abc import ABC, abstractmethod
from typing import Any, Dict

from pytorch_lightning.core.mixins import HyperparametersMixin

from pie_core.hf_hub_mixin import PieModelHFHubMixin
from pie_core.registrable import Registrable


class Model(PieModelHFHubMixin, HyperparametersMixin, Registrable, ABC):

    def _config(self) -> Dict[str, Any]:
        config = super()._config() or {}
        config[self.config_type_key] = Model.name_for_object_class(self)
        # add all hparams
        config.update(self.hparams)
        return config


class AutoModel(PieModelHFHubMixin):

    @classmethod
    def from_config(cls, config: dict, **kwargs) -> Model:
        """Build a model from a config dict."""
        config = config.copy()
        class_name = config.pop(cls.config_type_key)
        # the class name may be overridden by the kwargs
        class_name = kwargs.pop(cls.config_type_key, class_name)
        clazz = Model.by_name(class_name)
        return clazz._from_config(config, **kwargs)
