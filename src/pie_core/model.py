import logging
from typing import Any, Dict

from pytorch_lightning.core.mixins import HyperparametersMixin

from pie_core.auto import Auto
from pie_core.hf_hub_mixin import PieModelHFHubMixin
from pie_core.registrable import Registrable

logger = logging.getLogger(__name__)


class Model(PieModelHFHubMixin, HyperparametersMixin, Registrable["Model"]):

    def _config(self) -> Dict[str, Any]:
        config = super()._config() or {}
        if self.has_base_class():
            config[self.config_type_key] = self.base_class().name_for_object_class(self)
        else:
            logger.warning(
                f"{self.__class__.__name__} does not have a base class. It will not work"
                " with AutoModel.from_pretrained() or"
                " AutoModel.from_config(). Consider to annotate the class with"
                " @Model.register() or @Model.register(name='...') to register it at as a Model"
                " which will allow to load it via AutoModel."
            )
        # add all hparams
        config.update(self.hparams)
        return config


class AutoModel(PieModelHFHubMixin, Auto[Model]):

    BASE_CLASS = Model
