from typing import Any, Dict

from pytorch_lightning.core.mixins import HyperparametersMixin

from pie_core.auto import Auto
from pie_core.hf_hub_mixin import PieModelHFHubMixin
from pie_core.registrable import Registrable


class Model(PieModelHFHubMixin, HyperparametersMixin, Registrable["Model"]):

    def _config(self) -> Dict[str, Any]:
        config = super()._config() or {}
        config[self.config_type_key] = self.base_class().name_for_object_class(self)
        # add all hparams
        config.update(self.hparams)
        return config


class AutoModel(PieModelHFHubMixin, Auto[Model]):

    BASE_CLASS = Model
