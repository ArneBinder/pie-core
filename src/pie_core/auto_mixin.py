from typing import Generic, Type, TypeVar

from pie_core.hf_hub_mixin import PieBaseHFHubMixin
from pie_core.registrable import Registrable


class PieAutoHFHubMixin(Registrable, PieBaseHFHubMixin):
    """Base class for auto classes."""

    pass


TBase = TypeVar("TBase", bound=PieBaseHFHubMixin)


class AutoMixin(PieBaseHFHubMixin, Generic[TBase]):
    """Mixin for auto classes."""

    base_class: Type[TBase]

    @classmethod
    def from_config(cls, config: dict, **kwargs) -> TBase:  # type: ignore
        """Build a model from a config dict."""
        config = config.copy()
        class_name = config.pop(cls.config_type_key)
        # the class name may be overridden by the kwargs
        class_name = kwargs.pop(cls.config_type_key, class_name)
        clazz = cls.base_class.by_name(class_name)
        return clazz._from_config(config, **kwargs)
