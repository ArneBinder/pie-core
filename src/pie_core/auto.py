from typing import Generic, Type, TypeVar

from pie_core import Registrable
from pie_core.hf_hub_mixin import PieBaseHFHubMixin

T = TypeVar("T", bound="Auto")


class Auto(PieBaseHFHubMixin, Registrable[T], Generic[T]):

    @classmethod
    def from_config(cls: Type[T], config: dict, **kwargs) -> T:
        """Build a task module from a config dict."""
        config = config.copy()
        class_name = config.pop(cls.config_type_key)
        # the class name may be overridden by the kwargs
        class_name = kwargs.pop(cls.config_type_key, class_name)
        clazz: Type[T] = cls.base_class().by_name(class_name)
        return clazz._from_config(config, **kwargs)
