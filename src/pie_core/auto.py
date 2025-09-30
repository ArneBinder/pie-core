from typing import Generic, Protocol, Type, TypeVar

from pie_core.hf_hub_mixin import HFHubMixin, HFHubProtocol
from pie_core.registrable import Registrable, RegistrableProtocol


class RegistrableBaseHFHubProtocol(RegistrableProtocol, HFHubProtocol, Protocol):
    pass


T = TypeVar("T", bound=RegistrableBaseHFHubProtocol)


class Auto(HFHubMixin, Registrable[T], Generic[T]):

    @classmethod
    def from_config(cls, config: dict, **kwargs) -> T:  # type: ignore
        config = config.copy()
        class_name = config.pop(cls.config_type_key, None)
        # The class name may be overridden by the kwargs.
        class_name = kwargs.pop(cls.config_type_key, class_name)
        if class_name is None:
            raise ValueError(
                f"Missing required key {cls.config_type_key!r} to select a concrete "
                f"{cls.base_class().__name__} for {cls.__name__}. Provide it in the config "
                f"or pass it as a keyword argument ({cls.config_type_key}=...). Received "
                f"config: {config!r}"
            )
        # The returned class should not be an instance of Auto
        # which introduces a return type mismatch. This is fine,
        # so we ignore it (see method signature).
        clazz: Type[T] = cls.base_class().by_name(class_name)
        return clazz._from_config(config, **kwargs)
