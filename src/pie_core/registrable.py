import inspect
from collections import defaultdict
from typing import Callable, Dict, Optional, Protocol, Type, TypeVar


class RegistrationError(Exception):
    pass


T = TypeVar("T", bound="RegistrableProtocol")
T2 = TypeVar("T2", bound="RegistrableProtocol")


class RegistrableProtocol(Protocol[T2]):
    BASE_CLASS: Optional[Type[T2]] = None

    @classmethod
    def base_class(cls) -> Type[T2]:
        """Returns the base class of this registrable class."""
        if cls.BASE_CLASS is None:
            raise RegistrationError(
                f"{cls.__name__} has no defined base class. "
                f"Please annotate {cls.__name__} with @<SOME-PARENT-OF-{cls.__name__}>.register() "
                f"to register it at <SOME-PARENT-OF-{cls.__name__}>."
            )
        return cls.BASE_CLASS

    @classmethod
    def has_base_class(cls) -> bool:
        """Returns True if this registrable class has a base class."""
        return cls.BASE_CLASS is not None

    @classmethod
    def register(
        cls: Type[T],
        name: Optional[str] = None,
    ) -> Callable[[Type[T]], Type[T]]: ...

    @classmethod
    def by_name(cls: Type[T], name: str) -> Type[T]: ...

    @classmethod
    def registered_name_for_class(cls: Type[T], clazz: Type[T]) -> Optional[str]: ...

    @classmethod
    def name_for_object_class(cls: Type[T], obj: T) -> str:
        obj_class = obj.__class__
        registered_name = cls.registered_name_for_class(obj_class)
        return registered_name if registered_name is not None else obj_class.__name__


T3 = TypeVar("T3", bound="RegistrableProtocol")


class Registrable(RegistrableProtocol[T3]):
    """A class that can be registered and instantiated by name.

    This class is used to create a registry of classes that can be instantiated
    by name. The registry is stored in the class variable `_registry`, which is
    a dictionary mapping class names to classes.
    """

    _registry: Dict[Type, Dict[str, Type]] = defaultdict(dict)

    @classmethod
    def register(
        cls: Type[T],
        name: Optional[str] = None,
    ) -> Callable[[Type[T]], Type[T]]:
        if cls.BASE_CLASS is not None and cls is not cls.BASE_CLASS:
            raise RegistrationError(
                f"Cannot register {cls.__name__}; it is already registered as a subclass of {cls.BASE_CLASS.__name__}"
            )
        cls.BASE_CLASS = cls

        registry = Registrable._registry[cls]

        def add_subclass_to_registry(subclass: Type[T]) -> Type[T]:
            register_name = subclass.__name__ if name is None else name

            if not inspect.isclass(subclass) or not issubclass(subclass, cls):
                raise RegistrationError(
                    f"Cannot register {subclass.__name__} as {register_name}; "
                    f"{subclass.__name__} must be a subclass of {cls.__name__}"
                )

            if register_name in registry:
                raise RegistrationError(
                    f"Cannot register {subclass.__name__} as {register_name}; "
                    f"name already in use for {registry[register_name].__name__}"
                )

            registry[register_name] = subclass
            return subclass

        return add_subclass_to_registry

    @classmethod
    def by_name(cls: Type[T], name: str) -> Type[T]:
        if name in Registrable._registry[cls]:
            return Registrable._registry[cls][name]

        raise RegistrationError(f"{name} is not a registered name for {cls.__name__}.")

    @classmethod
    def registered_name_for_class(cls: Type[T], clazz: Type[T]) -> Optional[str]:
        inverse_lookup = {v: k for k, v in Registrable._registry[cls].items()}
        return inverse_lookup.get(clazz)
