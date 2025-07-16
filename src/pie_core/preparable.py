import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PreparableMixin:
    """Mixin for preparable classes.

    Provides common function to prepare class attributes and make sure they are set.

    Usage:

    - List all attributes that must exist after `prepare()` call in [`PREPARED_ATTRIBUTES`].
    - Override `_prepare()` method for custom preparation.
    - Override `_post_prepare()` method for further preparation steps that need prepared attributes.
    - `post_prepare()` is called within `prepare()`, but you may want to call it manually if object is created
    without prepare() (e.g. loaded from config file with all needed attributes).
    """

    # list of attribute names that need to be set by _prepare()
    PREPARED_ATTRIBUTES: List[str] = []

    @property
    def is_prepared(self) -> bool:
        """Returns True, iff all attributes listed in PREPARED_ATTRIBUTES are set.

        Note: Attributes set to None are not considered to be prepared!
        """
        return all(
            getattr(self, attribute, None) is not None for attribute in self.PREPARED_ATTRIBUTES
        )

    @property
    def prepared_attributes(self) -> Dict[str, Any]:
        if not self.is_prepared:
            raise Exception(f"The {self.__class__.__name__} is not prepared.")
        return {param: getattr(self, param) for param in self.PREPARED_ATTRIBUTES}

    def _prepare(self, *args, **kwargs) -> None:
        """This method needs to set all attributes listed in PREPARED_ATTRIBUTES."""

    def _post_prepare(self) -> None:
        """Any code to do further one-time setup, but that requires the prepared attributes."""

    def assert_is_prepared(self, msg: Optional[str] = None) -> None:
        if not self.is_prepared:
            attributes_not_prepared = [
                param for param in self.PREPARED_ATTRIBUTES if getattr(self, param, None) is None
            ]
            raise Exception(
                f"{'' if not msg else msg + ' '}Required attributes that are not set: {str(attributes_not_prepared)}"
            )

    def post_prepare(self) -> None:
        self.assert_is_prepared()
        self._post_prepare()

    def prepare(self, *args, **kwargs) -> None:
        if self.is_prepared:
            if len(self.PREPARED_ATTRIBUTES) > 0:
                msg = f"The {self.__class__.__name__} is already prepared, do not prepare again."
                for k, v in self.prepared_attributes.items():
                    msg += f"\n{k} = {str(v)}"
                logger.warning(msg)
        else:
            self._prepare(*args, **kwargs)
            self.assert_is_prepared(
                msg=f"_prepare() was called, but the {self.__class__.__name__} is not prepared."
            )
        self._post_prepare()
        return None
