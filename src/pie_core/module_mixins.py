import logging
from abc import ABC, abstractmethod
from typing import Optional, Type

from pie_core.document import Document

logger = logging.getLogger(__name__)


class WithDocumentTypeMixin:

    DOCUMENT_TYPE: Optional[Type[Document]] = None

    @property
    def document_type(self) -> Optional[Type[Document]]:
        return self.DOCUMENT_TYPE

    # TODO: remove, functionality can be replaced downstream with https://github.com/ArneBinder/pie-datasets/pull/175
    def convert_dataset(self, dataset: "pie_datasets.DatasetDict") -> "pie_datasets.DatasetDict":  # type: ignore # noqa
        logger.warning(
            "The WithDocumentTypeMixin.convert_dataset(DatasetDict) method is deprecated "
            "and will be removed in the future. Use DatasetDict.to_document_type(WithDocumentTypeMixin) "
            "instead (this requires at least pie-datasets 0.10.8)."
        )

        name = type(self).__name__
        # auto-convert the dataset if a document type is specified
        if self.document_type is not None:
            if issubclass(dataset.document_type, self.document_type):
                logger.info(
                    f"the dataset is already of the document type that is specified by {name}: "
                    f"{self.document_type}"
                )
            else:
                logger.info(
                    f"convert the dataset to the document type that is specified by {name}: "
                    f"{self.document_type}"
                )
                dataset = dataset.to_document_type(self.document_type)
        else:
            logger.warning(
                f"{name} does not specify a document type. The dataset can not be automatically converted "
                f"to a document type."
            )

        return dataset


class EnterDatasetMixin(ABC):
    """Mixin for processors that enter a dataset context."""

    @abstractmethod
    def enter_dataset(self, dataset, name: Optional[str] = None) -> None:
        """Enter dataset context."""


class ExitDatasetMixin(ABC):
    """Mixin for processors that exit a dataset context."""

    @abstractmethod
    def exit_dataset(self, dataset, name: Optional[str] = None) -> None:
        """Exit dataset context."""


class EnterDatasetDictMixin(ABC):
    """Mixin for processors that enter a dataset dict context."""

    @abstractmethod
    def enter_dataset_dict(self, dataset_dict) -> None:
        """Enter dataset dict context."""


class ExitDatasetDictMixin(ABC):
    """Mixin for processors that exit a dataset dict context."""

    @abstractmethod
    def exit_dataset_dict(self, dataset_dict) -> None:
        """Exit dataset dict context."""
