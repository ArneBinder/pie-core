from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Iterable, Optional, TypeVar, Union

from pie_core.document import Document
from pie_core.module_mixins import WithDocumentTypeMixin

T = TypeVar("T")


class DocumentMetric(ABC, WithDocumentTypeMixin, Generic[T]):
    """This defines the interface for a document metric."""

    def __init__(self) -> None:
        self.reset()
        self._current_split: Optional[str] = None

    @abstractmethod
    def reset(self) -> None:
        """Any reset logic that needs to be performed before the metric is called again."""

    def __call__(
        self,
        document_or_collection: Union[Iterable[Document], Document, Dict[str, Iterable[Document]]],
    ) -> Union[Dict[str, T], T]:
        """This method is called to update the metric with a document or collection of documents.

        If a collection is passed, the metric is also computed and the result is returned. If the
        collection is a dictionary, the metric is computed for each split and the result is
        returned as a dictionary.
        """
        if isinstance(document_or_collection, Document):
            # do not reset here to allow for multiple calls
            self._update(document_or_collection)
            return self.compute(reset=False)
        elif isinstance(document_or_collection, dict):
            result: Dict[str, T] = {}
            for split_name, split in document_or_collection.items():
                self._current_split = split_name
                self.reset()
                split_values: T = self(split)  # type: ignore
                result[split_name] = split_values
                self._current_split = None
            return result
        elif isinstance(document_or_collection, Iterable):
            for doc in document_or_collection:
                if not isinstance(doc, Document):
                    raise TypeError(
                        f"document_or_collection contains an object that is not a document: {type(doc)}"
                    )
                self._update(doc)
            # do not reset here to allow for multiple calls
            return self.compute(reset=False)
        else:
            raise TypeError(
                f"document_or_collection has unknown type: {type(document_or_collection)}"
            )

    def compute(self, reset: bool = True) -> T:
        metric_values = self._compute()
        if reset:
            self.reset()
        return metric_values

    @abstractmethod
    def _update(self, document: Document) -> None:
        """This method is called to update the metric with the new document."""

    @abstractmethod
    def _compute(self) -> T:
        """This method is called to get the metric values."""

    @property
    def current_split(self) -> Optional[str]:
        """The current split that is being processed."""
        return self._current_split


TPredictions = TypeVar("TPredictions")
TTargets = TypeVar("TTargets")


class EncodingMetric(Generic[TPredictions, TTargets], ABC):
    """This defines the interface for a metric that is used to compute score(s) based on (model)
    predictions and targets (probably in batch form).

    It loosely follows the Metric API of torchmetrics.
    """

    def reset(self) -> None:
        """Any (state) reset logic that needs to be performed before the metric is called again."""
        pass

    def update(self, predictions: TPredictions, targets: TTargets) -> None:
        """This method is called to update the metric state with the predictions and targets."""
        pass

    def compute(self) -> Any:
        """This method is called to compute the metric value(s) from the metric state."""
        pass
