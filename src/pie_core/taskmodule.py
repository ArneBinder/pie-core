import copy
import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from tqdm import tqdm

from pie_core.auto import Auto
from pie_core.document import Annotation, Document
from pie_core.hf_hub_mixin import HFHubMixin, TNestedBoolDict
from pie_core.hparams_mixin import PieHyperparametersMixin
from pie_core.metric import EncodingMetric
from pie_core.module_mixins import WithDocumentTypeMixin
from pie_core.preparable import PreparableMixin
from pie_core.registrable import Registrable
from pie_core.taskencoding import (
    IterableTaskEncodingDataset,
    TaskEncoding,
    TaskEncodingDataset,
    TaskEncodingSequence,
)

DocumentType = TypeVar("DocumentType", bound=Document)
InputEncoding = TypeVar("InputEncoding")
TargetEncoding = TypeVar("TargetEncoding")
# TaskEncoding: defined below
InputBatchEncoding = TypeVar("InputBatchEncoding")
TargetBatchEncoding = TypeVar("TargetBatchEncoding")
# TaskBatchEncoding: TypeAlias = Tuple[InputBatchEncoding, Optional[TargetBatchEncoding]]
# TODO: remove in favor of InputBatchEncoding and TargetBatchEncoding
TaskBatchEncoding = TypeVar("TaskBatchEncoding")
# ModelBatchEncoding: defined in models
ModelBatchOutput = TypeVar("ModelBatchOutput")
TaskOutput = TypeVar("TaskOutput")


logger = logging.getLogger(__name__)

TTaskModuleHFHubMixin = TypeVar("TTaskModuleHFHubMixin", bound="TaskModuleHFHubMixin")


class TaskModuleHFHubMixin(HFHubMixin):
    config_name = "taskmodule_config.json"
    config_type_key = "taskmodule_type"

    def _save_pretrained(self, save_directory) -> None:
        return None

    @classmethod
    def _from_pretrained(
        cls: Type[TTaskModuleHFHubMixin],
        *,
        model_id: str,
        revision: Optional[str],
        cache_dir: Optional[Union[str, Path]],
        force_download: bool,
        proxies: Optional[Dict],
        resume_download: bool,
        local_files_only: bool,
        token: Union[str, bool, None],
        config: Optional[dict] = None,
        **kwargs,
    ) -> TTaskModuleHFHubMixin:

        taskmodule = cls.from_config(config=config or {}, **kwargs)

        return taskmodule


class TaskModule(
    ABC,
    TaskModuleHFHubMixin,
    PieHyperparametersMixin,
    Registrable["TaskModule"],
    WithDocumentTypeMixin,
    PreparableMixin,
    Generic[
        DocumentType,
        InputEncoding,
        TargetEncoding,
        # TODO: replace with InputBatchEncoding and TargetBatchEncoding
        TaskBatchEncoding,
        ModelBatchOutput,
        TaskOutput,
    ],
):
    def __init__(self, encode_document_batch_size: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.encode_document_batch_size = encode_document_batch_size

    def _config(self) -> Dict[str, Any]:
        config = super()._config() or {}
        if (
            self.has_base_class()
            and self.base_class().registered_name_for_class(self.__class__) is not None
        ):
            config[self.config_type_key] = self.base_class().name_for_object_class(self)
        else:
            logger.warning(
                f"{self.__class__.__name__} is not registered. It will not work "
                "with AutoTaskModule.from_pretrained() or "
                "AutoTaskModule.from_config(). Consider to annotate the class with "
                "@TaskModule.register() or @TaskModule.register(name='...') "
                "to register it as a TaskModule which will allow to load it via AutoTaskModule."
            )
        # add all hparams
        config.update(self.hparams)
        # add all prepared attributes
        config.update(self.prepared_attributes)
        return config

    @classmethod
    def _from_config(
        cls: Type["TaskModule"],
        config: Dict[str, Any],
        config_override: Optional[TNestedBoolDict] = None,
        **kwargs,
    ) -> "TaskModule":
        taskmodule: TaskModule = super()._from_config(
            config, config_override=config_override, **kwargs
        )
        taskmodule.post_prepare()
        return taskmodule

    def on_encode_start(self) -> None:
        """This method is called when the encoding starts.

        It can be used to reset the task module state before encoding, e.g., resetting the
        statistics, etc.
        """
        pass

    def on_encode_end(self) -> None:
        """This method is called when the encoding ends.

        Important: When using lazy encoding (i.e. as_iterator=True), this will not be called!

        It can be used to reset the task module state after encoding, show collected
        statistics, etc.
        """
        pass

    def batch_encode(
        self,
        documents: Sequence[DocumentType],
        encode_target: bool,
        show_progress: bool = False,
    ) -> Tuple[
        Sequence[TaskEncoding[DocumentType, InputEncoding, TargetEncoding]], Sequence[DocumentType]
    ]:
        """Encode a batch of documents and return task encodings and documents in corresponding
        order.

        If 'encode_target = True' is passed, target encodings will be assigned to task encodings.
        Only encodings that got targets will be returned.
        """
        task_encodings, documents_in_order = self.encode_inputs(
            documents, show_progress=show_progress
        )

        if encode_target:
            task_encodings = self.encode_targets(task_encodings, show_progress=show_progress)
        return task_encodings, documents_in_order

    def _encoding_iterator(
        self,
        documents: Iterable[DocumentType],
        encode_target: bool,
        batch_size: Optional[int] = None,
        show_progress: bool = False,
    ) -> Iterator[TaskEncoding[DocumentType, InputEncoding, TargetEncoding]]:
        document_batch = []
        if show_progress and batch_size is not None:
            logger.warning(
                "do not show document encoding progress because we encode lazily with an iterator"
            )
        for i, doc in enumerate(documents):
            document_batch.append(doc)

            if batch_size is not None and len(document_batch) >= batch_size:
                yield from self.batch_encode(
                    documents=document_batch[:batch_size],
                    encode_target=encode_target,
                    show_progress=False,
                )[0]
                document_batch = document_batch[batch_size:]

        if len(document_batch) > 0:
            yield from self.batch_encode(
                documents=document_batch,
                encode_target=encode_target,
                show_progress=show_progress and batch_size is None,
            )[0]

    def encode(
        self,
        documents: Union[DocumentType, Iterable[DocumentType]],
        encode_target: bool = False,
        document_batch_size: Optional[int] = None,
        as_task_encoding_sequence: Optional[bool] = None,
        as_iterator: Optional[bool] = None,
        as_dataset: bool = False,
        show_progress: bool = False,
    ) -> Union[
        Sequence[TaskEncoding[DocumentType, InputEncoding, TargetEncoding]],
        TaskEncodingSequence[
            TaskEncoding[DocumentType, InputEncoding, TargetEncoding], DocumentType
        ],
        Iterator[TaskEncoding[DocumentType, InputEncoding, TargetEncoding]],
        TaskEncodingDataset[TaskEncoding[DocumentType, InputEncoding, TargetEncoding]],
        IterableTaskEncodingDataset[TaskEncoding[DocumentType, InputEncoding, TargetEncoding]],
    ]:
        """Encode a single or multiple documents and return a sequence of TaskEncodings:
        objects that hold the model inputs, optionally training targets, and the source document.

        Parameters:
            documents (Iterable[DocumentType]): Document or documents to encode.
            encode_target (bool, optional): Whether to create target encodings. Defaults to False.
            document_batch_size (Optional[int], optional): If provided, encode documents in batches of
                document_batch_size, otherwise use self.encode_document_batch_size. Defaults to None.
            as_task_encoding_sequence (Optional[bool], optional): Whether to return a TaskEncodingSequence,
                a wrapper around a sequence of TaskEncodings that also holds the documents in the order
                they were encoded.
            Return type should be a Sequence
                of TaskEncodings. Defaults to None. If not set - this will be set to True if NOT
                encoding targets ('encode_target = False').
            as_iterator (Optional[bool], optional): Whether to return an iterator over the
                TaskEncodings instead of a sequence. If not set, this will be set to True if
                documents is an iterable (i.e. not a Sequence). Defaults to None.
            as_dataset (bool): Return type should be a Dataset. Cannot be used with
                'as_task_encoding_sequence'. Defaults to False.
            show_progress (bool, optional): Show progress bar. Defaults to False.
        """
        self.assert_is_prepared()

        self.on_encode_start()

        # backwards compatibility
        if as_task_encoding_sequence is None:
            as_task_encoding_sequence = not encode_target

        if isinstance(documents, Document):
            documents = [documents]  # type: ignore

        if as_iterator is None:
            as_iterator = not isinstance(documents, Sequence)

        if document_batch_size is None:
            document_batch_size = self.encode_document_batch_size

        result: Union[
            Sequence[TaskEncoding[DocumentType, InputEncoding, TargetEncoding]],
            TaskEncodingSequence[
                TaskEncoding[DocumentType, InputEncoding, TargetEncoding], DocumentType
            ],
            Iterator[TaskEncoding[DocumentType, InputEncoding, TargetEncoding]],
            TaskEncodingDataset[TaskEncoding[DocumentType, InputEncoding, TargetEncoding]],
            IterableTaskEncodingDataset[TaskEncoding[DocumentType, InputEncoding, TargetEncoding]],
        ]

        if as_iterator:
            if as_task_encoding_sequence:
                raise ValueError("can not return a TaskEncodingSequence as Iterator")
            encodings_iterator = self._encoding_iterator(
                documents=documents,
                encode_target=encode_target,
                batch_size=document_batch_size,
                show_progress=show_progress,
            )
            if as_dataset:
                result = IterableTaskEncodingDataset(encodings=encodings_iterator)
            else:
                result = encodings_iterator
        else:
            encodings: List[TaskEncoding[DocumentType, InputEncoding, TargetEncoding]] = []
            documents_in_order: List[DocumentType] = []
            docs_as_list = list(documents)
            bs = document_batch_size or len(docs_as_list)
            for i in tqdm(
                range(0, len(docs_as_list), bs),
                disable=not (show_progress and document_batch_size is not None),
                desc="encode documents",
            ):
                cur_task_encodings, cur_documents_in_order = self.batch_encode(
                    documents=docs_as_list[i : i + bs],
                    encode_target=encode_target,
                    show_progress=show_progress and document_batch_size is None,
                )
                encodings.extend(cur_task_encodings)
                documents_in_order.extend(cur_documents_in_order)

            if as_task_encoding_sequence:
                if as_dataset:
                    raise ValueError("can not return a TaskEncodingSequence as a dataset")
                result = TaskEncodingSequence(
                    task_encodings=encodings,
                    documents_in_order=documents_in_order,
                )
            else:
                # during training, we return only the sequence of task_encodings, because
                # we don't need the ordering of input documents and also don't re-assign
                # task encodings to input documents
                if as_dataset:
                    result = TaskEncodingDataset(encodings=encodings)
                else:
                    result = encodings

        if not as_iterator:
            self.on_encode_end()

        return result

    def encode_inputs(
        self,
        documents: Sequence[DocumentType],
        show_progress: bool = False,
    ) -> Tuple[
        Sequence[TaskEncoding[DocumentType, InputEncoding, TargetEncoding]],
        Sequence[DocumentType],
    ]:
        """Encode a batch of documents and return task encodings and documents in corresponding
        order."""
        documents_in_order: List[DocumentType] = []
        task_encodings: List[TaskEncoding[DocumentType, InputEncoding, TargetEncoding]] = []
        for document in tqdm(documents, disable=not show_progress, desc="encode inputs"):
            # a document might be generated on the fly (e.g. with a Dataset), so we add it here
            documents_in_order.append(document)

            possible_task_encodings = self.encode_input(document)

            # encode_input returns None or an empty list
            if possible_task_encodings is None or not possible_task_encodings:
                continue

            elif isinstance(possible_task_encodings, TaskEncoding):
                task_encodings.append(possible_task_encodings)

            else:
                task_encodings.extend(possible_task_encodings)

        return task_encodings, documents_in_order

    @abstractmethod
    def encode_input(
        self,
        document: DocumentType,
    ) -> Optional[
        Union[
            TaskEncoding[DocumentType, InputEncoding, TargetEncoding],
            Sequence[TaskEncoding[DocumentType, InputEncoding, TargetEncoding]],
        ]
    ]:
        """Create one or multiple task encodings including the model inputs from a given
        document."""

    def encode_targets(
        self,
        task_encodings: Sequence[TaskEncoding[DocumentType, InputEncoding, TargetEncoding]],
        show_progress: bool = False,
    ) -> List[TaskEncoding[DocumentType, InputEncoding, TargetEncoding]]:
        """Given a list of task encodings, get and assign the respective target encodings and
        return all task encodings that got a target.

        In that means, this will filter out all encodings without a target. This can be useful when
        different sets of encodings are required for training and inference. It mitigates the need
        to implement special logic that depends on target information in encode_input(). Encodings
        that are not suitable for training, i.e. where no target information is available, can be
        filtered out easily by letting encode_target() return None.
        """
        res = []
        for task_encoding in tqdm(
            task_encodings, disable=not show_progress, desc="encode targets"
        ):
            target_encoding = self.encode_target(task_encoding)
            if target_encoding is not None:
                task_encoding.targets = target_encoding
                res.append(task_encoding)
        return res

    @abstractmethod
    def encode_target(
        self,
        task_encoding: TaskEncoding[DocumentType, InputEncoding, TargetEncoding],
    ) -> Optional[TargetEncoding]:
        """Create a training target for a model input (which is wrapped in a task encoding). May
        return None, in which case the task encoding will not be included in a training batch
        (i.e., it will be excluded from training).

        This may use the model inputs, data (text, annotations, etc.) of the underlying document,
        or any other metadata attached to the task encoding in encode input.
        """

    @abstractmethod
    def unbatch_output(self, model_output: ModelBatchOutput) -> Sequence[TaskOutput]:
        """This method has to convert the batch output of the model (i.e. a dict of lists) to the
        list of individual outputs (i.e. a list of dicts).

        This is in preparation to generate a list of all model outputs that has the same length as
        all model inputs.
        """

    def decode(
        self,
        task_encodings: Sequence[TaskEncoding[DocumentType, InputEncoding, TargetEncoding]],
        task_outputs: Sequence[TaskOutput],
        inplace: bool = True,
    ) -> Sequence[DocumentType]:
        """This method takes the model inputs and (unbatched) model outputs and creates a list of
        documents that hold the new annotations created from model predictions."""
        documents: Dict[int, DocumentType] = {}

        # TaskEncodingSequence provides us with the correct ordering
        if isinstance(task_encodings, TaskEncodingSequence):
            for document in task_encodings.documents_in_order:
                document_id = id(document)
                documents[document_id] = document if inplace else copy.deepcopy(document)
        # Otherwise we assume that documents are ordered according to the sequence of
        # unique documents defined by the sequence of task encodings
        else:
            for task_encoding in task_encodings:
                document = task_encoding.document
                document_id = id(document)
                if document_id not in documents:
                    documents[document_id] = document if inplace else copy.deepcopy(document)

        if not inplace:
            task_encodings = [
                TaskEncoding[DocumentType, InputEncoding, TargetEncoding](
                    document=documents[id(task_encoding.document)],
                    inputs=task_encoding.inputs,
                    targets=task_encoding.targets if task_encoding.has_targets else None,
                    metadata=task_encoding.metadata,
                )
                for task_encoding in task_encodings
            ]

        self.combine_outputs(task_encodings, task_outputs)

        unique_documents = list(documents.values())
        return unique_documents

    def combine_outputs(
        self,
        task_encodings: Sequence[TaskEncoding[DocumentType, InputEncoding, TargetEncoding]],
        task_outputs: Sequence[TaskOutput],
    ) -> None:
        """Create annotations from task encodings and respective task outputs.

        The annotations will be attached as predictions to annotation layer(s) of the source
        document.
        """
        for task_encoding, task_output in zip(task_encodings, task_outputs):
            self.combine_output(task_encoding, task_output)

    def combine_output(
        self,
        task_encoding: TaskEncoding[DocumentType, InputEncoding, TargetEncoding],
        task_output: TaskOutput,
    ) -> None:
        """Create an annotation from task encoding and a task output.

        The annotation will be attached as prediction to respective annotation layer of the source
        document.
        """
        for annotation_name, annotation in self.create_annotations_from_output(
            task_encoding, task_output
        ):
            task_encoding.document[annotation_name].predictions.append(annotation)

    @abstractmethod
    def create_annotations_from_output(
        self,
        task_encoding: TaskEncoding[DocumentType, InputEncoding, TargetEncoding],
        task_output: TaskOutput,
    ) -> Iterator[Tuple[str, Annotation]]:
        """Create annotations from a task output (a single model prediction) and the respective
        task encoding (including model inputs and the source document). The annotations will be
        attached as predictions to annotation layer(s) of the source document.

        The method has to yield tuples (annotation_layer_name, annotation).
        """

    @abstractmethod
    def collate(
        self, task_encodings: Sequence[TaskEncoding[DocumentType, InputEncoding, TargetEncoding]]
    ) -> TaskBatchEncoding:
        """Convert a list of task encodings to a batch that will be passed to the model."""

    def configure_model_metric(
        self, stage: str
    ) -> Optional[EncodingMetric[ModelBatchOutput, TargetBatchEncoding]]:
        """Configure the model metric. This method is called by the model to configure the metric
        for the given stage. It will be used to compute the metric score(s) based on the model
        predictions and targets.

        Args:
            stage: The stage for which the metric is configured, e.g., "train", "val", "test".

        Returns:
            The metric for the given stage.
        """
        logger.warning(
            f"TaskModule {self.__class__.__name__} does not implement a model metric. "
            f"Override configure_model_metric(stage: str) to configure a metric for stage '{stage}'."
        )
        return None


class AutoTaskModule(TaskModuleHFHubMixin, Auto[TaskModule]):

    BASE_CLASS = TaskModule
