import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
    Any,
    Dict,
    Generic,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
    overload,
)

from pie_core.auto import Auto
from pie_core.document import Document
from pie_core.hf_hub_mixin import HFHubMixin
from pie_core.hparams_mixin import PieHyperparametersMixin
from pie_core.model import AutoModel, Model
from pie_core.registrable import Registrable
from pie_core.taskmodule import AutoTaskModule, TaskModule

logger = logging.getLogger(__name__)


TAnnotationPipelineHFHubMixin = TypeVar(
    "TAnnotationPipelineHFHubMixin", bound="AnnotationPipelineHFHubMixin"
)


class AnnotationPipelineHFHubMixin(HFHubMixin):
    config_name = "pipeline_config.json"
    config_type_key = "pipeline_type"
    auto_model_class = AutoModel
    auto_taskmodule_class = AutoTaskModule

    def _save_pretrained(self, save_directory) -> None:
        return None

    @classmethod
    def _from_pretrained(
        cls: Type[TAnnotationPipelineHFHubMixin],
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
    ) -> TAnnotationPipelineHFHubMixin:

        taskmodule_or_taskmodule_kwargs = kwargs.pop("taskmodule", None)
        if "taskmodule_kwargs" in kwargs:
            logger.warning("taskmodule_kwargs is deprecated. Use taskmodule instead.")
            taskmodule_or_taskmodule_kwargs = kwargs.pop("taskmodule_kwargs")
        model_or_model_kwargs = kwargs.pop("model", None)
        if "model_kwargs" in kwargs:
            logger.warning("model_kwargs is deprecated. Use model instead.")
            model_or_model_kwargs = kwargs.pop("model_kwargs")

        if isinstance(model_or_model_kwargs, Model):
            # if model is already a Model instance, use it directly
            model = model_or_model_kwargs
        else:
            # otherwise, create a new Model instance via AutoModel
            model = cls.auto_model_class.from_pretrained(
                pretrained_model_name_or_path=model_id,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                **(model_or_model_kwargs or {}),
            )

        if isinstance(taskmodule_or_taskmodule_kwargs, TaskModule):
            # if taskmodule is already a TaskModule instance, use it directly
            taskmodule = taskmodule_or_taskmodule_kwargs
        else:
            # otherwise:
            # 1. try to retrieve the taskmodule config file
            taskmodule_config_file = cls.auto_taskmodule_class.retrieve_config_file(
                model_id=model_id,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
            # 2. If the taskmodule config file is found, load the taskmodule via from_pretrained()
            if taskmodule_config_file is not None:
                taskmodule = cls.auto_taskmodule_class.from_pretrained(
                    pretrained_model_name_or_path=model_id,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    cache_dir=cache_dir,
                    local_files_only=local_files_only,
                    **(taskmodule_or_taskmodule_kwargs or {}),
                )
            # 3. Otherwise, do not load a taskmodule.
            #    It is assumed that the model contains the taskmodule.
            else:
                taskmodule = None

        kwargs["model"] = model
        if taskmodule is not None:
            kwargs["taskmodule"] = taskmodule

        pipeline = cls.from_config(config=config or {}, **kwargs)

        return pipeline


TModel = TypeVar("TModel", bound="Model")
TTaskModule = TypeVar("TTaskModule", bound="TaskModule")


class AnnotationPipeline(
    AnnotationPipelineHFHubMixin,
    PieHyperparametersMixin,
    Registrable["AnnotationPipeline"],
    Generic[TModel, TTaskModule],
    ABC,
):

    def __init__(self, model: TModel, taskmodule: Optional[TTaskModule] = None, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=["model", "taskmodule"])
        self._model = model
        self._taskmodule = taskmodule

    @property
    def model(self) -> TModel:
        """The model used in the pipeline."""
        return self._model

    @model.setter
    def model(self, model: TModel) -> None:
        """Set the model used in the pipeline."""
        self._model = model

    @property
    def taskmodule(self) -> TTaskModule:
        """The taskmodule used in the pipeline."""
        if self._taskmodule is not None:
            return self._taskmodule
        # if the taskmodule is None, try to retrieve it from the model
        if hasattr(self.model, "taskmodule"):
            return self.model.taskmodule
        raise ValueError("The taskmodule is None and the model does not contain a taskmodule.")

    @taskmodule.setter
    def taskmodule(self, taskmodule: Optional[TTaskModule]) -> None:
        """Set the taskmodule used in the pipeline."""
        self._taskmodule = taskmodule

    def save_pretrained(
        self,
        save_directory: Union[str, Path],
        *,
        repo_id: Optional[str] = None,
        push_to_hub: bool = False,
        **kwargs,
    ) -> Optional[str]:
        """Save the model, taskmodule and pipeline config to a local directory or the Huggingface
        Hub."""

        all_kwargs = kwargs.copy()
        all_kwargs["repo_id"] = repo_id
        all_kwargs["push_to_hub"] = push_to_hub
        all_kwargs["save_directory"] = save_directory

        # save model
        self._model.save_pretrained(**all_kwargs)

        # save taskmodule, if it exists
        if self._taskmodule is not None:
            self._taskmodule.save_pretrained(**all_kwargs)

        # save pipeline config and maybe upload to hub
        super().save_pretrained(**all_kwargs)

        return None

    def _config(self) -> Dict[str, Any]:
        config = super()._config() or {}
        if (
            self.has_base_class()
            and self.base_class().registered_name_for_class(self.__class__) is not None
        ):
            config[self.config_type_key] = self.base_class().name_for_object_class(self)
        else:
            logger.warning(
                f"{self.__class__.__name__} is not registered. It will not work"
                " with AutoAnnotationPipeline.from_pretrained() or"
                " AutoAnnotationPipeline.from_config(). Consider to annotate the class with"
                " @AnnotationPipeline.register() or @AnnotationPipeline.register(name='...')"
                " to register it as an AnnotationPipeline which will allow to load it via"
                " AutoAnnotationPipeline."
            )
        # add all hparams
        config.update(self.hparams)
        return config

    @overload
    def __call__(
        self,
        documents: Document,
        inplace: bool = True,
        *args,
        **kwargs,
    ) -> Document: ...

    @overload
    def __call__(
        self,
        documents: Sequence[Document],
        inplace: bool = True,
        *args,
        **kwargs,
    ) -> Sequence[Document]: ...

    @abstractmethod
    def __call__(
        self,
        documents: Union[Document, Sequence[Document]],
        inplace: bool = True,
        *args,
        **kwargs,
    ) -> Union[Document, Sequence[Document]]: ...


class AutoAnnotationPipeline(AnnotationPipelineHFHubMixin, Auto[AnnotationPipeline]):

    BASE_CLASS = AnnotationPipeline
