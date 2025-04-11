import logging
from abc import abstractmethod
from typing import Any, Dict, Optional, Sequence, Union, overload

from pytorch_lightning.core.mixins import HyperparametersMixin

from pie_core import AutoModel, AutoTaskModule, Model, TaskModule
from pie_core.auto import Auto
from pie_core.document import Document
from pie_core.hf_hub_mixin import AnnotationPipelineHFHubMixin
from pie_core.registrable import Registrable

logger = logging.getLogger(__name__)


class AnnotationPipeline(
    AnnotationPipelineHFHubMixin, HyperparametersMixin, Registrable["AnnotationPipeline"]
):
    def __init__(self, model: Model, taskmodule: Optional[TaskModule] = None, **kwargs):
        """Initialize the AnnotationPipeline.

        The taskmodule is optional, it may be embedded in the model.
        Args:
            model: The model to use for the pipeline.
            taskmodule: The task module to use for the pipeline.
            kwargs: Additional keyword arguments to pass to the parent class.
        """
        super().__init__(**kwargs)
        self._model = model
        self._taskmodule = taskmodule

    @property
    def model(self) -> Model:
        """Get the model used by the pipeline."""
        return self._model

    @model.setter
    def model(self, model: Model) -> None:
        self._model = model

    @property
    def taskmodule(self) -> TaskModule:
        """Get the task module used by the pipeline."""
        if self._taskmodule is not None:
            return self._taskmodule
        elif hasattr(self.model, "taskmodule"):
            return self.model.taskmodule
        else:
            raise ValueError("No taskmodule found in the model. Please provide a taskmodule.")

    @taskmodule.setter
    def taskmodule(self, taskmodule: Optional[TaskModule]) -> None:
        self._taskmodule = taskmodule

    def _config(self) -> Dict[str, Any]:
        config = super()._config() or {}
        config[self.config_type_key] = self.base_class().name_for_object_class(self)
        # add all hparams
        config.update(self.hparams)
        return config

    def _save_pretrained(self, save_directory) -> None:
        raise NotImplementedError(
            "AnnotationPipeline does not yet support saving to file or the Huggingface Hub."
        )

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

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        force_download: bool = False,
        resume_download: bool = False,
        proxies: Optional[Dict] = None,
        use_auth_token: Optional[str] = None,
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
        **kwargs,
    ) -> "AnnotationPipeline":
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
            model = AutoModel.from_pretrained(
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                use_auth_token=use_auth_token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                **(model_or_model_kwargs or {}),
            )

        if isinstance(taskmodule_or_taskmodule_kwargs, TaskModule):
            # if taskmodule is already a TaskModule instance, use it directly
            taskmodule = taskmodule_or_taskmodule_kwargs
        else:
            # otherwise:
            # 1. try to retrieve the taskmodule config
            taskmodule_config = AutoTaskModule.retrieve_config(
                model_id=pretrained_model_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                **(taskmodule_or_taskmodule_kwargs or {}),
            )
            # 2. If the taskmodule config is not None, create a taskmodule from it
            if taskmodule_config is not None:
                # the taskmodule_kwargs are already consumed, so we do not pass them again
                taskmodule = AutoTaskModule.from_config(config=taskmodule_config)
            # 3. If the taskmodule config is None, create a taskmodule from the kwargs
            elif taskmodule_or_taskmodule_kwargs is not None:
                taskmodule = AutoTaskModule.from_config(
                    config={}, **taskmodule_or_taskmodule_kwargs
                )
            # 4. If the taskmodule is still None, do not create a taskmodule.
            #    It is assumed that the model contains the taskmodule.
            else:
                taskmodule = None

        kwargs["taskmodule"] = taskmodule
        kwargs["model"] = model

        pipeline = super().from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            use_auth_token=use_auth_token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            **kwargs,
        )

        return pipeline
