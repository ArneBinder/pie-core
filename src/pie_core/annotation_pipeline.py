from abc import abstractmethod
from typing import Any, Dict, Optional, Sequence, Type, Union, overload

from pytorch_lightning.core.mixins import HyperparametersMixin

from pie_core import AutoModel, AutoTaskModule, Model, TaskModule
from pie_core.auto import Auto
from pie_core.document import Document
from pie_core.hf_hub_mixin import AnnotationPipelineHFHubMixin
from pie_core.registrable import Registrable


class AnnotationPipeline(
    AnnotationPipelineHFHubMixin, HyperparametersMixin, Registrable["AnnotationPipeline"]
):
    def __init__(self, model: Model, taskmodule: Optional[TaskModule] = None, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.taskmodule = taskmodule

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
        device: int = -1,
        binary_output: bool = False,
        **kwargs,
    ) -> "AnnotationPipeline":
        taskmodule_kwargs = kwargs.pop("taskmodule_kwargs")
        model_kwargs = kwargs.pop("model_kwargs")

        model = AutoModel.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            use_auth_token=use_auth_token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            **(model_kwargs or {}),
        )

        # TODO: Use AutoTaskModule.retrieve_config to check if a taskmodule config
        # is available, and then:
        #   - use AutoTaskModule.from_config to create the taskmodule (pass taskmodule_kwargs)
        # If no config is available:
        #   - raise an error if taskmodule_kwargs is not empty,
        #   - raise an error if model.taskmodule is not available (after the model is created).
        # Requires https://github.com/ArneBinder/pie-core/pull/28.
        taskmodule = AutoTaskModule.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            use_auth_token=use_auth_token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            **(taskmodule_kwargs or {}),
        )

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
            taskmodule_kwargs=taskmodule_kwargs,
            model_kwargs=model_kwargs,
            device=device,
            binary_output=binary_output,
            **kwargs,
        )

        return pipeline
