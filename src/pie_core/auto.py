import os
from pathlib import Path
from typing import Dict, Optional, Type, Union

from huggingface_hub.file_download import hf_hub_download

from pie_core.hf_hub_mixin import (
    PieModelHFHubMixin,
    PieTaskModuleHFHubMixin,
    TOverride,
    dict_update_nested,
)
from pie_core.model import PyTorchIEModel
from pie_core.taskmodule import TaskModule


class AutoModel(PieModelHFHubMixin):
    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: Optional[str],
        cache_dir: Optional[Union[str, Path]],
        force_download: bool,
        proxies: Optional[Dict],
        resume_download: bool,
        local_files_only: bool,
        token: Union[str, bool, None],
        map_location: str = "cpu",
        strict: bool = False,
        config: Optional[dict] = None,
        config_override: Optional[TOverride] = None,
        **model_kwargs,
    ) -> PyTorchIEModel:
        """Overwrite this method in case you wish to initialize your model in a different way."""

        config = (config or {}).copy()
        dict_update_nested(config, model_kwargs, override=config_override)
        class_name = config.pop(cls.config_type_key)
        clazz = PyTorchIEModel.by_name(class_name)
        model = clazz(**config)
        """Load Pytorch pretrained weights and return the loaded model."""
        if os.path.isdir(model_id):
            model_file = os.path.join(model_id, model.weights_file_name)
        else:
            model_file = hf_hub_download(
                repo_id=model_id,
                filename=model.weights_file_name,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                token=token,
                local_files_only=local_files_only,
            )

        model.load_model_file(model_file, map_location=map_location, strict=strict)

        return model

    @classmethod
    def from_config(cls, config: dict, **kwargs) -> PyTorchIEModel:
        """Build a model from a config dict."""
        config = config.copy()
        class_name = config.pop(cls.config_type_key)
        clazz = PyTorchIEModel.by_name(class_name)
        return clazz._from_config(config, **kwargs)


class AutoTaskModule(PieTaskModuleHFHubMixin):
    @classmethod
    def _from_pretrained(  # type: ignore
        cls,
        *,
        model_id: str,
        revision: Optional[str],
        cache_dir: Optional[Union[str, Path]],
        force_download: bool,
        proxies: Optional[Dict],
        resume_download: bool,
        local_files_only: bool,
        token: Union[str, bool, None],
        map_location: str = "cpu",
        strict: bool = False,
        config: Optional[dict] = None,
        config_override: Optional[TOverride] = None,
        **taskmodule_kwargs,
    ) -> TaskModule:
        config = (config or {}).copy()
        dict_update_nested(config, taskmodule_kwargs)
        class_name = config.pop(cls.config_type_key)
        clazz: Type[TaskModule] = TaskModule.by_name(class_name)
        taskmodule = clazz(**config)
        taskmodule.post_prepare()
        return taskmodule

    @classmethod
    def from_config(cls, config: dict, **kwargs) -> TaskModule:
        """Build a task module from a config dict."""
        config = config.copy()
        class_name = config.pop(cls.config_type_key)
        clazz: Type[TaskModule] = TaskModule.by_name(class_name)
        return clazz._from_config(config, **kwargs)
