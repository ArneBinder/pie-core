import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type, TypeVar, Union

from huggingface_hub.file_download import hf_hub_download

from pie_core.auto import Auto
from pie_core.hf_hub_mixin import HFHubMixin
from pie_core.registrable import Registrable

logger = logging.getLogger(__name__)

TModelHFHubMixin = TypeVar("TModelHFHubMixin", bound="ModelHFHubMixin")


class ModelHFHubMixin(HFHubMixin):
    """Implementation of [`ModelHFHubMixin`] to provide model Hub upload/download capabilities to
    models.

    Example for a Pytorch model:

    ```python
    >>> import torch
    >>> import torch.nn as nn
    >>> from pie_core import Model


    >>> class MyModel(Model, nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.param = nn.Parameter(torch.rand(3, 4))
    ...         self.linear = nn.Linear(4, 5)
    ...
    ...     def forward(self, x):
    ...         return self.linear(x + self.param)
    ...
    ...     def save_model_file(self, model_file: str) -> None:
    ...         torch.save(self.state_dict(), model_file)
    ...
    ...     def load_model_file(
    ...         self, model_file: str, map_location: str = "cpu", strict: bool = False
    ...     ) -> None:
    ...         state_dict = torch.load(model_file, map_location=torch.device(map_location))
    ...         self.load_state_dict(state_dict, strict=strict)

    >>> model = MyModel()

    # Save model weights to local directory
    >>> model.save_pretrained("my-awesome-model")

    # Push model weights to the Hub
    >>> model.push_to_hub("my-awesome-model")

    # Download and initialize weights from the Hub
    >>> model = MyModel.from_pretrained("username/my-awesome-model")
    ```
    """

    config_name = "config.json"
    config_type_key = "model_type"
    weights_file_name = "model.bin"

    def save_model_file(self, model_file: str) -> None:
        """Save weights from a model to a local directory."""
        raise NotImplementedError

    def load_model_file(self, model_file: str, **kwargs) -> None:
        """Load weights from a model file."""
        raise NotImplementedError

    def _save_pretrained(self, save_directory: Path) -> None:
        """Save weights from a model to a local directory."""
        self.save_model_file(str(save_directory / self.weights_file_name))

    @classmethod
    def retrieve_model_file(
        cls,
        model_id: str,
        revision: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        force_download: bool = False,
        proxies: Optional[Dict] = None,
        resume_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
    ) -> str:
        """Retrieve the model file from the Huggingface Hub or local directory."""
        if os.path.isdir(model_id):
            logger.info("Loading weights from local directory")
            model_file = os.path.join(model_id, cls.weights_file_name)
        else:
            model_file = hf_hub_download(
                repo_id=model_id,
                filename=cls.weights_file_name,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                token=token,
                local_files_only=local_files_only,
            )

        return model_file

    @classmethod
    def _from_pretrained(
        cls: Type[TModelHFHubMixin],
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
        load_model_file: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> TModelHFHubMixin:

        load_model_file_kwargs = load_model_file or {}
        if "map_location" in kwargs:
            map_location = kwargs.pop("map_location")
            logger.warning(
                f'map_location is deprecated. Use load_model_file={{"map_location": "{map_location}"}} instead.'
            )
            load_model_file_kwargs["map_location"] = map_location
        if "strict" in kwargs:
            strict = kwargs.pop("strict")
            logger.warning(
                f'strict is deprecated. Use load_model_file={{"strict": {strict}}} instead.'
            )
            load_model_file_kwargs["strict"] = strict

        model = cls.from_config(config=config or {}, **kwargs)
        model_file = model.retrieve_model_file(
            model_id=model_id,
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            local_files_only=local_files_only,
            token=token,
        )
        # load the model weights
        model.load_model_file(model_file, **load_model_file_kwargs)

        return model


class Model(ModelHFHubMixin, Registrable["Model"]):

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
                " with AutoModel.from_pretrained() or"
                " AutoModel.from_config(). Consider to annotate the class with"
                " @Model.register() or @Model.register(name='...') to register it at as a Model"
                " which will allow to load it via AutoModel."
            )

        return config


class AutoModel(ModelHFHubMixin, Auto[Model]):

    BASE_CLASS = Model
