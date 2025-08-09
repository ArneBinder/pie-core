import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Type, TypeVar, Union

import requests
from huggingface_hub.constants import DEFAULT_ETAG_TIMEOUT
from huggingface_hub.file_download import hf_hub_download
from huggingface_hub.hf_api import HfApi
from huggingface_hub.utils import SoftTemporaryDirectory, validate_hf_hub_args

from pie_core.utils.dictionary import TNestedBoolDict, dict_update_nested

logger = logging.getLogger(__name__)


T = TypeVar("T", bound="HFHubProtocol")


class HFHubProtocol(Protocol):
    """Implementation of [`HFHubProtocol`] to provide basic HF and local upload/download
    functionality It is based on an early version of ModelHubMixin, see:

    https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/hub_mixin.py
    """

    config_name: str = "not_implemented.json"
    config_type_key: str = "not_implemented"

    def _config(self) -> Optional[Dict[str, Any]]:
        """This method should return dictionary with all class attributes needed to reproduce your
        object."""
        return None

    @property
    def has_config(self) -> bool:
        return self._config() is not None

    @property
    def config(self) -> Dict[str, Any]:
        return dict(self._config() or {})  # soft-copy to avoid mutating input

    def save_pretrained(
        self,
        save_directory: Union[str, Path],
        *,
        repo_id: Optional[str] = None,
        push_to_hub: bool = False,
        **kwargs,
    ) -> Optional[str]:
        """Save weights in local directory.

        Args:
            save_directory (`str` or `Path`):
                Path to directory in which the model weights and configuration will be saved.
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Huggingface Hub after saving it.
            repo_id (`str`, *optional*):
                ID of your repository on the Hub. Used only if `push_to_hub=True`. Will default to the folder name if
                not provided.
            kwargs:
                Additional key word arguments passed along to the [`~ModelHubMixin._from_pretrained`] method.
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        # saving model weights/files
        self._save_pretrained(save_directory)

        # saving config
        if self.has_config:
            (save_directory / self.config_name).write_text(json.dumps(self.config, indent=2))

        if push_to_hub:
            kwargs = kwargs.copy()  # soft-copy to avoid mutating input
            if self.has_config:  # kwarg for `push_to_hub`
                kwargs["config"] = self.config
            if repo_id is None:
                repo_id = save_directory.name  # Defaults to `save_directory` name
            return self.push_to_hub(repo_id=repo_id, **kwargs)
        return None

    def _save_pretrained(self, save_directory: Path) -> None:
        """Overwrite this method in subclass to define how to save your model. Check out our
        [integration guide](../guides/integrations) for instructions.

        Args:
            save_directory (`str` or `Path`):
                Path to directory in which the model weights and configuration will be saved.
        """
        raise NotImplementedError

    @classmethod
    def retrieve_config_file(
        cls,
        model_id: Union[str, Path],
        fail_silently: bool = False,
        **hub_download_kwargs: Any,
    ) -> Optional[str]:
        """Retrieve the configuration file from the Huggingface Hub or local directory.

        Returns None if the config file is not found.
        """

        config_file: Optional[str] = None
        if os.path.isdir(model_id):
            if cls.config_name in os.listdir(model_id):
                config_file = os.path.join(model_id, cls.config_name)
            else:
                logger.warning(f"{cls.config_name} not found in {Path(model_id).resolve()}")
        elif isinstance(model_id, str):
            try:
                config_file = hf_hub_download(
                    repo_id=str(model_id), filename=cls.config_name, **hub_download_kwargs
                )
            except requests.exceptions.RequestException:
                if not fail_silently:
                    logger.warning(f"{cls.config_name} not found in HuggingFace Hub.")

        return config_file

    @classmethod
    @validate_hf_hub_args
    def from_pretrained(
        cls: Type[T],
        pretrained_model_name_or_path: Union[str, Path],
        *,
        subfolder: Optional[str] = None,
        repo_type: Optional[str] = None,
        revision: Optional[str] = None,
        library_name: Optional[str] = None,
        library_version: Optional[str] = None,
        cache_dir: Union[str, Path, None] = None,
        local_dir: Union[str, Path, None] = None,
        user_agent: Union[Dict, str, None] = None,
        force_download: bool = False,
        proxies: Optional[Dict] = None,
        etag_timeout: float = DEFAULT_ETAG_TIMEOUT,
        token: Union[bool, str, None] = None,
        local_files_only: bool = False,
        headers: Optional[Dict[str, str]] = None,
        endpoint: Optional[str] = None,
        **model_kwargs,
    ) -> T:
        """Download a model from the Huggingface Hub and instantiate it.

        Args:
            pretrained_model_name_or_path (`str`, `Path`):
                - Either the `model_id` (string) of a model hosted on the Hub, e.g. `bigscience/bloom`.
                - Or a path to a `directory` containing model weights saved using
                    [`~transformers.PreTrainedModel.save_pretrained`], e.g., `../path/to/my_model_directory/`.
            subfolder (`str`, *optional*):
                An optional value corresponding to a folder inside the model repo.
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if downloading from a dataset or space,
                `None` or `"model"` if downloading from a model. Default is `None`.
            revision (`str`, *optional*):
                An optional Git revision id which can be a branch name, a tag, or a
                commit hash.
            library_name (`str`, *optional*):
                The name of the library to which the object corresponds.
            library_version (`str`, *optional*):
                The version of the library.
            cache_dir (`str`, `Path`, *optional*):
                Path to the folder where cached files are stored.
            local_dir (`str` or `Path`, *optional*):
                If provided, the downloaded file will be placed under this directory.
            user_agent (`dict`, `str`, *optional*):
                The user-agent info in the form of a dictionary or a string.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether the file should be downloaded even if it already exists in
                the local cache.
            proxies (`dict`, *optional*):
                Dictionary mapping protocol to the URL of the proxy passed to
                `requests.request`.
            etag_timeout (`float`, *optional*, defaults to `10`):
                When fetching ETag, how many seconds to wait for the server to send
                data before giving up which is passed to `requests.request`.
            token (`str`, `bool`, *optional*):
                A token to be used for the download.
                    - If `True`, the token is read from the HuggingFace config
                      folder.
                    - If a string, it's used as the authentication token.
            local_files_only (`bool`, *optional*, defaults to `False`):
                If `True`, avoid downloading the file and return the path to the
                local cached file if it exists.
            headers (`dict`, *optional*):
                Additional headers to be sent with the request.
            model_kwargs (`Dict`, *optional*):
                Additional kwargs to pass to the model during initialization.
        """
        model_id = pretrained_model_name_or_path

        config_file = cls.retrieve_config_file(
            model_id=model_id,
            subfolder=subfolder,
            repo_type=repo_type,
            revision=revision,
            library_name=library_name,
            library_version=library_version,
            cache_dir=cache_dir,
            local_dir=local_dir,
            user_agent=user_agent,
            force_download=force_download,
            proxies=proxies,
            etag_timeout=etag_timeout,
            token=token,
            local_files_only=local_files_only,
            headers=headers,
            endpoint=endpoint,
            fail_silently=False,
        )

        if config_file is not None:
            with open(config_file, encoding="utf-8") as f:
                config = json.load(f)
            model_kwargs.update({"config": config})

        # The value of is_from_pretrained is set to True when the model is loaded from pretrained.
        # Note that the value may be already available in model_kwargs.
        model_kwargs["is_from_pretrained"] = True

        return cls._from_pretrained(
            model_id=str(model_id),
            subfolder=subfolder,
            repo_type=repo_type,
            revision=revision,
            library_name=library_name,
            library_version=library_version,
            cache_dir=cache_dir,
            local_dir=local_dir,
            user_agent=user_agent,
            force_download=force_download,
            proxies=proxies,
            etag_timeout=etag_timeout,
            token=token,
            local_files_only=local_files_only,
            headers=headers,
            endpoint=endpoint,
            **model_kwargs,
        )

    @classmethod
    def _from_pretrained(
        cls: Type[T],
        *,
        model_id: str,
        subfolder: Optional[str] = None,
        repo_type: Optional[str] = None,
        revision: Optional[str] = None,
        library_name: Optional[str] = None,
        library_version: Optional[str] = None,
        cache_dir: Union[str, Path, None] = None,
        local_dir: Union[str, Path, None] = None,
        user_agent: Union[Dict, str, None] = None,
        force_download: bool = False,
        proxies: Optional[Dict] = None,
        etag_timeout: float = DEFAULT_ETAG_TIMEOUT,
        token: Union[bool, str, None] = None,
        local_files_only: bool = False,
        headers: Optional[Dict[str, str]] = None,
        endpoint: Optional[str] = None,
        **model_kwargs,
    ) -> T:
        """Overwrite this method in subclass to define how to load your model from pretrained.

        Use [`hf_hub_download`] or [`snapshot_download`] to download files from the Hub before loading them. Most
        args taken as input can be directly passed to those 2 methods. If needed, you can add more arguments to this
        method using "model_kwargs". For example [`PyTorchModelHubMixin._from_pretrained`] takes as input a `map_location`
        parameter to set on which device the model should be loaded.

        Check out our [integration guide](../guides/integrations) for more instructions.

        Args:
            model_id (`str`):
                ID of the model to load from the Huggingface Hub (e.g. `bigscience/bloom`).
            revision (`str`, *optional*):
                Revision of the model on the Hub. Can be a branch name, a git tag or any commit id. Defaults to the
                latest commit on `main` branch.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether to force (re-)downloading the model weights and configuration files from the Hub, overriding
                the existing cache.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether to delete incompletely received files. Will attempt to resume the download if such a file exists.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint (e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`).
            token (`str` or `bool`, *optional*):
                The token to use as HTTP bearer authorization for remote files. By default, it will use the token
                cached when running `huggingface-cli login`.
            cache_dir (`str`, `Path`, *optional*):
                Path to the folder where cached files are stored.
            local_files_only (`bool`, *optional*, defaults to `False`):
                If `True`, avoid downloading the file and return the path to the local cached file if it exists.
            model_kwargs:
                Additional keyword arguments passed along to the [`~ModelHubMixin._from_pretrained`] method.
        """
        raise NotImplementedError

    @validate_hf_hub_args
    def push_to_hub(
        self,
        repo_id: str,
        *,
        config: Optional[dict] = None,
        api_endpoint: Optional[str] = None,
        private: bool = False,
        token: Union[bool, str, None] = None,
        **upload_folder_kwargs,
    ) -> str:
        """Upload model checkpoint to the Hub.

        See [`upload_folder`] reference for more details.

        Args:
            repo_id (`str`):
                ID of the repository to push to (example: `"username/my-model"`).
            config (`dict`, *optional*):
                Configuration object to be saved alongside the model weights.
            api_endpoint (`str`, *optional*):
                The API endpoint to use when pushing the model to the hub.
            private (`bool`, *optional*, defaults to `False`):
                Whether the repository created should be private.
            token (`str`, *optional*):
                The token to use as HTTP bearer authorization for remote files. By default, it will use the token
                cached when running `huggingface-cli login`.

        Returns:
            The url of the commit of your model in the given repository.
        """
        api = HfApi(endpoint=api_endpoint, token=token)
        repo_id = api.create_repo(repo_id=repo_id, private=private, exist_ok=True).repo_id

        # Push the files to the repo in a single commit
        with SoftTemporaryDirectory() as tmp:
            saved_path = Path(tmp) / repo_id
            self.save_pretrained(saved_path, config=config)
            return api.upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=saved_path,
                **upload_folder_kwargs,
            )

    @classmethod
    def from_config(cls: Type[T], config: dict, **kwargs) -> T:
        """Instantiate from a configuration object.

        Args:
            config (`dict`):
                The configuration object to instantiate.
            kwargs:
                Additional keyword arguments passed along to the specific model class.
        """
        config = config.copy()
        # remove config_type_key entry, e.g. model_type, from config and kwargs, if present
        config.pop(cls.config_type_key, None)
        kwargs.pop(cls.config_type_key, None)
        return cls._from_config(config=config, **kwargs)

    @classmethod
    def _from_config(
        cls: Type[T], config: dict, config_override: Optional[TNestedBoolDict] = None, **kwargs
    ) -> T:
        """Instantiate from a configuration object.

        Args:
            config (`dict`):
                The configuration object to instantiate.
            kwargs:
                Additional keyword arguments passed along to the specific model class.
        """
        config = config.copy()
        dict_update_nested(config, kwargs, override=config_override)
        return cls(**config)


class HFHubMixin(HFHubProtocol):
    """This mixin provides basic HF and local config upload/download functionality for models,
    taskmodules and pipelines."""

    def __init__(self, *args, is_from_pretrained: bool = False, **kwargs):
        # skip the __init__ of HFHubProtocol: this would interrupt the
        # constructor chain and disallow passing the args and kwargs to
        # any other class in the case of multiple inheritance
        super(HFHubProtocol, self).__init__(*args, **kwargs)
        self._is_from_pretrained = is_from_pretrained

    @property
    def is_from_pretrained(self) -> bool:
        return self._is_from_pretrained
