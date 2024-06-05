import typing
import logging
import socket
import hashlib
from contextlib import closing, contextmanager

import pathlib as plb
import yaml
from pydantic import BaseModel
from google.cloud import secretmanager

logger = logging.getLogger("cajajejo.utils")


@contextmanager
def _gcp_secret_manager():
    """Context manager for GCP Secret Manager"""
    client = secretmanager.SecretManagerServiceClient()
    yield client


def hash_file(path: typing.Union[plb.Path, str]) -> str:
    with plb.Path(path).open("r") as f:
        content = f.read()
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def get_secret(secret_resource_name) -> str:
    """Retrieves the most recent secret version"""
    with _gcp_secret_manager() as client:
        secret_version = client.access_secret_version(
            request={"name": secret_resource_name}
        )
    return secret_version.payload.data.decode("UTF-8")


def check_path_exists(path: str) -> plb.Path:
    _path = plb.Path(path).resolve()
    if not _path.exists():
        raise FileNotFoundError(f"Config at '{path}' not found")
    return _path


class ConnectionError(Exception):
    """Exception raised for errors in the evaluation process."""

    pass


def load_config(path: typing.Union[str, plb.Path], config_cls: BaseModel) -> BaseModel:
    """Load a YAML configuration file from disk"""
    path = plb.Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config at '{path}' not found")
    with path.resolve().open("r") as inFile:
        cnf_raw = yaml.safe_load(inFile)
        cnf_out = config_cls(**cnf_raw)
    return cnf_out


def yaml_str_presenter(dumper, data):
    """YAML string presenter that preserves newlines"""
    if len(data.splitlines()) > 1 or "\n" in data:
        text_list = [line.rstrip() for line in data.splitlines()]
        # Check if last line finishes with newline
        if data.endswith("\n"):
            ends_with_newline = True
        else:
            ends_with_newline = False
        fixed_data = "\n".join(text_list)
        if ends_with_newline:
            fixed_data += "\n"
        return dumper.represent_scalar("tag:yaml.org,2002:str", fixed_data, style=">")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


def requires_extras(
    extras_installed: bool, extras_name: typing.Union[typing.List[str], str]
):
    """Decorator that raises an error if the required extras are not installed

    Parameters
    ----------
    extras_installed : bool
        flag indicating whether or not the extras are installed
    extras_name : typing.Union[typing.List[str], str]
        name of the extras that should be installed

    Raises
    ------
    ImportError
        if the extras are not installed

    Examples
    ------
    >>> from cajajejo.utils import requires_extras
    >>> try:
    >>>     import mlflow
    >>> except ImportError:
    >>>    _has_training_extras = False
    >>> else:
    >>>   _has_training_extras = True
    >>>
    >>> @requires_extras(_has_training_extras, "training")
    >>> def dummy(x):
    >>>     return x
    """

    def decorator(function):
        def wrapper(*args, **kwargs):
            if not extras_installed:
                if isinstance(extras_name, list):
                    pip_extras_str = f'[{",".join(extras_name)}]'
                    poetry_extras_str = " ".join(
                        [f"-E {extra}" for extra in extras_name]
                    )
                else:
                    pip_extras_str = f"[{extras_name}]"
                    poetry_extras_str = f"-E {extras_name}"
                raise ImportError(
                    f"""🚨 You need to install the `cajajejo{pip_extras_str}` extra to use this command.
                    If you're using poetry, use `poetry install {poetry_extras_str}`. If you're using pip,
                    use `pip install cajajejo{pip_extras_str}`"""
                )
            return function(*args, **kwargs)

        return wrapper

    return decorator


def check_host(host, port):
    """Check if we can reach a given host and port"""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        status = sock.connect_ex((host, port))
    return status


def parse_tags(tags: typing.Dict[str, typing.Any]):
    """Parse tags for MLFlow"""
    _tags = {}
    for k, v in tags.items():
        if isinstance(v, list):
            v = ",".join(v)
        if v is None:
            continue
        _tags[k] = v
    return _tags


class PandasDataFrame(typing.Protocol):
    """Duck-typing pandas dataframe because it's an extra but also want to type hint it"""

    ...


class HuggingFaceDataset(typing.Protocol):
    """Duck-typing HF Dataset because it's an extra but also want to type hint it"""

    ...
