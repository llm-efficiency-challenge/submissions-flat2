import os
import logging
import pathlib as plb
from contextlib import contextmanager

try:
    import pendulum
except ImportError:
    _has_cli_extras = False
else:
    _has_cli_extras = True
from coolname import generate_slug
import yaml
from google.cloud import storage
import pandas as pd

from cajajejo.utils import load_config, check_path_exists, hash_file
from cajajejo.config import DataVersionConfig, DataConfig
from cajajejo.utils import requires_extras

logger = logging.getLogger("cajajejo.commands.data.utils")


@contextmanager
def gcs_bucket(bucket_name: str):
    """Context manager for GCP Storage"""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        yield bucket
    finally:
        storage_client.close()


@contextmanager
def _data_version_config() -> plb.Path:
    plb_data_version_config = plb.Path(".data").resolve()
    plb_data_version_config.mkdir(exist_ok=True)
    config_file = plb_data_version_config / "version.yml"
    yield config_file


@contextmanager
def _data_remote_config() -> str:
    plb_data_config = plb.Path(".data").resolve()
    plb_data_config.mkdir(exist_ok=True)
    config_file = plb_data_config / "config.yml"
    yield config_file


def write_data_remote(remote: str, local_data_path: str):
    with _data_remote_config() as config_file:
        config = DataConfig(remote=remote, local_data_path=local_data_path)
        with config_file.open("w") as f:
            yaml.dump(config.dict(), f)


@requires_extras(_has_cli_extras, "dev")
def write_version_config(data_hash: str, slug: str, gs_uri: str) -> bool:
    with _data_version_config() as config_file:
        if config_file.exists():
            logger.debug(f"Found existing config file at {str(config_file)}")
            config = load_config(config_file, DataVersionConfig)
            hash_current = config.hash
            if hash_current != data_hash:
                logger.debug("Config has changed.")
                logger.debug(f"Old hash: {hash_current}")
                logger.debug(f"New hash: {data_hash}")
                _write_config = True
            else:
                logger.info("Data has not changed.")
                _write_config = False
        else:
            _write_config = True
        if _write_config:
            config = DataVersionConfig(
                hash=data_hash,
                slug=slug,
                created=pendulum.now("Europe/Amsterdam").isoformat(),
                gs_uri=gs_uri,
            )
            with config_file.open("w") as f:
                yaml.dump(config.dict(), f)
        return _write_config


def requires_remote(function):
    def decorator(*args, **kwargs):
        with _data_remote_config() as config:
            if not config.exists():
                raise FileNotFoundError(
                    f"Config file not found at {str(config)}. Please run `cajajejo data config set-remote` to set up the config file."
                )
            return function(*args, **kwargs)

    return decorator


@requires_remote
def _list_data_versions() -> pd.DataFrame:
    with _data_remote_config() as remote_config:
        remote = load_config(remote_config, DataConfig).remote
        logger.debug(f"Found remote at {remote}")
    bucket = remote.strip("gs://").split("/")[0]
    folder = "/".join(remote.strip("gs://").split("/")[1:]) + "/"
    logger.debug(f"Remote path: {folder}")
    files = []
    with gcs_bucket(bucket) as bclient:
        blobs = bclient.list_blobs(prefix=folder)
        for blob in blobs:
            logger.debug(f"Found blob: {blob.name}")
            if blob.name.endswith("commit.txt"):
                continue
            _, slug, hash, fn = blob.name.split("/")
            files.append(
                {
                    "slug": slug,
                    "hash": hash,
                    "filename": fn,
                    "updated": blob.updated,
                }
            )
    df = pd.DataFrame(files)
    return df.sort_values("updated", ascending=False)


@requires_remote
def upload_dataset_version(path_to_dataset: str, commit_message: str):
    _path = check_path_exists(path_to_dataset)
    logger.debug(f"Found dataset at {str(_path)}")
    # Check extension (should be json file)
    with _data_remote_config() as remote_config:
        remote = load_config(remote_config, DataConfig).remote
        logger.debug(f"Found remote at {remote}")
    # Get data hash
    fn = _path.name
    hash = hash_file(_path)
    slug = generate_slug(2)
    bucket = remote.strip("gs://").split("/")[0]
    folder = "/".join(remote.strip("gs://").split("/")[1:])
    ds_remote_path = os.path.join(folder, slug, hash, fn)
    logger.debug(f"Remote path: {ds_remote_path}")
    commit_file = os.path.join(folder, slug, "commit.txt")
    # Push data, then write version. Should be atomic.
    config_changed = write_version_config(hash, slug, ds_remote_path)
    if config_changed:
        with gcs_bucket(bucket) as bclient:
            blob = bclient.blob(ds_remote_path)
            blob.upload_from_filename(_path)
            blob_commit = bclient.blob(commit_file)
            blob_commit.upload_from_string(commit_message)
        logger.debug(f"Uploaded dataset to {ds_remote_path}")


@requires_remote
def download_dataset_version(gs_uri: str):
    with _data_remote_config() as remote_config:
        config = load_config(remote_config, DataConfig)
        remote = config.remote
        local_data_path = config.local_data_path
        logger.debug(f"Found remote at {remote}")
    bucket = remote.strip("gs://").split("/")[0]
    remote_dataset = gs_uri.strip(remote)
    slug, hash, _ = remote_dataset.split("/")
    with gcs_bucket(bucket) as bclient:
        blobs = bclient.list_blobs(prefix=os.path.join(remote.strip("gs://"), slug))
        for blob in blobs:
            logger.debug(f"Found blob: {blob.name}")
            local_dataset = (
                plb.Path(local_data_path) / slug / hash / blob.name.split("/")[-1]
            )
            local_dataset.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(local_dataset)
            logger.debug(f"Downloaded dataset to {local_dataset}")


@requires_remote
def list_dataset_versions():
    df = _list_data_versions()
    print(df)


@requires_remote
def latest_version():
    df = _list_data_versions()
    with _data_remote_config() as remote_config:
        remote = load_config(remote_config, DataConfig).remote
    slug = df["slug"].iloc[0]
    hash = df["hash"].iloc[0]
    fn = df["filename"].iloc[0]
    print(f"{remote}/{slug}/{hash}/{fn}")
