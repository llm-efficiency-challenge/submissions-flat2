import logging

import typer

from .config import config_cmd
from .utils import (
    upload_dataset_version,
    list_dataset_versions,
    latest_version,
    download_dataset_version,
)

logger = logging.getLogger("cajajejo.commands.data.management")


data_cmd = typer.Typer(
    help="📊 Commands for interacting with training data",
    no_args_is_help=True,
)
data_cmd.add_typer(config_cmd, name="config")


@data_cmd.command(
    name="push",
    help="Push data to a remote bucket.",
    short_help="Push data to a remote bucket.",
    no_args_is_help=True,
)
def _push(
    path_to_data: str = typer.Argument(None, help="Path to a JSONL dataset."),
    commit_message: str = typer.Option(
        "Updated data", "--commit-message", "-m", help="Commit message for the data."
    ),
):
    if not path_to_data.endswith(".jsonl"):
        logger.error("Data must be in JSONL format.")
        raise ValueError("Data must be in JSONL format.")
    upload_dataset_version(path_to_data, commit_message)


@data_cmd.command(
    name="latest-version",
    help="Get the latest version of the dataset.",
    short_help="Get the latest version of the dataset.",
    no_args_is_help=False,
)
def _latest_version():
    latest_version()


@data_cmd.command(
    name="list",
    help="List all versions of the dataset.",
    short_help="List all versions of the dataset.",
    no_args_is_help=False,
)
def _list_versions():
    list_dataset_versions()


@data_cmd.command(
    name="pull",
    help="Download data from a remote bucket to the local data folder.",
    short_help="Download data from a remote bucket to the local data folder.",
    no_args_is_help=False,
)
def _pull(gs_uri: str):
    download_dataset_version(gs_uri)
