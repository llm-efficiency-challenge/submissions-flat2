import logging

import typer
from cajajejo.commands.data.utils import write_data_remote

logger = logging.getLogger("cajajejo.commands.data.config")


config_cmd = typer.Typer(
    help="📋 Sub-commands for setting up training data config", no_args_is_help=True
)


@config_cmd.command(
    name="set-remote",
    help="",
    short_help="",
    no_args_is_help=True,
)
def _set_remote(gs_bucket: str, local_data_path: str):
    write_data_remote(gs_bucket, local_data_path)
