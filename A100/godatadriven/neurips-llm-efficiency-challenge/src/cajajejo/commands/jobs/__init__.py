import logging

import typer

from .training import training_cmd
from .evaluation import evaluation_cmd

logger = logging.getLogger("cajajejo.commands.jobs")


jobs_cmd = typer.Typer(
    help="🚀 Command-line tools for deploying training & evaluation jobs to the GKE autopilot cluster",
    no_args_is_help=True,
)

jobs_cmd.add_typer(training_cmd, name="training")
jobs_cmd.add_typer(evaluation_cmd, name="evaluation")
