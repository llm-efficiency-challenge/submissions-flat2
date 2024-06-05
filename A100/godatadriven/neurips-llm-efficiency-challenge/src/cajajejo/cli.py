import logging
import warnings

import typer

from .commands import jobs_cmd, api_cmd, models_cmd, data_cmd
from . import __version__

warnings.filterwarnings("ignore")

logger = logging.getLogger("cajajejo")
handler = logging.StreamHandler()
format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
handler.setFormatter(format)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


app = typer.Typer(
    help="🧰 Command-line tools for NeurIPS 2023 LLM efficiency challenge Xebia Data submission",
    no_args_is_help=True,
)
app.add_typer(jobs_cmd, name="jobs")
app.add_typer(api_cmd, name="api-server")
app.add_typer(models_cmd, name="models")
app.add_typer(data_cmd, name="data")


@app.callback()
def main(trace: bool = False):
    if trace:
        logger.setLevel(logging.DEBUG)


@app.command(short_help="📌 Displays the current version number of the cajajejo library")
def version():
    print(__version__)


def entrypoint():
    app()
