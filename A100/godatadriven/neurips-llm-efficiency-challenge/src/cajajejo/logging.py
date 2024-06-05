import typing
import logging
import os

try:
    from google.cloud.logging import Client
except ImportError:
    _has_google_cloud_logging = False
else:
    _has_google_cloud_logging = True

try:
    import mlflow
except ImportError:
    _has_mlflow = False
else:
    _has_mlflow = True

from cajajejo import __version__


def get_mlflow_context() -> typing.Dict:
    """Retrieve mlflow context variables from active run and parse them into dict.

    Returns:
        typing.Dict: dict containing context variables
    """
    if not _has_mlflow:
        raise ImportError(
            """🚨 You need to install the `cajajejo[training]` or `cajajejo[evaluation]` extra to use this command.
            If you're using poetry, use `poetry install -E training` or `poetry install -E evaluation`."""
        )

    mlflow_context_vars = {}
    if mlflow.active_run() is not None:
        proto = mlflow.active_run().to_proto()
        mlflow_context_vars["mlflow_run_id"] = proto.info.run_id
        mlflow_context_vars["mlflow_experiment_id"] = proto.info.experiment_id
        mlflow_context_vars["mlflow_experiment_name"] = mlflow.get_experiment(
            proto.info.experiment_id
        ).name
        mlflow_context_vars["mlflow_artifact_uri"] = proto.info.artifact_uri
        for tag in proto.data.tags:
            if tag.key == "mlflow.runName":
                mlflow_context_vars["mlflow_run_name"] = tag.value
            if tag.key == "mlflow.user":
                mlflow_context_vars["mlflow_user"] = tag.value
    return mlflow_context_vars


class evaluationLogFilter(logging.Filter):
    """Log filter that automatically adds labels to log records"""

    def filter(self, record):
        if hasattr(record, "labels"):
            labels = record.labels.copy()
        else:
            labels = {}
        labels["job_name"] = os.environ.get("JOB_NAME", "Unknown")
        labels["image_build_cajajejo_version"] = __version__
        labels["config_git_sha"] = os.environ.get("CONFIG_VERSION", "Unknown")
        # NB: only get added if MLFlow run is active
        labels = {**labels, **get_mlflow_context()}
        record.labels = labels
        return True


def set_up_gcp_log_sink():
    """Set up Google Cloud Logging sink"""
    if not _has_google_cloud_logging:
        raise ImportError(
            """🚨 You need to install `google.cloud.logging` to use this command.
            Install using `pip install google-cloud-logging`."""
        )
    client = Client()
    client.setup_logging()
