import logging

try:
    import mlflow
except ImportError:
    _has_training_extras = False
else:
    _has_training_extras = True

from cajajejo.utils import requires_extras

logger = logging.getLogger("cajajejo.models.inference.utils")


@requires_extras(_has_training_extras, "training")
def download_mlflow_artifact(run_id, artifact_path, dst_path):
    """Utility for downloading MLFlow artifacts from a run to a local path"""
    logger.info(
        f"Downloading artifact {artifact_path} from run {run_id} to {dst_path}."
    )
    mlflow_client = mlflow.tracking.MlflowClient()
    mlflow_client.download_artifacts(run_id, artifact_path, dst_path)
