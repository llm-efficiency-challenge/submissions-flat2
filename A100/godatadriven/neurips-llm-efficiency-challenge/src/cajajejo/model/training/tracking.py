from contextlib import contextmanager
import logging
import os
import pathlib as plb

try:
    import mlflow
except ImportError:
    _has_training_extras = False
else:
    _has_training_extras = True

from cajajejo import __version__
from cajajejo.utils import load_config, parse_tags, requires_extras
from cajajejo.config import TrackingConfig, TrainingConfig

logger = logging.getLogger("cajajejo.training.tracking")


@requires_extras(_has_training_extras, "training")
def configure_mlflow(tracking_config: TrackingConfig):
    """Set up MLFlow tracking so that TRL SFTrainer logs to MLFlow."""
    logger.info(
        f"Setting to tracking URI {tracking_config.tracking_uri} and experiment name {tracking_config.experiment_name}."
    )
    mlflow.set_tracking_uri(tracking_config.tracking_uri)
    os.environ["HF_MLFLOW_LOG_ARTIFACTS"] = "1"
    os.environ["MLFLOW_EXPERIMENT_NAME"] = tracking_config.experiment_name


@requires_extras(_has_training_extras, "training")
@contextmanager
def neurips_mlflow_tracking(config_path: str):
    """Context manager for MLFlow tracking of NeurIPS training runs. This does some initial setup, logging and cleanup."""
    config = load_config(config_path, TrainingConfig)
    configure_mlflow(config.tracking_config)
    final_model_path = (
        plb.Path(config.train_arguments_config.output_dir) / "final_model"
    )
    try:
        run = mlflow.start_run(
            tags=parse_tags(config.tracking_config.tags.dict()),
            description=config.tracking_config.description,
        )
        logger.info(f"Logging to MLFlow run {run.info.run_id}.")
        mlflow.log_param("cajajejo_version", __version__)
        mlflow.log_artifact(config_path)
        yield run
    finally:
        if final_model_path.exists():
            logger.info("Logging final model to MLFlow.")
            mlflow.log_artifacts(str(final_model_path), artifact_path="final_model")
        mlflow.end_run()
