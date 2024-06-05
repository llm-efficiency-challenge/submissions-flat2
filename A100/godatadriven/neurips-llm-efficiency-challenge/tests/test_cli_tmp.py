# import pathlib as plb
# import os
# import logging
# import json

# import pytest
# import yaml
# import mlflow
# from cajajejo.config import (
#     EvaluationConfig,
#     ComputeConfig,
#     MlflowConfig,
#     ModelConfig,
#     TrackingConfig,
#     TrackingConfigTags,
#     HelmConfig,
#     MlflowArtifactConfig,
# )
# from cajajejo.utils import yaml_str_presenter
# from cajajejo import __version__

# import cli_utils

# yaml.add_representer(str, yaml_str_presenter)
# yaml.representer.SafeRepresenter.add_representer(str, yaml_str_presenter)

# os.environ["CONFIG_VERSION"] = "hfir4832478gyegfg"
# os.environ["CONFIG_PATH_GIT_REPO"] = "/path/to/config.yml"
# os.environ["HELM_CONFIG_PATH_GIT_REPO"] = "/path/to/helm/config.conf"
# os.environ["JOB_NAME"] = "test-job"


# @pytest.fixture(scope="session")
# def mlflow_tracking_server(tmp_path_factory: plb.Path) -> plb.Path:
#     mlflow_server_path = tmp_path_factory.mktemp("mlflow")  # type: ignore
#     return mlflow_server_path


# @pytest.fixture(scope="session")
# def helm_run_output_path(tmp_path_factory: plb.Path) -> plb.Path:
#     helm_run_output = tmp_path_factory.mktemp("benchmark_outputs")  # type: ignore
#     (helm_run_output / "test_runs.txt").touch()
#     return helm_run_output


# @pytest.fixture
# def evaluation_config(mlflow_tracking_server: plb.Path) -> EvaluationConfig:
#     return EvaluationConfig(
#         version="v5",
#         image="test/image",
#         image_tag="latest",
#         compute_config=ComputeConfig(
#             accelerator="nvidia-tesla-t4", cpu="1", memory="1Gi"
#         ),
#         helm=HelmConfig(suite="test-suite", max_eval_instances=10),
#         model_config=ModelConfig(
#             mlflow_artifact=MlflowArtifactConfig(
#                 run_id="test-run-id",
#                 artifact_path="model",
#                 model_directory="model",
#                 tokenizer_directory="components/tokenizer",
#             ),
#         ),
#         mlflow_config=MlflowConfig(tracking_uri=str(mlflow_tracking_server)),
#         tracking_config=TrackingConfig(
#             experiment_name="test-experiment",
#             description="test-description",
#             run_name=None,
#             tags=TrackingConfigTags(
#                 creator="pytest",
#                 model="test-model",
#                 lifecycle="testing",
#             ),
#         ),
#     )


# @pytest.fixture
# def evaluation_config_on_disk(
#     tmp_path: plb.Path, evaluation_config: EvaluationConfig
# ) -> str:
#     config_path = tmp_path / "config.yml"
#     with config_path.open("w") as f:
#         yaml.dump(evaluation_config.dict(), f)
#     return str(config_path)


# @pytest.fixture()
# def helm_config_on_disk(tmp_path: plb.Path) -> str:
#     config_path = tmp_path / "run_specs.conf"
#     with config_path.open("w") as f:
#         f.write(
#             """entries: [{description: "mmlu:subject=philosophy,model=huggingface/gpt2", priority: 1}]\n"""
#         )
#     return str(config_path)


# @pytest.fixture()
# def mlflow_model_directory(tmp_path: plb.Path) -> plb.Path:
#     model_directory = tmp_path / "artifacts" / "model"
#     model_directory.mkdir(parents=True)
#     return model_directory


# @pytest.fixture()
# def mlflow_tokenizer_directory(tmp_path: plb.Path) -> plb.Path:
#     tokenizer_directory = tmp_path / "artifacts" / "components" / "tokenizer"
#     tokenizer_directory.mkdir(parents=True)
#     return tokenizer_directory


# @pytest.fixture()
# def logger() -> logging.Logger:
#     logger = logging.getLogger("test_logger")
#     ch = logging.StreamHandler()
#     ch.setLevel(logging.DEBUG)
#     ch.addFilter(cli_utils.evaluationLogFilter())
#     logger.addHandler(ch)
#     logger.setLevel(logging.DEBUG)
#     return logger


# @pytest.fixture()
# def core_scenario_metrics():
#     return [
#         {
#             "header": [
#                 {"value": "Model/adapter", "markdown": False, "metadata": {}},
#                 {
#                     "value": "Mean win rate",
#                     "description": "How many models this model outperform on average (over columns).",
#                     "markdown": False,
#                     "lower_is_better": False,
#                     "metadata": {},
#                 },
#                 {
#                     "value": "MMLU - EM",
#                     "description": "The Massive Multitask Language Understanding (MMLU) benchmark for knowledge-intensive question answering across 57 domains [(Hendrycks et al., 2021)](https://openreview.net/forum?id=d7KBjmI3GmQ).\n\nExact match: Fraction of instances that the predicted output matches a correct reference exactly.",
#                     "markdown": False,
#                     "lower_is_better": False,
#                     "metadata": {"metric": "EM", "run_group": "MMLU"},
#                 },
#             ],
#             "rows": [
#                 [
#                     {
#                         "value": "huggingface/model",
#                         "description": "",
#                         "markdown": False,
#                     },
#                     {"markdown": False},
#                     {
#                         "value": 0.335978835978836,
#                         "description": "min=0.222, mean=0.336, max=0.5, sum=1.008 (3)",
#                         "style": {"font-weight": "bold"},
#                         "markdown": False,
#                     },
#                 ]
#             ],
#             "title": "Accuracy",
#         },
#         {
#             "header": [
#                 {"value": "Model/adapter", "markdown": False, "metadata": {}},
#                 {
#                     "value": "Mean win rate",
#                     "description": "How many models this model outperform on average (over columns).",
#                     "markdown": False,
#                     "lower_is_better": False,
#                     "metadata": {},
#                 },
#                 {
#                     "value": "MMLU - ECE (10-bin)",
#                     "description": "The Massive Multitask Language Understanding (MMLU) benchmark for knowledge-intensive question answering across 57 domains [(Hendrycks et al., 2021)](https://openreview.net/forum?id=d7KBjmI3GmQ).\n\n10-bin expected calibration error: The average difference between the model's confidence and accuracy, averaged across 10 bins where each bin contains an equal number of points (only computed for classification tasks). Warning - not reliable for small datasets (e.g., with < 300 examples) because each bin will have very few examples.",
#                     "markdown": False,
#                     "lower_is_better": True,
#                     "metadata": {"metric": "ECE (10-bin)", "run_group": "MMLU"},
#                 },
#             ],
#             "rows": [
#                 [
#                     {
#                         "value": "huggingface/model",
#                         "description": "",
#                         "markdown": False,
#                     },
#                     {"markdown": False},
#                     {
#                         "value": 0.6640211640211641,
#                         "description": "min=0.5, mean=0.664, max=0.778, sum=1.992 (3)",
#                         "style": {"font-weight": "bold"},
#                         "markdown": False,
#                     },
#                 ]
#             ],
#             "title": "Calibration",
#         },
#         {
#             "header": [
#                 {"value": "Model/adapter", "markdown": False, "metadata": {}},
#                 {
#                     "value": "Mean win rate",
#                     "description": "How many models this model outperform on average (over columns).",
#                     "markdown": False,
#                     "lower_is_better": False,
#                     "metadata": {},
#                 },
#                 {
#                     "value": "MMLU - EM (Robustness)",
#                     "description": "The Massive Multitask Language Understanding (MMLU) benchmark for knowledge-intensive question answering across 57 domains [(Hendrycks et al., 2021)](https://openreview.net/forum?id=d7KBjmI3GmQ).\n\nExact match: Fraction of instances that the predicted output matches a correct reference exactly.\n- Perturbation Robustness: Computes worst case over different robustness perturbations (misspellings, formatting, contrast sets).",
#                     "markdown": False,
#                     "lower_is_better": False,
#                     "metadata": {
#                         "metric": "EM",
#                         "run_group": "MMLU",
#                         "perturbation": "Robustness",
#                     },
#                 },
#             ],
#             "rows": [
#                 [
#                     {
#                         "value": "huggingface/model",
#                         "description": "",
#                         "markdown": False,
#                     },
#                     {"markdown": False},
#                     {
#                         "value": 0.2943121693121693,
#                         "description": "min=0.222, mean=0.294, max=0.375, sum=0.883 (3)",
#                         "style": {"font-weight": "bold"},
#                         "markdown": False,
#                     },
#                 ]
#             ],
#             "title": "Robustness",
#         },
#     ]


# @pytest.fixture()
# def core_scenario_metrics_on_disk(tmp_path: plb.Path, core_scenario_metrics: dict):
#     out_path = plb.Path(tmp_path) / "core_metrics.json"
#     with out_path.open("w") as f:
#         json.dump(core_scenario_metrics, f)
#     return str(out_path)


# def test_parse_tags():
#     tags = cli_utils._parse_tags(
#         {
#             "creator": "pytest",
#             "datasets": ["test-dataset", "test-dataset2"],
#         }
#     )
#     assert tags["creator"] == "pytest"
#     assert tags["datasets"] == "test-dataset,test-dataset2"


# def test_register_results_with_mlflow(
#     evaluation_config,
#     helm_run_output_path,
#     evaluation_config_on_disk,
#     helm_config_on_disk,
#     core_scenario_metrics_on_disk,
# ):
#     cli_utils.register_results_with_mlflow(
#         evaluation_config,
#         helm_run_output_path,
#         evaluation_config_on_disk,
#         helm_config_on_disk,
#         core_scenario_metrics_on_disk,
#     )
#     run = mlflow.get_run(mlflow.last_active_run().info.run_id)
#     assert run.data.tags["cpu"] == "1"
#     assert run.data.tags["memory"] == "1Gi"
#     assert run.data.tags["accelerator"] == "nvidia-tesla-t4"
#     assert run.data.tags["creator"] == "pytest"
#     assert run.data.tags["image_build_cajajejo_version"] == __version__
#     assert run.data.tags["config_git_sha"] == "hfir4832478gyegfg"
#     assert run.data.tags["config_path_git_repo"] == "/path/to/config.yml"
#     assert run.data.tags["helm_config_path_git_repo"] == "/path/to/helm/config.conf"
#     assert run.data.tags["image"] == "test/image"
#     assert run.data.tags["image_tag"] == "latest"
#     assert run.data.tags["lifecycle"] == "testing"
#     assert run.data.tags["model_run_id"] == "test-run-id"
#     assert run.data.tags["model_artifact_path"] == "model"
#     assert run.data.tags["model_gs_uri"] == "None"
#     assert run.data.tags["job_name"] == "test-job"
#     for k, v in evaluation_config.helm.dict().items():
#         assert run.data.params.get(f"helm_{k}") == str(v)

#     assert len(os.listdir(run.info.artifact_uri)) == 3


# def test_sanity_check_mlflow_model_directory(mlflow_model_directory):
#     cli_utils._check_mlflow_model_path_exists(
#         "artifacts",
#         "model",
#         base_path=str(mlflow_model_directory.parent.parent),
#     )


# def test_sanity_check_mlflow_tokenizer_directory(mlflow_tokenizer_directory):
#     cli_utils._check_mlflow_tokenizer_path_exists(
#         "artifacts",
#         "components/tokenizer",
#         base_path=str(mlflow_tokenizer_directory.parent.parent.parent),
#     )


# # def test_helm_sanity_checks_succeed(evaluation_config, helm_config_on_disk):
# #     with mock.patch.object(cli_utils, "sh", mock.Mock()) as mock_sh:
# #         cli_utils._check_helm_run_debug(helm_config_on_disk, evaluation_config)
# #         mock_sh.helm_run.assert_called_once_with(
# #             [
# #                 "--conf-paths",
# #                 helm_config_on_disk,
# #                 "--suite",
# #                 "test-suite",
# #                 "--output-path",
# #                 "/scratch/debug",
# #                 "--max-eval-instances",
# #                 "10",
# #             ],
# #         )


# # def test_helm_sanity_checks_fail(evaluation_config, helm_config_on_disk):
# #     class ErrorReturnCode(Exception):
# #         def __init__(self, message):
# #             super().__init__(message)
# #             self.stderr = b"This is an error message"
# #             self.exit_code = 1
# #
# #     with mock.patch.object(cli_utils, "sh", mock.Mock()) as mock_sh:
# #         mock_sh.ErrorReturnCode = ErrorReturnCode
# #         mock_sh.helm_run.side_effect = mock_sh.ErrorReturnCode("Exception")
# #         with pytest.raises(ErrorReturnCode):
# #             cli_utils._check_helm_run_debug(helm_config_on_disk, evaluation_config)
# #         mock_sh.helm_run.assert_called_once_with(
# #             [
# #                 "--conf-paths",
# #                 helm_config_on_disk,
# #                 "--suite",
# #                 "test-suite",
# #                 "--output-path",
# #                 "/scratch/debug",
# #                 "--max-eval-instances",
# #                 "10",
# #             ],
# #         )


# def test_logging_without_mlflow(logger, caplog):
#     logger.debug("This is a debug statement")
#     assert caplog.records[0].labels == {
#         "job_name": "test-job",
#         "image_build_cajajejo_version": __version__,
#         "config_git_sha": "hfir4832478gyegfg",
#     }


# def test_logging_with_mlflow(logger, caplog):
#     mlflow.set_experiment("test_logging")
#     with mlflow.start_run() as _:
#         logger.debug("This is a debug statement with mlflow capture")
#     custom_dims_keys = list(caplog.records[0].labels.keys())
#     custom_dims_keys.sort()
#     assert custom_dims_keys == [
#         "config_git_sha",
#         "image_build_cajajejo_version",
#         "job_name",
#         "mlflow_artifact_uri",
#         "mlflow_experiment_id",
#         "mlflow_experiment_name",
#         "mlflow_run_id",
#         "mlflow_run_name",
#         "mlflow_user",
#     ]
