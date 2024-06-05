import pytest
import yaml


@pytest.mark.usefixtures("job_spec")
class TestJinjaParser:
    def test_number_jobs_equals_three(self):
        assert len(self.job_spec) == 3

    def test_eval_conf_config_map(self, evaluation_config_var):
        assert (
            yaml.safe_load(self.job_spec[1]["data"]["eval_config.yml"])
            == evaluation_config_var.dict()
        )

    def test_helm_conf_config_map(self, helm_config):
        assert self.job_spec[1]["data"]["run_specs.conf"] == helm_config

    def test_gs_cmd_parsed(self):
        init_container = self.job_spec[0]["spec"]["template"]["spec"]["initContainers"][
            0
        ]
        if init_container["image"] != "google/cloud-sdk:slim":
            pytest.skip("Skipping because test only relevant for GCS init container")
        assert (
            " ".join(init_container["command"])
            == "sh -c gcloud alpha storage cp -r gs://test-bucket/test-artifact /scratch"
        )

    def test_mlflow_tracking_server_parsed(self, mlflow_tracking_server):
        init_container = self.job_spec[0]["spec"]["template"]["spec"]["initContainers"][
            0
        ]
        if init_container["image"] == "google/cloud-sdk:slim":
            pytest.skip("Skipping because test only relevant for MLFlow init container")
        assert init_container["env"][0]["value"] == str(mlflow_tracking_server)

    def test_mlflow_cmd_parsed(self):
        init_container = self.job_spec[0]["spec"]["template"]["spec"]["initContainers"][
            0
        ]
        if init_container["image"] == "google/cloud-sdk:slim":
            pytest.skip("Skipping because test only relevant for MLFlow init container")
        assert init_container["command"] == ["mlflow"]

    def test_mlflow_args_parsed(self):
        init_container = self.job_spec[0]["spec"]["template"]["spec"]["initContainers"][
            0
        ]
        if init_container["image"] == "google/cloud-sdk:slim":
            pytest.skip("Skipping because test only relevant for MLFlow init container")
        assert (
            " ".join(init_container["args"])
            == "artifacts download -r 5478238igfgyu284t3 -a artifacts -d /scratch"
        )

    def test_job_name(self):
        assert self.job_spec[0]["metadata"]["name"] == "eval-neurips-test"

    def test_node_selector_parsed(self):
        node_selector = self.job_spec[0]["spec"]["template"]["spec"].get("nodeSelector")
        if node_selector is None:
            pytest.skip("Skipping because test only relevant when using GPU")
        assert node_selector == {"cloud.google.com/gke-accelerator": "nvidia-tesla-t4"}

    def test_image_parsed(self):
        assert (
            self.job_spec[0]["spec"]["template"]["spec"]["containers"][0]["image"]
            == "test/image:latest"
        )

    def test_cpu_value_parsed(self):
        assert (
            self.job_spec[0]["spec"]["template"]["spec"]["containers"][0]["resources"][
                "requests"
            ]["cpu"]
            == "1"
        )

    def test_memory_value_parsed(self):
        assert (
            self.job_spec[0]["spec"]["template"]["spec"]["containers"][0]["resources"][
                "requests"
            ]["memory"]
            == "1Gi"
        )

    def test_env_eval_config_path(self, evaluation_config_var_on_disk):
        assert self.job_spec[0]["spec"]["template"]["spec"]["containers"][0]["env"][
            0
        ] == {
            "name": "CONFIG_PATH_GIT_REPO",
            "value": evaluation_config_var_on_disk,
        }

    def test_env_helm_config_path(self, helm_config_on_disk):
        assert self.job_spec[0]["spec"]["template"]["spec"]["containers"][0]["env"][
            1
        ] == {
            "name": "HELM_CONFIG_PATH_GIT_REPO",
            "value": helm_config_on_disk,
        }

    def test_env_job_name(self):
        assert self.job_spec[0]["spec"]["template"]["spec"]["containers"][0]["env"][
            2
        ] == {
            "name": "JOB_NAME",
            "value": "eval-neurips-test",
        }

    def test_env_config_version(self):
        assert self.job_spec[0]["spec"]["template"]["spec"]["containers"][0]["env"][
            3
        ] == {
            "name": "CONFIG_VERSION",
            "value": "778ff65567df57s6fd",
        }
