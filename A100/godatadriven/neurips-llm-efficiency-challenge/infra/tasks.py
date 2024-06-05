import base64
import json
import os
import pathlib as plb
import warnings
import typing

from google.api_core.exceptions import NotFound, PermissionDenied
from google.cloud import secretmanager
from invoke import task

warnings.filterwarnings("ignore")

root_dir = plb.Path(__file__).parent.parent.absolute()

client = secretmanager.SecretManagerServiceClient()


def _retrieve_most_recent_secret_version(secret_name) -> str:
    """Retrieves the most recent secret version"""
    secret_version = client.access_secret_version(
        request={"name": f"{secret_name}/versions/latest"}
    )
    return secret_version.payload.data.decode("UTF-8")


@task(help={"echo": "Print the GCP project number to stdout"})
def get_gcp_project_number(c, echo=False) -> int:
    """⅐ Get the GCP project number or retrieve it from env variables"""
    if os.environ.get("GCP_PROJECT_NUMBER") is not None:
        gcp_project_id = os.environ.get("GCP_PROJECT_NUMBER")
    else:
        gcp_project_id = c.run(
            """gcloud projects list --filter="$(gcloud config get-value project)" --format='value(PROJECT_NUMBER)'""",
            hide=True,
        ).stdout.replace("\n", "")
    if echo:
        print(gcp_project_id)
    return gcp_project_id


@task
def tf_plan(c):
    """📜 Run terraform plan"""
    c.run("terraform plan -var-file=vars/prod.tfvars")


@task
def tf_apply(c):
    """🚀 Run terraform apply"""
    c.run("terraform apply -var-file=vars/prod.tfvars")


@task
def tf_docs(c):
    """📕 Generate terraform docs"""
    c.run("terraform-docs .")


@task
def write_json_key_dvc_storage_admin(c):
    """📝 Print json key for dvc storage admin GCP service account"""
    gcp_project_number = get_gcp_project_number(c)
    secret_value_b64 = _retrieve_most_recent_secret_version(
        f"projects/{gcp_project_number}/secrets/GCP_DATA_BUCKET_STORAGE_ADMIN_SA_JSON_KEY_B64"
    )
    out = c.run(
        f"echo {secret_value_b64} | python -m base64 -d",
        hide=True,
    )
    with open("../.secrets/data-bucket-service-account-key.json", "w") as f:
        f.write(out.stdout)


def _get_private_key_output(c, name, machine):
    """Get the private key output from GCP secret manager and decode it"""
    gcp_project_number = get_gcp_project_number(c)
    secret_name = f"projects/{gcp_project_number}/secrets/COMPUTE-NEURIPS-2023-{machine.upper()}-{name.upper()}-PRIVATE-KEY"
    try:
        secret_value_b64 = _retrieve_most_recent_secret_version(secret_name)
    except (PermissionDenied, NotFound):
        print(
            f"Cannot access secret {secret_name}. Please make sure you have the correct permissions and that the secret exists."
        )
        secret_value_b64 = base64.b64encode("Secret not found.".encode("utf-8")).decode(
            "utf-8"
        )
    tf_out = json.loads(
        c.run(f"terraform output -json compute-{machine}", hide=True).stdout
    )[name]
    tf_out["private_key"] = base64.b64decode(secret_value_b64.encode("utf-8")).decode(
        "utf-8"
    )
    return tf_out


def _write_private_key(c, name, key):
    ssh_config_path = root_dir / ".secrets" / ".ssh" / f"id_rsa_{name}"
    if ssh_config_path.exists():
        os.remove(ssh_config_path)
    with ssh_config_path.open("w") as outFile:
        outFile.write(key)
    c.run(f"chmod 400 {ssh_config_path}")
    return ssh_config_path


def _create_ssh_config(name, machine, public_ip, key_path):
    return f"""Host neurips-compute-{machine}\n  HostName {public_ip}\n  User {name}\n  IdentityFile {key_path}\n  strictHostKeyChecking=no\n  NoHostAuthenticationForLocalhost yes\n"""


@task(
    help={
        "name": "Name of the user for which you want to create a key",
    }
)
def write_gcp_compute_ssh_key(
    c,
    name: str,
    machine: str = "gpu",
    ssh_user: typing.Optional[str] = None,
    # cpu: bool = False,
    # gpu: bool = True,
):
    """📝 Write ssh key for gcp compute"""
    allowed_machine_names = ["cpu", "gpu", "a100"]
    if machine not in allowed_machine_names:
        raise ValueError("At least one of cpu and gpu must be True")

    machine_type = "cpu" if machine == "cpu" else "gpu"
    tf_output = _get_private_key_output(c, name, machine_type)
    if ".ssh" not in os.listdir(f"{root_dir}/.secrets"):
        os.mkdir(f"{root_dir}/.secrets/.ssh")  # TODO(jetze): use pathlib

    ssh_compute_config_content = []
    key_path = _write_private_key(c, name, tf_output["private_key"])
    ssh_compute_config_content.append(
        _create_ssh_config(
            name if ssh_user is None else ssh_user,
            machine,
            tf_output["public_ip"],
            key_path,
        )
    )
    with (root_dir / "ssh_compute_config").open("w") as outFile:
        outFile.write("\n".join(ssh_compute_config_content))
