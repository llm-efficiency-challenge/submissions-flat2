import base64
import json
import logging
import os
import pathlib as plb
import typing
import warnings
import tempfile
import hashlib
from contextlib import contextmanager

import git
import pendulum
from google.cloud import secretmanager
from invoke import task

logger = logging.getLogger("cli")
handler = logging.StreamHandler()
format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
handler.setFormatter(format)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

warnings.filterwarnings("ignore")

GCP_PROJECT_NUMBER = "750004521723"
REGISTRY_URL = "europe-west4-docker.pkg.dev/neurips-llm-eff-challenge-2023/container-registry-neurips-2023/"


@contextmanager
def _gcp_secret_manager():
    """Context manager for GCP Secret Manager"""
    client = secretmanager.SecretManagerServiceClient()
    yield client


@contextmanager
def _remove_build_files_on_exit():
    """Context manager for removing dist folder"""
    path = plb.Path("dist").resolve()
    path.mkdir(exist_ok=True)
    before = os.listdir("dist")
    if len(before) > 0:
        logger.info("Found existing build files. These will be removed.")
        inp = input("Continue? y/n... : ")
        if inp.lower() != "y":
            raise InterruptedError("User aborted.")
        else:
            for f in before:
                (path / f).unlink()
    try:
        yield
    finally:
        # Should be empty, but leaving this here for future ref
        after = os.listdir("dist")
        new = set(after) - set(before)
        for f in new:
            (path / f).unlink()


def _export_requirements(c, job: str, out_file: str) -> None:
    if job == "training":
        c.run(f"poetry export --without-hashes -E training -o {out_file}")
    elif job == "evaluation":
        c.run(f"poetry export --without-hashes -E training -E submission -o {out_file}")
    else:
        c.run(f"poetry export --without-hashes -o {out_file}")


def _hash_requirements_file(path: typing.Union[plb.Path, str]) -> str:
    with plb.Path(path).open("r") as f:
        requirements = f.read()
    return hashlib.sha256(requirements.encode("utf-8")).hexdigest()


def _export_requirements_if_changed(c, job: str) -> None:
    """Export requirements to .requirements_cache"""
    plb_requirements = plb.Path(".requirements_cache").resolve()
    plb_requirements.mkdir(exist_ok=True)
    requirements_file = (
        (plb_requirements / "requirements.txt")
        if job is None
        else (plb_requirements / f"requirements_{job}.txt")
    )
    if requirements_file.exists():
        logger.info(f"Found existing requirements file at {requirements_file}")
        hash_current = _hash_requirements_file(requirements_file)
        # Export new requirements
        with tempfile.TemporaryDirectory() as td:
            _export_requirements(c, job, os.path.join(td, "requirements.txt"))
            hash_new = _hash_requirements_file(os.path.join(td, "requirements.txt"))
        if hash_current != hash_new:
            logger.info("Requirements have changed. Exporting new requirements.")
            logger.debug(f"Old hash: {hash_current}")
            logger.debug(f"New hash: {hash_new}")
            _export_requirements(c, job, requirements_file)
        else:
            logger.info("Requirements have not changed. Skipping export.")
    else:
        _export_requirements(c, job, requirements_file)


def retrieve_most_recent_secret_version(secret_name) -> str:
    """Retrieves the most recent secret version"""
    with _gcp_secret_manager() as client:
        secret_version = client.access_secret_version(
            request={"name": f"{secret_name}/versions/latest"}
        )
    return secret_version.payload.data.decode("UTF-8")


def _b64_decode(b64_str: str) -> str:
    """Decode a base64 encoded string"""
    return base64.b64decode(b64_str.encode("utf-8")).decode("utf-8")


def _get_commit_sha(allow_dirty_repo: bool) -> str:
    """Retrieve the commit sha of the current git repository"""
    repo = git.Repo(search_parent_directories=True)
    if not allow_dirty_repo:
        if repo.is_dirty():
            raise git.RepositoryDirtyError(
                repo=repo, message="Repository is dirty. Please commit changes."
            )
    return repo.head.object.hexsha


def _create_docker_tags(
    registry_url: str, allow_dirty_repo: bool
) -> typing.Tuple[str, str]:
    """Create docker tags for the current commit"""
    tag = f"{registry_url}:latest"
    tag_sha = f"{registry_url}:{_get_commit_sha(allow_dirty_repo)[:10]}"
    return tag, tag_sha


@task
def update_pip(c):
    """☝ This will update pip in the virtual environment"""
    c.run("poetry run python -m pip install -U pip")


@task(
    help={
        "dev": "Install development dependency group",
        "docs": "Install docs dependency group",
        "evaluation": "Install evaluation extras",
        "training": "Install training extras",
    }
)
def install_dependencies(c, dev=False, docs=False, evaluation=False, training=False):
    """⬇ Install project dependencies and optional dependencies"""
    update_pip(c)
    cmd = "poetry install"
    if dev:
        cmd += " --with dev"
    if docs:
        cmd += " --with docs"
    if evaluation:
        cmd += " -E evaluation"
    if training:
        cmd += " -E training"
    c.run(cmd)
    if training:
        # Unfortunately, this is necessary workaround
        c.run("poetry run pip install xformers==0.0.21 --no-dependencies")


@task
def serve_docs(c):
    """📓 Serve the documentation locally"""
    c.run("mkdocs serve")


@task
def docker_login(c, gcp_project_number: typing.Optional[str] = None):
    """🐳 Log in to the GCP docker registry"""
    if gcp_project_number is not None:
        logger.info("Overriding default GCP project number")
    secret_value_b64 = retrieve_most_recent_secret_version(
        f"projects/{GCP_PROJECT_NUMBER if gcp_project_number is None else gcp_project_number}/secrets/GCP_ARTIFACT_WRITER_SA_JSON_KEY_B64"
    )
    secret_value = _b64_decode(secret_value_b64)
    c.run(f"docker login -u _json_key -p '{secret_value}' europe-west4-docker.pkg.dev")
    logger.info("🙏 Successfully logged into docker registry!")


@task()
def build_mlflow_image(c):
    """🐳 Build the mlflow image"""
    registry_uri = os.path.join(REGISTRY_URL, "mlflow-skinny-gcp")
    tag_latest, tag_sha = _create_docker_tags(registry_uri, allow_dirty_repo=True)
    c.run(
        f"DOCKER_BUILDKIT=1 docker build -t {tag_latest} -t {tag_sha} -t neurips/mlflow-skinny-gcp -f mlflow.Dockerfile ."
    )


@task()
def push_mlflow_image(c):
    """🌬 Push mlflow image to GCP docker registry"""
    registry_uri = os.path.join(REGISTRY_URL, "mlflow-skinny-gcp")
    tag_latest, tag_sha = _create_docker_tags(registry_uri, allow_dirty_repo=True)
    for tag in [tag_latest, tag_sha]:
        c.run(f"docker push {tag}")


@task()
def build_and_push_mlflow_image(c):
    """🐳+🌬 Build and push mlflow image to GCP docker registry"""
    build_mlflow_image(c)
    push_mlflow_image(c)


@task()
def build_helm_image(c):
    """🐳 Build helm image"""
    registry_uri = os.path.join(REGISTRY_URL, "helm")
    tag_latest, tag_sha = _create_docker_tags(registry_uri, allow_dirty_repo=True)
    with _remove_build_files_on_exit():
        c.run("poetry build")
        # Installing the dependencies before installing the library
        #  separates the dependencies from the library in the docker image
        #  which allows for faster builds and uploading to remote registry
        _export_requirements_if_changed(c, "helm")
        # See: https://docs.docker.com/build/buildkit/
        c.run(
            f"DOCKER_BUILDKIT=1 docker build -t {tag_latest} -t {tag_sha} -t neurips/helm -f helm.Dockerfile ."
        )


@task()
def push_helm_image(c):
    """🌬 Push helm image to GCP docker registry"""
    registry_uri = os.path.join(REGISTRY_URL, "helm")
    tag_latest, tag_sha = _create_docker_tags(registry_uri, allow_dirty_repo=True)
    for tag in [tag_latest, tag_sha]:
        c.run(f"docker push {tag}")


@task()
def build_and_push_helm_image(c):
    """🐳+🌬 Build and push mlflow image to GCP docker registry"""
    build_helm_image(c)
    push_helm_image(c)


@task(
    help={
        "job": "Job for which to build image. Must be one of 'training' or 'evaluation'.",
        "allow_dirty_repo": "Allow building image from dirty repository. Not recommended as this will yield a non-reproducible workflow.",
    }
)
def build_job_image(c, job: str, allow_dirty_repo: typing.Optional[bool] = False):
    """🐳 Build docker image needed for evaluation or training job"""
    if job not in ["evaluation", "training"]:
        raise ValueError(
            f"Job {job} not supported. Must be 'evaluation', or 'training'"
        )
    registry_uri = os.path.join(REGISTRY_URL, job)
    try:
        tag_latest, tag_sha = _create_docker_tags(registry_uri, allow_dirty_repo)
    except git.RepositoryDirtyError:
        msg = (
            "Repository is dirty. Please commit changes or run with --allow-dirty-repo."
        )
        logger.error(msg)
        raise ValueError(msg)
    # Build package
    with _remove_build_files_on_exit():
        c.run("poetry build")
        # Installing the dependencies before installing the library
        #  separates the dependencies from the library in the docker image
        #  which allows for faster builds and uploading to remote registry
        _export_requirements_if_changed(c, job)
        # See: https://docs.docker.com/build/buildkit/
        c.run(
            f"DOCKER_BUILDKIT=1 docker build -t {tag_latest} -t {tag_sha} -t neurips/{job}:latest -f {job}.Dockerfile ."
        )


@task(
    help={
        "job": "Job for which to push image. Must be one of 'training' or 'evaluation'.",
    }
)
def push_job_image(c, job: str, allow_dirty_repo: typing.Optional[bool] = False):
    """🌬 Push job image to GCP docker registry"""
    if job not in ["evaluation", "training"]:
        raise ValueError(f"Job {job} not supported. Must be 'evaluation' or 'training'")
    registry_uri = os.path.join(REGISTRY_URL, job)
    tag_latest, tag_sha = _create_docker_tags(
        registry_uri, allow_dirty_repo=allow_dirty_repo
    )
    for tag in [tag_latest, tag_sha]:
        c.run(f"docker push {tag}")


@task(
    help={
        "job": "Job for which to build and push image. Must be one of 'training' or 'evaluation'.",
    }
)
def build_and_push_job_image(
    c, job: str, allow_dirty_repo: typing.Optional[bool] = False
):
    """🐳+🌬 Build and push evaluation image to GCP docker registry"""
    build_job_image(c, job, allow_dirty_repo)
    push_job_image(c, job, allow_dirty_repo)


@task()
def authenticate_kubectl(c):
    """🔑 Authenticate GKE cluster in kubectl"""
    # Update brew: brew upgrade --cask google-cloud-sdk
    # Install: gcloud components install gke-gcloud-auth-plugin
    os.environ["USE_GKE_GCLOUD_AUTH_PLUGIN"] = "true"
    c.run(
        "gcloud container clusters get-credentials cluster-neurips-2023-autopilot --region europe-west4 --project neurips-llm-eff-challenge-2023"
    )


@task(
    help={
        "path_to_config": "Path to evaluation config file",
        "path_to_helm_config": "Path to helm config file",
        "kubernetes_job_spec_output_path": "Path to output kubernetes job spec (optional).",
        "allow_dirty_repo": "Allow building image from dirty repository. Not recommended as this will yield a non-reproducible workflow.",
    }
)
def submit_evaluation_job(
    c,
    path_to_config: str,
    path_to_helm_config: str,
    kubernetes_job_spec_output_path: typing.Optional[str] = None,
    allow_dirty_repo: typing.Optional[bool] = False,
):
    """🚀 Submit evaluation job to GKE cluster"""
    try:
        commit_sha = _get_commit_sha(allow_dirty_repo)
    except git.RepositoryDirtyError:
        msg = (
            "Repository is dirty. Please commit changes or run with --allow-dirty-repo."
        )
        logger.error(msg)
        raise ValueError(msg)
    cmd = f"""cajajejo jobs evaluation submit-job --path-to-config {path_to_config} --path-to-helm-config {path_to_helm_config} --config-version {commit_sha}"""
    if kubernetes_job_spec_output_path is not None:
        cmd += f" --job-spec-output-path {kubernetes_job_spec_output_path}"
    c.run(cmd)


@task(
    help={
        "job_name": "Name of the *parsed* job (e.g. 'eval-neurips-amusing-sloth').",
        "shift_unit": "Unit of time to shift current date/time for searching logs. Must be one of 'days', 'hours', 'minutes'. Defaults to 'days'.",
        "shift_amount": "Amount of time to shift current date/time for searching logs. Defaults to 1.",
        "output_path": "Write logs to file at this path (optional).",
    },
    positional=["job_name"],
    optional=["shift_unit", "shift_amount", "output_path"],
)
def get_job_logs(
    c,
    job_name: str,
    shift_unit: typing.Optional[str] = "days",
    shift_amount: typing.Optional[int] = 1,
    output_path: typing.Optional[str] = None,
):
    """📜 Get logs for a training or evaluation job"""
    if shift_unit not in ["days", "hours", "minutes"]:
        raise ValueError("shift_unit must be one of 'days', 'hours', 'minutes'")
    if output_path is not None:
        _output_path_parent = plb.Path(output_path).resolve().parent
        if not _output_path_parent.exists():
            raise ValueError(f"Parent directory {_output_path_parent} does not exist.")
    now = pendulum.now("Europe/Amsterdam")
    if shift_unit == "days":
        now = now.add(days=-int(shift_amount))
    elif shift_unit == "hours":
        now = now.add(hours=-int(shift_amount))
    else:
        now = now.add(minutes=-int(shift_amount))
    now_str = now.to_iso8601_string()
    cmd = f"""gcloud logging read --order=desc --format=json "labels.job_name=\\"{job_name}\\" AND timestamp>=\\"{now_str}\\"" """.rstrip()
    logger.debug(f"Running command: {cmd}")
    cmd_out = c.run(cmd, hide=True).stdout
    logs = json.loads(cmd_out)
    for log in reversed(logs):
        print(log["textPayload"])
    if output_path is not None and len(logs) > 0:
        logger.info(f"Writing logs to {output_path}")
        _path = plb.Path(output_path)
        with _path.open("w") as f:
            f.write(cmd_out)
    else:
        logger.info("No logs found. Perhaps you need to increase the search window?")
