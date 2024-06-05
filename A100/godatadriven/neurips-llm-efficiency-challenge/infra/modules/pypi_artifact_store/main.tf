data "google_project" "project" {}

locals {
  project_id = data.google_project.project.id
}

resource "google_artifact_registry_repository" "python-registry" {
  provider      = google-beta
  location      = var.region
  repository_id = "pypi-registry-${var.prefix}"
  format        = "PYTHON"
}
