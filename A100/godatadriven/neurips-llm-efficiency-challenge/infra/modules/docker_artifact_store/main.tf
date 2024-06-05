// See https://cloud.google.com/artifact-registry/docs/docker/authentication for info on authenticating with the registry
resource "google_artifact_registry_repository" "docker-registry" {
  provider      = google-beta
  location      = var.region
  repository_id = "container-registry-${var.prefix}"
  format        = "DOCKER"
}
