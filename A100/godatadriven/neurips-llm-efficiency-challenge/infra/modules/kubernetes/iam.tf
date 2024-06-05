resource "google_service_account" "kubernetes" {
  account_id  = "kubernetes"
  description = "Terraform-managed service account for cluster simple-autopilot-private-cluster"
}

resource "google_project_iam_member" "artifact-registry-reader" {
  role    = "roles/artifactregistry.reader"
  member  = "serviceAccount:${google_service_account.kubernetes.email}"
  project = local.project_id
}

resource "google_project_iam_member" "log-writer" {
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.kubernetes.email}"
  project = local.project_id
}

resource "google_project_iam_member" "monitoring-metric-writer" {
  role    = "roles/monitoring.metricWriter"
  member  = "serviceAccount:${google_service_account.kubernetes.email}"
  project = local.project_id
}

resource "google_project_iam_member" "monitoring-viewer" {
  role    = "roles/monitoring.viewer"
  member  = "serviceAccount:${google_service_account.kubernetes.email}"
  project = local.project_id
}

resource "google_project_iam_member" "stackdriver-resource-metadata-writer" {
  role    = "roles/stackdriver.resourceMetadata.writer"
  member  = "serviceAccount:${google_service_account.kubernetes.email}"
  project = local.project_id
}

resource "google_project_iam_member" "secret-accessor" {
  role    = "roles/secretmanager.secretAccessor"
  member  = "serviceAccount:${google_service_account.kubernetes.email}"
  project = local.project_id
}
