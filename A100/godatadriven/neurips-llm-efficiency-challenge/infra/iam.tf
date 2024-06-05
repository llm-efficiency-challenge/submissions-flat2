// Give owner rights to users / admins
resource "google_project_iam_binding" "this" {
  project = var.gcp_project_id
  role    = "roles/owner"
  members = local.admins
}

// Reader on data, admin on evaluation bucket
resource "google_storage_bucket_iam_member" "gke-data-bucket-reader" {
  bucket = module.storage.data-bucket-name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${module.kubernetes.service_account_email}"
}

resource "google_storage_bucket_iam_member" "gke-mlflow-bucket-reader" {
  bucket = module.mlflow.mlflow-artifact-bucket-name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${module.kubernetes.service_account_email}"
}

resource "google_storage_bucket_iam_member" "gke-mlflow-bucket-object-creator" {
  bucket = module.mlflow.mlflow-artifact-bucket-name
  role   = "roles/storage.objectCreator"
  member = "serviceAccount:${module.kubernetes.service_account_email}"
}

resource "google_storage_bucket_iam_member" "gke-evaluation-bucket-admin" {
  bucket = module.storage.evaluation-bucket-name
  role   = "roles/storage.admin"
  member = "serviceAccount:${module.kubernetes.service_account_email}"
}

// Compute instances
resource "google_storage_bucket_iam_member" "compute-mlflow-bucket-user" {
  for_each = toset(keys(var.users))

  bucket = module.mlflow.mlflow-artifact-bucket-name
  role   = "roles/storage.objectUser"
  member = "serviceAccount:${module.compute-gpu[each.key].service-account-email}"
}

resource "google_storage_bucket_iam_member" "compute-mlflow-bucket-object-creator" {
  for_each = toset(keys(var.users))

  bucket = module.mlflow.mlflow-artifact-bucket-name
  role   = "roles/storage.objectCreator"
  member = "serviceAccount:${module.compute-gpu[each.key].service-account-email}"
}

resource "google_storage_bucket_iam_member" "compute-data-bucket-user" {
  for_each = toset(keys(var.users))

  bucket = module.storage.data-bucket-name
  role   = "roles/storage.objectUser"
  member = "serviceAccount:${module.compute-gpu[each.key].service-account-email}"
}

resource "google_storage_bucket_iam_member" "compute-data-bucket-object-creator" {
  for_each = toset(keys(var.users))

  bucket = module.storage.data-bucket-name
  role   = "roles/storage.objectCreator"
  member = "serviceAccount:${module.compute-gpu[each.key].service-account-email}"
}

// Add a service account to read and write to the registry from e.g. poetry or pip or docker
resource "google_service_account" "artifact-writer" {
  account_id = "artifact-reader"
}

resource "google_project_iam_member" "artifact-reader" {
  role    = "roles/artifactregistry.reader"
  member  = "serviceAccount:${google_service_account.artifact-writer.email}"
  project = var.gcp_project_id
}

resource "google_project_iam_member" "artifact-writer" {
  role    = "roles/artifactregistry.writer"
  member  = "serviceAccount:${google_service_account.artifact-writer.email}"
  project = var.gcp_project_id
}

// Create a service account with access to bucket for DVC
resource "google_service_account" "data-storage-admin" {
  account_id   = "dvc-storage-admin"
  display_name = "DVC Storage Admin"
  description  = "Service account with access to the data bucket for DVC."
}

resource "google_storage_bucket_iam_member" "data-storage-admin" {
  bucket = module.storage.data-bucket-name
  role   = "roles/storage.admin"
  member = "serviceAccount:${google_service_account.data-storage-admin.email}"
}
