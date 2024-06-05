data "google_project" "project" {}

locals {
  project_id   = data.google_project.project.id
  network_tags = concat(var.extra_network_tags, [])
}

resource "google_sql_database" "mlflow" {
  name     = "mlflow"
  instance = var.sql_instance_name
}

resource "google_storage_bucket" "mlflow" {
  name          = "mlflow-${var.prefix}"
  location      = var.region
  force_destroy = false
  storage_class = "STANDARD"
}

resource "google_service_account" "mlflow" {
  account_id = "mlflow"
}

resource "google_storage_bucket_iam_member" "member" {
  bucket = google_storage_bucket.mlflow.name
  role   = "roles/storage.admin"
  member = "serviceAccount:${google_service_account.mlflow.email}"
}

resource "google_project_iam_member" "sql_admin" {
  role    = "roles/cloudsql.admin"
  member  = "serviceAccount:${google_service_account.mlflow.email}"
  project = local.project_id
}

resource "google_compute_instance" "compute" {
  name         = "mlflow-${var.prefix}"
  machine_type = var.machine_type
  tags         = local.network_tags

  zone = var.zone

  metadata_startup_script = templatefile("${abspath(path.module)}/static/startup.sh", {
    artifact_bucket_name = "gs://${google_storage_bucket.mlflow.name}"
    backend_database_uri = var.sql_connection_string
  })

  network_interface {
    network = "default"

    access_config {
      nat_ip = var.static_ip
    }
  }

  boot_disk {
    initialize_params {
      image = "debian-cloud/debian-11"
    }
  }

  service_account {
    email  = google_service_account.mlflow.email
    scopes = ["cloud-platform"]
  }

  allow_stopping_for_update = true

}
