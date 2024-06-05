resource "google_storage_bucket" "data" {
  name          = "data-${var.prefix}"
  location      = var.region
  force_destroy = false
  storage_class = "STANDARD"
  versioning {
    enabled = true
  }
  logging {
    log_bucket = google_storage_bucket.data-logs.name
  }
}

resource "google_storage_bucket" "evaluation-results" {
  name          = "evaluation-results-${var.prefix}"
  location      = var.region
  force_destroy = false
  storage_class = "STANDARD"
}

resource "google_storage_bucket" "data-logs" {
  name          = "data-logs-${var.prefix}"
  location      = var.region
  force_destroy = false
  storage_class = "STANDARD"
}
