resource "google_service_account_key" "artifact-writer" {
  service_account_id = google_service_account.artifact-writer.id
}

resource "google_service_account_key" "data-storage-admin" {
  service_account_id = google_service_account.data-storage-admin.id
}

resource "google_secret_manager_secret" "credentials" {
  for_each = local.credentials

  secret_id = each.key

  labels = {
    terraform = true
  }

  replication {
    automatic = true
  }
}

resource "google_secret_manager_secret_version" "credentials" {
  for_each = local.credentials

  secret      = local.credential_ids[each.key]
  secret_data = each.value
}
