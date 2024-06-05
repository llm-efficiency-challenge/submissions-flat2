// TODO: proper account setup for lakefs, mlflow and prefect. They shouldn't be able to access each other's database.
locals {
  credentials = {
    "POSTGRESQL-PASSWORD" = random_password.password.result,
    "POSTGRESQL-USERNAME" = var.username
  }
  credential_ids = {
    for key, value in local.credentials : key => google_secret_manager_secret.credentials[key].id
  }
}

resource "random_password" "password" {
  length           = 16
  special          = false
  override_special = "!#$%&*()-_=+[]{}<>:?"
}

resource "random_pet" "database" {}

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

resource "google_sql_database_instance" "main" {
  name                = "ds-platform-${var.prefix}-${random_pet.database.id}"
  database_version    = "POSTGRES_13"
  region              = var.region
  deletion_protection = false

  settings {
    # Second-generation instance tiers are based on the machine
    # type. See argument reference below.
    tier              = "db-f1-micro"
    availability_type = "ZONAL"
    disk_autoresize   = true
    disk_type         = "PD_SSD"
    user_labels = {
      terraform = true
    }
    ip_configuration {

      dynamic "authorized_networks" {
        for_each = toset(var.whitelist_ips)
        iterator = address

        content {
          value = address.value
        }
      }
    }
  }
}

resource "google_sql_user" "user" {
  name     = var.username
  instance = google_sql_database_instance.main.name
  password = random_password.password.result
}
