locals {
  admins = formatlist("user:%s", var.admins)

  compute_gpu_connection_details = var.deploy_gpu_compute ? {
    for username in keys(var.users) :
    username => module.compute-gpu[username].connection-details
    } : {
    for username in keys(var.users) :
    username => { "public_ip" : "", "private_key_secret_key" : "" }
  }
  compute_cpu_connection_details = var.deploy_cpu_compute ? {
    for username in keys(var.users) :
    username => module.compute-cpu[username].connection-details
    } : {
    for username in keys(var.users) :
    username => { "public_ip" : "", "private_key_secret_key" : "" }
  }
  compute_firewall_tag = {
    for username in keys(var.users) :
    username => ["allow-ssh-${var.prefix}-${username}"]
  }

  credentials = {
    "GCP_ARTIFACT_WRITER_SA_JSON_KEY_B64"           = google_service_account_key.artifact-writer.private_key,
    "GCP_DATA_BUCKET_STORAGE_ADMIN_SA_JSON_KEY_B64" = google_service_account_key.data-storage-admin.private_key
  }
  credential_ids = {
    for key, value in local.credentials : key => google_secret_manager_secret.credentials[key].id
  }

  mlflow_whitelist_ips = concat(var.mlflow_whitelist_ips, [module.kubernetes.pods_ip_range])
}

data "google_client_config" "default" {}

data "google_secret_manager_secret_version" "basic" {
  secret     = "POSTGRESQL-PASSWORD"
  depends_on = [module.sql, module.services]
}

module "services" {
  source = "./modules/services"
}

module "storage" {
  source     = "./modules/cloud_storage"
  prefix     = var.prefix
  region     = var.gcp_region
  depends_on = [module.services]
}

module "pypi-registry" {
  source     = "./modules/pypi_artifact_store"
  prefix     = var.prefix
  region     = var.gcp_region
  depends_on = [module.services]
}

module "docker-registry" {
  source     = "./modules/docker_artifact_store"
  prefix     = var.prefix
  region     = var.gcp_region
  depends_on = [module.services]
}

module "gpu-compute-configure" {
  for_each = var.deploy_gpu_compute ? toset(keys(var.users)) : []

  source                  = "./modules/compute_configure"
  prefix                  = "${var.prefix}-gpu"
  username                = each.value
  persistent_disk_size_gb = var.compute_persistent_disk_size_gb
  persistent_disk_type    = var.compute_persistent_disk_type
  image                   = "debian-cloud/debian-11"
  depends_on              = [module.services]
  zone                    = lookup(var.users[each.value][0], "zone", null) == null ? "europe-west4-a" : var.users[each.value][0].zone
}

module "compute-gpu" {
  for_each = var.deploy_gpu_compute ? toset(keys(var.users)) : []

  source             = "./modules/compute"
  prefix             = "${var.prefix}-gpu"
  username           = each.value
  ssh_username       = lookup(var.users[each.value][0], "ssh_username", "") == null ? "" : var.users[each.value][0].ssh_username
  machine_type       = var.users[each.value][0].machine_type
  guest_accelerator  = var.users[each.value][0].guest_accelerator
  boot_disk_image    = var.users[each.value][0].boot_disk_image
  boot_disk_snapshot = lookup(var.users[each.value][0], "boot_disk_snapshot", null) == null ? "" : var.users[each.value][0].boot_disk_snapshot
  zone               = lookup(var.users[each.value][0], "zone", null) == null ? "europe-west4-a" : var.users[each.value][0].zone
  disk_id            = module.gpu-compute-configure[each.value].disk-id
  public_ip          = module.gpu-compute-configure[each.value].public-ip
  network_tags       = local.compute_firewall_tag[each.value]
  depends_on         = [module.services]
}

module "cpu-compute-configure" {
  for_each = var.deploy_cpu_compute ? toset(keys(var.users)) : []

  source                  = "./modules/compute_configure"
  prefix                  = "${var.prefix}-cpu"
  username                = each.value
  persistent_disk_size_gb = var.compute_persistent_disk_size_gb
  persistent_disk_type    = var.compute_persistent_disk_type
  image                   = "debian-cloud/debian-11"
  depends_on              = [module.services]
  zone                    = lookup(var.users[each.value][0], "zone", null) == null ? "europe-west4-a" : var.users[each.value][0].zone
}

module "compute-cpu" {
  for_each = var.deploy_cpu_compute ? toset(keys(var.users)) : []

  source       = "./modules/compute"
  prefix       = "${var.prefix}-cpu"
  username     = each.value
  machine_type = "n2-standard-4"
  disk_id      = module.cpu-compute-configure[each.value].disk-id
  public_ip    = module.cpu-compute-configure[each.value].public-ip
  network_tags = local.compute_firewall_tag[each.value]
  depends_on   = [module.services]
}

module "sql" {
  source        = "./modules/sql"
  prefix        = var.prefix
  username      = var.sql_user
  whitelist_ips = [google_compute_address.static_ip.address]
  depends_on    = [module.services]
}

module "mlflow" {
  source                = "./modules/mlflow"
  prefix                = var.prefix
  sql_instance_name     = module.sql.sql-instance-name
  sql_connection_string = "postgresql+psycopg2://${var.sql_user}:${urlencode(data.google_secret_manager_secret_version.basic.secret_data)}@${module.sql.sql-output-ip}:5432/mlflow?sslmode=disable"
  region                = var.gcp_region
  extra_network_tags    = ["mlflow"]
  machine_type          = "e2-small"
  static_ip             = google_compute_address.static_ip.address
  depends_on            = [module.services]
}

module "kubernetes" {
  source     = "./modules/kubernetes"
  prefix     = var.prefix
  region     = var.gcp_region
  zones      = [var.gcp_zone]
  depends_on = [module.services]
}
