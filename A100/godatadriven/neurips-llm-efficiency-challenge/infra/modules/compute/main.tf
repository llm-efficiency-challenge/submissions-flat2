locals {
  private_key_secret_name = upper("compute-${var.prefix}-${var.username}-private-key")
  ssh_username            = var.ssh_username == "" ? var.username : var.ssh_username
}

resource "tls_private_key" "ssh" {
  algorithm = "RSA"
  rsa_bits  = 4096
}

resource "google_service_account" "compute" {
  account_id  = "compute-${var.username}-sa"
  description = "Terraform-managed service account for compute-${var.prefix}-${var.username}"
}

resource "google_compute_disk" "compute" {
  count                     = var.boot_disk_snapshot == "" ? 0 : 1
  name                      = "compute-${var.prefix}-${var.username}-boot-disk"
  description               = "Terraform-managed boot disk for compute-${var.prefix}-${var.username}"
  size                      = 100
  physical_block_size_bytes = 4096
  snapshot                  = var.boot_disk_snapshot
  zone                      = var.zone
}

resource "google_compute_instance" "compute" {
  provider     = google-beta
  name         = "compute-${var.prefix}-${var.username}"
  machine_type = var.machine_type
  dynamic "guest_accelerator" {
    for_each = var.guest_accelerator == "" ? [] : [1]

    content {
      type  = var.guest_accelerator
      count = 1
    }
  }

  tags = var.network_tags
  zone = var.zone

  metadata = {
    ssh-keys = "${local.ssh_username}:${tls_private_key.ssh.public_key_openssh}"
  }

  metadata_startup_script = file("${abspath(path.module)}/static/cpu_monitor.sh")

  boot_disk {
    // Use image if snapshot not given.
    dynamic "initialize_params" {
      for_each = var.boot_disk_snapshot == "" ? [1] : []

      content {
        image = var.boot_disk_image
      }
    }

    source = (var.boot_disk_snapshot == "" ? null : google_compute_disk.compute[0].self_link)
  }

  network_interface {
    network = "default"

    access_config {
      nat_ip = var.public_ip
    }
  }

  service_account {
    email = google_service_account.compute.email
    scopes = [
      "https://www.googleapis.com/auth/devstorage.read_only",
      "https://www.googleapis.com/auth/logging.write",
      "https://www.googleapis.com/auth/monitoring.write",
      "https://www.googleapis.com/auth/service.management.readonly",
      "https://www.googleapis.com/auth/trace.append",
      "https://www.googleapis.com/auth/logging.write",
      "https://www.googleapis.com/auth/monitoring.read",
      "https://www.googleapis.com/auth/monitoring.write",
      "https://www.googleapis.com/auth/devstorage.full_control"
    ]
  }

  scheduling {
    on_host_maintenance = "TERMINATE"
  }

  lifecycle {
    ignore_changes = [attached_disk]
  }

}

resource "google_compute_attached_disk" "default" {
  disk     = var.disk_id
  instance = google_compute_instance.compute.id
}

resource "google_secret_manager_secret" "private_key" {

  secret_id = local.private_key_secret_name

  labels = {
    terraform = true
  }

  replication {
    automatic = true
  }
}

resource "google_secret_manager_secret_version" "private_key" {
  secret      = google_secret_manager_secret.private_key.id
  secret_data = base64encode(tls_private_key.ssh.private_key_pem)
}
