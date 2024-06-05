locals {
  instance_target_tags = "allow-ssh-${var.prefix}-${var.username}"
  region               = join("-", slice(split("-", var.zone), 0, 2))
}

resource "google_compute_disk" "default" {
  name  = "persistent-disk-${var.prefix}-${var.username}"
  type  = var.persistent_disk_type
  image = var.image
  size  = var.persistent_disk_size_gb
  zone  = var.zone
}

resource "google_compute_address" "static_ip" {
  name   = "compute-${var.prefix}-${var.username}"
  region = local.region
}
