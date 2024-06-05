data "google_compute_network" "default" {
  name = "default"
}

data "google_compute_network" "kubernetes" {
  name = module.kubernetes.network_name
}

// Allow access to MLflow from whitelisted IPs
resource "google_compute_firewall" "mlflow-access" {

  name          = "mlflow"
  network       = "default"
  target_tags   = ["mlflow"]
  source_ranges = local.mlflow_whitelist_ips

  allow {
    protocol = "tcp"
    ports    = ["5000"]
  }
}

// Assign static IP for mlflow compute VM
resource "google_compute_address" "static_ip" {
  name = "mlflow-${var.prefix}"
}

// Allow SSH access to compute instances
resource "google_compute_firewall" "allow_ssh" {
  for_each = toset(keys(var.users))

  name          = "allow-ssh-${var.prefix}-${each.value}"
  network       = "default"
  target_tags   = ["allow-ssh-${var.prefix}-${each.value}"]
  source_ranges = ["0.0.0.0/0"]

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }
}

// Create a peering connection between 'default' and k8s vpc
// This allows pods to access mlflow tracking on the internal
// ip of the mlflow compute instance
module "peering" {
  source = "terraform-google-modules/network/google//modules/network-peering"

  prefix        = var.prefix
  local_network = data.google_compute_network.default.id
  peer_network  = data.google_compute_network.kubernetes.id
}
