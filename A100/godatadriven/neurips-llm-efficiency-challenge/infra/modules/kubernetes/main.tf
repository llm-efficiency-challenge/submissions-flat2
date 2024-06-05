locals {
  prefix_appended        = "${var.prefix}-autopilot"
  cluster_type           = "cluster-${local.prefix_appended}"
  network_name           = "network-${local.prefix_appended}"
  subnet_name            = "subnet-${local.prefix_appended}"
  master_auth_subnetwork = "master-subnet-${local.prefix_appended}"
  pods_range_name        = "ip-range-pods-${local.prefix_appended}"
  svc_range_name         = "ip-range-svc-${local.prefix_appended}"
  subnet_names           = [for subnet_self_link in module.gcp-network.subnets_self_links : split("/", subnet_self_link)[length(split("/", subnet_self_link)) - 1]]
  project_id             = replace(data.google_project.project.id, "projects/", "")
}

data "google_project" "project" {}

module "gke" {
  source                          = "terraform-google-modules/kubernetes-engine/google//modules/beta-autopilot-private-cluster"
  project_id                      = local.project_id
  name                            = local.cluster_type
  regional                        = true
  region                          = var.region
  zones                           = var.zones
  network                         = module.gcp-network.network_name
  subnetwork                      = local.subnet_names[index(module.gcp-network.subnets_names, local.subnet_name)]
  ip_range_pods                   = local.pods_range_name
  ip_range_services               = local.svc_range_name
  release_channel                 = "REGULAR"
  enable_vertical_pod_autoscaling = true
  enable_private_endpoint         = false
  enable_private_nodes            = false
  master_ipv4_cidr_block          = "172.16.0.0/28"
  network_tags                    = [local.cluster_type]
  service_account                 = google_service_account.kubernetes.email
}

module "workload_identity_existing_gsa" {
  source              = "terraform-google-modules/kubernetes-engine/google//modules/workload-identity"
  project_id          = local.project_id
  name                = google_service_account.kubernetes.account_id
  namespace           = "default"
  use_existing_gcp_sa = true
  depends_on          = [google_service_account.kubernetes]
}
