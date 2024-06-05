output "kubernetes_endpoint" {
  description = "The cluster endpoint"
  sensitive   = true
  value       = module.gke.endpoint
}

output "cluster_name" {
  description = "Cluster name"
  value       = module.gke.name
}

output "location" {
  value = module.gke.location
}

output "master_kubernetes_version" {
  description = "Kubernetes version of the master"
  value       = module.gke.master_version
}

output "ca_certificate" {
  sensitive   = true
  description = "The cluster ca certificate (base64 encoded)"
  value       = module.gke.ca_certificate
}

output "service_account" {
  description = "The service account to default running nodes as if not overridden in `node_pools`."
  value       = module.gke.service_account
}

output "network_name" {
  description = "The name of the VPC being created"
  value       = module.gcp-network.network_name
}

output "subnet_names" {
  description = "The names of the subnet being created"
  value       = module.gcp-network.subnets_names
}

output "region" {
  description = "The region in which the cluster resides"
  value       = module.gke.region
}

output "zones" {
  description = "List of zones in which the cluster resides"
  value       = module.gke.zones
}

output "project_id" {
  description = "The project ID the cluster is in"
  value       = local.project_id
}

output "service_account_email" {
  description = "The service account to default running nodes as if not overridden in `node_pools`."
  value       = google_service_account.kubernetes.email
}

output "pods_ip_range" {
  description = "The IP range in CIDR notation to use for the pods in this cluster."
  value       = module.gcp-network.subnets_secondary_ranges[1][0].ip_cidr_range
}
