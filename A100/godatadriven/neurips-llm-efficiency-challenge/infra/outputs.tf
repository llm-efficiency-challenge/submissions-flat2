output "data-bucket-name" {
  value       = module.storage.data-bucket-name
  description = "Name of the 'data' bucket that can be used to store data."
}

output "data-bucket-service-account" {
  value       = google_service_account.data-storage-admin.email
  description = "Service account email that can be used to read/write data to the 'data' bucket"
}

output "docker-registry-name" {
  value       = module.docker-registry.registry-name
  description = "Name of the docker artifact registry."
}

output "pypi-registry-name" {
  value       = module.pypi-registry.registry-name
  description = "Name of the pypi artifact registry."
}

output "compute-gpu" {
  value       = local.compute_gpu_connection_details
  description = "Details needed to connect to the GPU compute instance"
  sensitive   = true
}

output "compute-cpu" {
  value       = local.compute_cpu_connection_details
  description = "Details needed to connect to the CPU compute instance"
  sensitive   = true
}

output "pods_ip_range" {
  value = module.kubernetes.pods_ip_range
}
