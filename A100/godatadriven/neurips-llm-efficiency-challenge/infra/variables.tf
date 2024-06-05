variable "prefix" {
  type        = string
  description = "Unique identifier that is postpended to all resources."
}

variable "gcp_project" {
  type        = string
  description = "GCP project name"
}

variable "gcp_project_id" {
  type        = string
  description = "GCP project ID"
}

variable "gcp_region" {
  type        = string
  description = "value of the region to deploy to"
}

variable "gcp_zone" {
  type        = string
  description = "value of the zone to deploy to"
}

variable "admins" {
  type        = list(string)
  description = "List of IAM members that will be granted the roles/owner IAM role"
}

variable "users" {
  type = map(list(object({
    machine_type       = string
    guest_accelerator  = string
    ssh_username       = optional(string)
    boot_disk_image    = optional(string)
    boot_disk_snapshot = optional(string)
    zone               = optional(string)
  })))
  description = "List of users for which compute engines will be created"
}

variable "sql_user" {
  type        = string
  description = "Name of the SQL user"
}

variable "deploy_cpu_compute" {
  type        = bool
  description = "Deploy CPU compute engine work stations"
  default     = false
}

variable "deploy_gpu_compute" {
  type        = bool
  description = "Deploy GPU compute engine work stations"
  default     = false
}

variable "compute_persistent_disk_size_gb" {
  type        = number
  description = "Size of the persistent disk in GB"
}

variable "compute_persistent_disk_type" {
  type        = string
  description = "Type of the persistent disk (e.g. 'pd-balanced')"
}

variable "mlflow_whitelist_ips" {
  type        = list(string)
  description = "List of IP addresses to add to the firewall whitelist rule for the MLFlow server."
}
