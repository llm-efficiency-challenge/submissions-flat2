variable "prefix" {
  description = "Prefix for all resources"
  type        = string
}

variable "username" {
  description = "Username with which you will log into the VM"
  type        = string
}

variable "persistent_disk_type" {
  description = "Type of the persistent disk"
  type        = string
  default     = "pd-balanced"
}

variable "persistent_disk_size_gb" {
  description = "Size of the persistent disk in GB"
  type        = number
  default     = 100
}

variable "image" {
  description = "Image to use for the VM. See <https://cloud.google.com/compute/docs/images>"
  type        = string
  default     = "debian-cloud/debian-11"
}

variable "zone" {
  description = "Zone to deploy the VM in"
  type        = string
  default     = "europe-west4-a"
}
