variable "prefix" {
  description = "Prefix for all resources"
  type        = string
}

variable "static_ip" {
  description = "Static IP address to assign to the VM"
  type        = string
}

variable "sql_instance_name" {
  type        = string
  description = "Name of the SQL instance in which the database will be created"
}

variable "sql_connection_string" {
  type        = string
  description = "Connection string to the SQL instance"
}

variable "zone" {
  description = "Zone to deploy the VM in"
  type        = string
  default     = "europe-west4-a"
}

variable "region" {
  type        = string
  description = "Region to deploy the storage bucket in"
}

variable "machine_type" {
  type        = string
  description = "Machine type to use for the VM"
  default     = "e2-small"
}

variable "extra_network_tags" {
  type        = list(string)
  description = "Additional network tags to add to the VM"
  default     = []
}
