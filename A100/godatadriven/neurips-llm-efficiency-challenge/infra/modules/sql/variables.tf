variable "prefix" {
  description = "Prefix for all resources"
  type        = string
}

variable "username" {
  description = "SQL username"
  type        = string
}

variable "region" {
  description = "Region to deploy the SQL server in"
  type        = string
  default     = "europe-west4"
}

variable "whitelist_ips" {
  type        = list(string)
  description = "List of IP addresses to whitelist"
}
