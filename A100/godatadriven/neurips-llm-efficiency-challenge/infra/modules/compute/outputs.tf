output "connection-details" {
  value = {
    "public_ip"              = var.public_ip,
    "private_key_secret_key" = local.private_key_secret_name
  }
  description = "Details needed to connect to the compute instance"
}

output "service-account-email" {
  value       = google_service_account.compute.email
  description = "Service account email"
}
