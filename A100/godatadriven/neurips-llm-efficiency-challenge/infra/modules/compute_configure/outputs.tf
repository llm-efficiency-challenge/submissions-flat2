output "disk-id" {
  value       = google_compute_disk.default.id
  description = "ID of the disk"
}

output "public-ip" {
  value       = google_compute_address.static_ip.address
  description = "Public that can be assigned to a compute instance."
}
