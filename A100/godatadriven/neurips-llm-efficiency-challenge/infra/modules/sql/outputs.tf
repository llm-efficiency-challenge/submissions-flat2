output "sql-instance-name" {
  value       = google_sql_database_instance.main.name
  description = "Name of the postgres SQL instance"
}

output "sql-output-ip" {
  value       = google_sql_database_instance.main.ip_address.0.ip_address
  description = "IP address of the postgres SQL instance"
}
