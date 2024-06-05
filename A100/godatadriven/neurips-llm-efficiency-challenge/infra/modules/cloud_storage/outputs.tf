output "data-bucket-name" {
  value       = google_storage_bucket.data.name
  description = "Name of the bucket that can be used to store data."
}

output "evaluation-bucket-name" {
  value       = google_storage_bucket.evaluation-results.name
  description = "Name of the bucket that can be used to store evaluation results."
}
