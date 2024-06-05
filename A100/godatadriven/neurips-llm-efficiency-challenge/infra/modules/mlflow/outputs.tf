output "mlflow_service_account_mail" {
  value       = google_service_account.mlflow.email
  description = "Email of the service account to use for MLflow"
}

output "mlflow-artifact-bucket-name" {
  value       = google_storage_bucket.mlflow.name
  description = "Name of the bucket to use for MLflow artifacts."
}
