output "service_urls" {
  description = "Cloud Run URLs keyed by service role."
  value = merge(
    { for role, service in google_cloud_run_v2_service.team : role => service.uri },
    { coordinator = google_cloud_run_v2_service.coordinator.uri },
  )
}

output "artifact_bucket" {
  description = "Versioned bucket for datasets, checkpoints, manifests, and evaluation reports."
  value       = google_storage_bucket.artifacts.name
}

output "runtime_service_account" {
  description = "Least-privilege runtime identity shared by the demo services."
  value       = google_service_account.runtime.email
}

output "hf_token_secret" {
  description = "Secret resource created without a secret value. Add a version outside Terraform."
  value       = google_secret_manager_secret.hf_token.secret_id
}
