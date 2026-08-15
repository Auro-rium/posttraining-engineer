locals {
  team_roles  = toset(["research", "execution"])
  bucket_name = var.artifact_bucket_name != "" ? var.artifact_bucket_name : "${var.project_id}-${var.name_prefix}-artifacts"

  required_apis = toset([
    "aiplatform.googleapis.com",
    "artifactregistry.googleapis.com",
    "firestore.googleapis.com",
    "logging.googleapis.com",
    "run.googleapis.com",
    "secretmanager.googleapis.com",
    "storage.googleapis.com",
    "trace.googleapis.com",
  ])

  runtime_roles = toset([
    "roles/aiplatform.user",
    "roles/artifactregistry.reader",
    "roles/cloudtrace.agent",
    "roles/datastore.user",
    "roles/logging.logWriter",
  ])
}

resource "google_project_service" "required" {
  for_each = local.required_apis

  service            = each.value
  disable_on_destroy = false
}

resource "google_service_account" "runtime" {
  account_id   = "${var.name_prefix}-runtime"
  display_name = "Autonomous post-training demo runtime"

  depends_on = [google_project_service.required]
}

resource "google_project_iam_member" "runtime" {
  for_each = local.runtime_roles

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.runtime.email}"
}

resource "google_storage_bucket" "artifacts" {
  name                        = local.bucket_name
  location                    = var.region
  uniform_bucket_level_access = true
  force_destroy               = false

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age        = 30
      with_state = "ARCHIVED"
    }
    action {
      type = "Delete"
    }
  }

  depends_on = [google_project_service.required]
}

resource "google_storage_bucket_iam_member" "runtime_artifacts" {
  bucket = google_storage_bucket.artifacts.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.runtime.email}"
}

resource "google_service_account_iam_member" "runtime_act_as_self" {
  service_account_id = google_service_account.runtime.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.runtime.email}"
}

resource "google_firestore_database" "state" {
  project     = var.project_id
  name        = "(default)"
  location_id = var.firestore_location
  type        = "FIRESTORE_NATIVE"

  lifecycle {
    prevent_destroy = true
  }

  depends_on = [google_project_service.required]
}

resource "google_secret_manager_secret" "hf_token" {
  secret_id = "${var.name_prefix}-hf-token"

  replication {
    auto {}
  }

  depends_on = [google_project_service.required]
}

resource "google_secret_manager_secret_iam_member" "hf_token_accessor" {
  secret_id = google_secret_manager_secret.hf_token.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.runtime.email}"
}

resource "google_cloud_run_v2_service" "team" {
  for_each = local.team_roles

  name                = "${var.name_prefix}-${each.key}"
  location            = var.region
  deletion_protection = var.deletion_protection

  template {
    service_account = google_service_account.runtime.email
    timeout         = "3600s"

    scaling {
      min_instance_count = 0
      max_instance_count = 2
    }

    containers {
      image = var.container_image

      ports {
        container_port = 8080
      }

      resources {
        limits = {
          cpu    = "1"
          memory = "1Gi"
        }
      }

      env {
        name  = "ENVIRONMENT"
        value = "cloud"
      }
      env {
        name  = "SERVICE_ROLE"
        value = each.key
      }
      env {
        name  = "GOOGLE_CLOUD_PROJECT"
        value = var.project_id
      }
      env {
        name  = "GOOGLE_CLOUD_LOCATION"
        value = var.region
      }
      env {
        name  = "GOOGLE_GENAI_USE_VERTEXAI"
        value = "TRUE"
      }
      env {
        name  = "STATE_BACKEND"
        value = "firestore"
      }
      env {
        name  = "ARTIFACT_BACKEND"
        value = "gcs"
      }
      env {
        name  = "ARTIFACT_BUCKET"
        value = google_storage_bucket.artifacts.name
      }
      env {
        name  = "VERTEX_STAGING_BUCKET"
        value = "gs://${google_storage_bucket.artifacts.name}"
      }
      env {
        name  = "TRAINING_CONTAINER_URI"
        value = var.training_container_image
      }
      env {
        name  = "HF_SECRET_ID"
        value = google_secret_manager_secret.hf_token.secret_id
      }
      env {
        name  = "OBJECTIVE_EXECUTION_URL"
        value = var.objective_execution_url
      }
      env {
        name  = "TARGET_MODEL"
        value = var.target_model
      }
      env {
        name  = "GEMINI_MODEL"
        value = var.gemini_model
      }
      env {
        name  = "MAX_CANDIDATES"
        value = tostring(var.max_candidates)
      }
      env {
        name  = "COMPUTE_BUDGET_MINUTES"
        value = tostring(var.compute_budget_minutes)
      }
      env {
        name  = "A2A_TIMEOUT_SECONDS"
        value = tostring(var.a2a_timeout_seconds)
      }
      env {
        name  = "OTEL_SERVICE_NAME"
        value = "${var.name_prefix}-${each.key}"
      }
      env {
        name  = "OTEL_EXPORT_TO_CLOUD"
        value = "true"
      }
      env {
        name  = "OTEL_CAPTURE_CONTENT"
        value = "false"
      }

      dynamic "env" {
        for_each = each.key == "research" && var.research_service_url != "" ? [1] : []
        content {
          name  = "PUBLIC_SERVICE_URL"
          value = var.research_service_url
        }
      }
      dynamic "env" {
        for_each = each.key == "execution" && var.execution_service_url != "" ? [1] : []
        content {
          name  = "PUBLIC_SERVICE_URL"
          value = var.execution_service_url
        }
      }
      dynamic "env" {
        for_each = each.key == "research" ? [1] : []
        content {
          name  = "RAG_CORPUS_URI"
          value = var.rag_corpus_uri
        }
      }
      dynamic "env" {
        for_each = each.key == "research" && var.rag_corpus_sha256 != "" ? [1] : []
        content {
          name  = "RAG_CORPUS_SHA256"
          value = var.rag_corpus_sha256
        }
      }
      startup_probe {
        http_get {
          path = "/.well-known/agent-card.json"
        }
        initial_delay_seconds = 2
        timeout_seconds       = 2
        period_seconds        = 5
        failure_threshold     = 12
      }

      liveness_probe {
        http_get {
          path = "/.well-known/agent-card.json"
        }
        timeout_seconds   = 2
        period_seconds    = 10
        failure_threshold = 3
      }
    }
  }

  depends_on = [
    google_firestore_database.state,
    google_project_iam_member.runtime,
    google_secret_manager_secret_iam_member.hf_token_accessor,
    google_storage_bucket_iam_member.runtime_artifacts,
  ]
}

resource "google_cloud_run_v2_service" "coordinator" {
  name                = "${var.name_prefix}-coordinator"
  location            = var.region
  deletion_protection = var.deletion_protection

  template {
    service_account = google_service_account.runtime.email
    timeout         = "3600s"

    scaling {
      min_instance_count = 1
      max_instance_count = 2
    }

    containers {
      image = var.container_image

      ports {
        container_port = 8080
      }

      resources {
        cpu_idle = false
        limits = {
          cpu    = "1"
          memory = "1Gi"
        }
      }

      env {
        name  = "ENVIRONMENT"
        value = "cloud"
      }
      env {
        name  = "SERVICE_ROLE"
        value = "coordinator"
      }
      env {
        name  = "GOOGLE_CLOUD_PROJECT"
        value = var.project_id
      }
      env {
        name  = "GOOGLE_CLOUD_LOCATION"
        value = var.region
      }
      env {
        name  = "GOOGLE_GENAI_USE_VERTEXAI"
        value = "TRUE"
      }
      env {
        name  = "STATE_BACKEND"
        value = "firestore"
      }
      env {
        name  = "ARTIFACT_BACKEND"
        value = "gcs"
      }
      env {
        name  = "ARTIFACT_BUCKET"
        value = google_storage_bucket.artifacts.name
      }
      env {
        name  = "VERTEX_STAGING_BUCKET"
        value = "gs://${google_storage_bucket.artifacts.name}"
      }
      env {
        name  = "TRAINING_CONTAINER_URI"
        value = var.training_container_image
      }
      env {
        name  = "HF_SECRET_ID"
        value = google_secret_manager_secret.hf_token.secret_id
      }
      env {
        name  = "TARGET_MODEL"
        value = var.target_model
      }
      env {
        name  = "GEMINI_MODEL"
        value = var.gemini_model
      }
      env {
        name  = "MAX_CANDIDATES"
        value = tostring(var.max_candidates)
      }
      env {
        name  = "COMPUTE_BUDGET_MINUTES"
        value = tostring(var.compute_budget_minutes)
      }
      env {
        name  = "A2A_TIMEOUT_SECONDS"
        value = tostring(var.a2a_timeout_seconds)
      }
      env {
        name  = "OTEL_SERVICE_NAME"
        value = "${var.name_prefix}-coordinator"
      }
      env {
        name  = "OTEL_EXPORT_TO_CLOUD"
        value = "true"
      }
      env {
        name  = "OTEL_CAPTURE_CONTENT"
        value = "false"
      }
      env {
        name  = "RESEARCH_A2A_URL"
        value = google_cloud_run_v2_service.team["research"].uri
      }
      env {
        name  = "EXECUTION_A2A_URL"
        value = google_cloud_run_v2_service.team["execution"].uri
      }

      startup_probe {
        http_get {
          path = "/health"
        }
        initial_delay_seconds = 2
        timeout_seconds       = 2
        period_seconds        = 5
        failure_threshold     = 12
      }

      liveness_probe {
        http_get {
          path = "/health"
        }
        timeout_seconds   = 2
        period_seconds    = 10
        failure_threshold = 3
      }
    }
  }

  depends_on = [
    google_cloud_run_v2_service.team,
    google_firestore_database.state,
    google_project_iam_member.runtime,
    google_secret_manager_secret_iam_member.hf_token_accessor,
    google_storage_bucket_iam_member.runtime_artifacts,
  ]
}

resource "google_cloud_run_v2_service_iam_member" "team_public" {
  for_each = var.allow_unauthenticated ? local.team_roles : toset([])

  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.team[each.key].name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

resource "google_cloud_run_v2_service_iam_member" "coordinator_public" {
  count = var.allow_unauthenticated ? 1 : 0

  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.coordinator.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

resource "google_cloud_run_v2_service_iam_member" "team_runtime_invoker" {
  for_each = local.team_roles

  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.team[each.key].name
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.runtime.email}"
}

resource "google_cloud_run_v2_service_iam_member" "coordinator_runtime_invoker" {
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.coordinator.name
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.runtime.email}"
}
