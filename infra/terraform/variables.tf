variable "project_id" {
  description = "Google Cloud project that owns the hackathon deployment."
  type        = string
}

variable "region" {
  description = "Region for Cloud Run, Artifact Storage, and Vertex AI."
  type        = string
  default     = "us-central1"
}

variable "firestore_location" {
  description = "Firestore multi-region or region identifier."
  type        = string
  default     = "nam5"
}

variable "name_prefix" {
  description = "Prefix for deployed resource names."
  type        = string
  default     = "post-training-demo"
}

variable "container_image" {
  description = "Immutable Artifact Registry image reference shared by all three service roles."
  type        = string
}

variable "training_container_image" {
  description = "Vertex AI custom training image containing the QLoRA worker entrypoint."
  type        = string
}

variable "objective_execution_url" {
  description = "HTTPS endpoint for the separately deployed sandboxed model and AgentGym evidence worker."
  type        = string

  validation {
    condition     = startswith(var.objective_execution_url, "https://")
    error_message = "objective_execution_url must be an absolute HTTPS URL."
  }
}

variable "rag_corpus_uri" {
  description = "GCS URI of the pre-uploaded leakage-safe research corpus JSON."
  type        = string

  validation {
    condition     = startswith(var.rag_corpus_uri, "gs://")
    error_message = "rag_corpus_uri must be an absolute gs:// URI."
  }
}

variable "rag_corpus_sha256" {
  description = "Optional lowercase SHA-256 digest pinned to the RAG corpus bytes."
  type        = string
  default     = ""

  validation {
    condition     = var.rag_corpus_sha256 == "" || can(regex("^[a-f0-9]{64}$", var.rag_corpus_sha256))
    error_message = "rag_corpus_sha256 must be empty or a lowercase SHA-256 digest."
  }
}

variable "artifact_bucket_name" {
  description = "Optional globally unique bucket name. An address derived from project_id is used when empty."
  type        = string
  default     = ""
}

variable "target_model" {
  description = "FunctionGemma model identifier recorded in run provenance."
  type        = string
  default     = "google/functiongemma-270m-it"
}

variable "gemini_model" {
  description = "Resolved Gemini model ID used by research agents."
  type        = string
  default     = "gemini-3.5-flash"
}

variable "max_candidates" {
  description = "Hard ceiling for candidate experiments in one research run."
  type        = number
  default     = 2

  validation {
    condition     = var.max_candidates >= 1 && var.max_candidates <= 2
    error_message = "max_candidates must be either 1 or 2 for the hackathon scope."
  }
}

variable "compute_budget_minutes" {
  description = "Per-run cloud work ceiling, kept below Cloud Run's request timeout."
  type        = number
  default     = 55

  validation {
    condition     = var.compute_budget_minutes >= 1 && var.compute_budget_minutes <= 55
    error_message = "compute_budget_minutes must be between 1 and 55."
  }
}

variable "a2a_timeout_seconds" {
  description = "Coordinator timeout for one authenticated A2A request."
  type        = number
  default     = 3300

  validation {
    condition     = var.a2a_timeout_seconds >= 1 && var.a2a_timeout_seconds <= 3500
    error_message = "a2a_timeout_seconds must be between 1 and 3500."
  }
}

variable "research_service_url" {
  description = "Optional second-pass research URL used only to publish an exact A2A Agent Card."
  type        = string
  default     = ""
}

variable "execution_service_url" {
  description = "Optional second-pass execution URL used only to publish an exact A2A Agent Card."
  type        = string
  default     = ""
}

variable "allow_unauthenticated" {
  description = "Expose Cloud Run services publicly for a judge demo. Keep false until explicit demo deployment."
  type        = bool
  default     = false
}

variable "deletion_protection" {
  description = "Protect Cloud Run services against deletion outside disposable hackathon environments."
  type        = bool
  default     = false
}
