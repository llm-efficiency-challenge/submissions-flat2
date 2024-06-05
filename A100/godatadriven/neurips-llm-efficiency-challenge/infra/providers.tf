terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 3.5.0"
    }
    kubernetes = {
      source = "hashicorp/kubernetes"
    }
  }
  backend "gcs" {
    bucket = "tfstate-neurips-2023"
    prefix = "terraform/state"
  }
}

provider "google" {
  credentials = file("../.secrets/neurips-llm-eff-challenge-2023-1fad172c3905.json")

  project = var.gcp_project
  region  = var.gcp_region
  zone    = var.gcp_zone
}

# https://developer.hashicorp.com/terraform/tutorials/azure-get-started/azure-remote
provider "google-beta" {
  credentials = file("../.secrets/neurips-llm-eff-challenge-2023-1fad172c3905.json")

  project = var.gcp_project
  region  = var.gcp_region
  zone    = var.gcp_zone
}

provider "tls" {
  // no config needed
}

provider "kubernetes" {
  host                   = "https://${module.kubernetes.kubernetes_endpoint}"
  token                  = data.google_client_config.default.access_token
  cluster_ca_certificate = base64decode(module.kubernetes.ca_certificate)
}
