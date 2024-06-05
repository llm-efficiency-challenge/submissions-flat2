prefix         = "neurips-2023"
gcp_project    = "neurips-llm-eff-challenge-2023"
gcp_project_id = "neurips-llm-eff-challenge-2023"
gcp_region     = "europe-west4"
gcp_zone       = "europe-west4-a"

deploy_cpu_compute              = false
deploy_gpu_compute              = true
compute_persistent_disk_size_gb = 200
compute_persistent_disk_type    = "pd-balanced"
sql_user                        = "neurips2023"
mlflow_whitelist_ips = [
  "37.17.221.89",
  "77.166.246.213",
  "84.84.213.50",
  "217.103.236.29", # Jetze's home
]

admins = [
  "jordi.smit@xebia.com",
  "jetze.schuurmans@xebia.com",
  "caio.benatti@xebia.com",
  "rens.dimmendaal@xebia.com",
  "jasper.ginn@xebia.com",
  "yke.rusticus@xebia.com"
]
# For accelerators, see: gcloud compute accelerator-types list
users = {
  "jordi" : [{ "machine_type" : "n1-standard-4", "guest_accelerator" : "nvidia-tesla-t4", "boot_disk_snapshot" : "neurips-boot-disk-snapshot", "ssh_username" : "user" }],
  "jetze" : [{ "machine_type" : "n1-standard-4", "guest_accelerator" : "nvidia-tesla-t4", "boot_disk_snapshot" : "neurips-boot-disk-snapshot", "ssh_username" : "user" }],
  "jetze-a100" : [{ "machine_type" : "a2-highgpu-1g", "guest_accelerator" : "nvidia-tesla-a100", "boot_disk_snapshot" : "neurips-boot-disk-snapshot", "ssh_username" : "user", "zone" : "asia-northeast1-c" }],
  "caio" : [{ "machine_type" : "n1-standard-4", "guest_accelerator" : "nvidia-tesla-t4", "boot_disk_snapshot" : "neurips-boot-disk-snapshot", "ssh_username" : "user" }],
  "jasper-l4" : [{ "machine_type" : "g2-standard-12", "guest_accelerator" : "nvidia-l4", "boot_disk_snapshot" : "neurips-boot-disk-snapshot", "ssh_username" : "user", "zone" : "europe-west4-a" }],
  "jasper-a100" : [{ "machine_type" : "a2-highgpu-1g", "guest_accelerator" : "nvidia-tesla-a100", "boot_disk_snapshot" : "neurips-boot-disk-snapshot", "ssh_username" : "user", "zone" : "asia-northeast1-c" }],
  "yke" : [{ "machine_type" : "n1-standard-4", "guest_accelerator" : "nvidia-tesla-t4", "boot_disk_snapshot" : "neurips-boot-disk-snapshot", "ssh_username" : "user" }],
}
