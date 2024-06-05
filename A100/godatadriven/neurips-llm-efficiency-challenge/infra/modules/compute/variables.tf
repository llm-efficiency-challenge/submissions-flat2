variable "prefix" {
  description = "Prefix for all resources"
  type        = string
}

variable "username" {
  description = "Username with which you will log into the VM and that will be used to name resources. If you pass 'ssh-username' this will override the username variable."
  type        = string
}

variable "ssh_username" {
  description = "Username that will be used to log into the VM. This will override the username variable. NB: this can be useful for some images/disk snapshots that you pre-configured a lot of things under a specific username."
  type        = string
  default     = ""
}

variable "machine_type" {
  description = "Machine type to use for the VM"
  type        = string
}

variable "guest_accelerator" {
  description = "Guest accelerator to use for the VM (e.g. 'nvidia-tesla-t4')"
  type        = string
  default     = ""
}

variable "boot_disk_image" {
  description = "Boot disk image to use for the VM"
  type        = string
  default     = "projects/ml-images/global/images/c0-deeplearning-common-cu113-v20230615-debian-11-py310"
}

variable "boot_disk_snapshot" {
  description = "Name of the snapshot to use for the boot disk. This will override the boot_disk_image variable."
  type        = string
  default     = ""
}

variable "disk_id" {
  description = "ID of the disk to attach to the VM"
  type        = string
}

variable "zone" {
  description = "Zone to deploy the VM in"
  type        = string
  default     = "europe-west4-a"
}

variable "public_ip" {
  description = "Public IP to assign to the VM"
  type        = string
  default     = null
}

variable "network_tags" {
  description = "Network tags to apply to the VM"
  type        = list(string)
  default     = null
}
