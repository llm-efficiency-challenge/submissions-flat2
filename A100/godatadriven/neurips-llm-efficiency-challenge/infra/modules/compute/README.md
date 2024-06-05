<!-- BEGIN_TF_DOCS -->
## Terraform Documentation
Below is the automatically generated Terraform documentation.



### Resources

| Name | Type |
|------|------|
| [google-beta_google_compute_instance.compute](https://registry.terraform.io/providers/hashicorp/google-beta/latest/docs/resources/google_compute_instance) | resource |
| [google_compute_attached_disk.default](https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/compute_attached_disk) | resource |
| [google_compute_disk.compute](https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/compute_disk) | resource |
| [google_secret_manager_secret.private_key](https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/secret_manager_secret) | resource |
| [google_secret_manager_secret_version.private_key](https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/secret_manager_secret_version) | resource |
| [google_service_account.compute](https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/service_account) | resource |
| [tls_private_key.ssh](https://registry.terraform.io/providers/hashicorp/tls/latest/docs/resources/private_key) | resource |

### Inputs

| Name | Description | Type | Default | Required |
|------|-------------|------|---------|:--------:|
| <a name="input_boot_disk_image"></a> [boot\_disk\_image](#input\_boot\_disk\_image) | Boot disk image to use for the VM | `string` | `"projects/ml-images/global/images/c0-deeplearning-common-cu113-v20230615-debian-11-py310"` | no |
| <a name="input_boot_disk_snapshot"></a> [boot\_disk\_snapshot](#input\_boot\_disk\_snapshot) | Name of the snapshot to use for the boot disk. This will override the boot\_disk\_image variable. | `string` | `""` | no |
| <a name="input_disk_id"></a> [disk\_id](#input\_disk\_id) | ID of the disk to attach to the VM | `string` | n/a | yes |
| <a name="input_guest_accelerator"></a> [guest\_accelerator](#input\_guest\_accelerator) | Guest accelerator to use for the VM (e.g. 'nvidia-tesla-t4') | `string` | `""` | no |
| <a name="input_machine_type"></a> [machine\_type](#input\_machine\_type) | Machine type to use for the VM | `string` | n/a | yes |
| <a name="input_network_tags"></a> [network\_tags](#input\_network\_tags) | Network tags to apply to the VM | `list(string)` | `null` | no |
| <a name="input_prefix"></a> [prefix](#input\_prefix) | Prefix for all resources | `string` | n/a | yes |
| <a name="input_public_ip"></a> [public\_ip](#input\_public\_ip) | Public IP to assign to the VM | `string` | `null` | no |
| <a name="input_ssh_username"></a> [ssh\_username](#input\_ssh\_username) | Username that will be used to log into the VM. This will override the username variable. NB: this can be useful for some images/disk snapshots that you pre-configured a lot of things under a specific username. | `string` | `""` | no |
| <a name="input_username"></a> [username](#input\_username) | Username with which you will log into the VM and that will be used to name resources. If you pass 'ssh-username' this will override the username variable. | `string` | n/a | yes |
| <a name="input_zone"></a> [zone](#input\_zone) | Zone to deploy the VM in | `string` | `"europe-west4-a"` | no |

### Outputs

| Name | Description |
|------|-------------|
| <a name="output_connection-details"></a> [connection-details](#output\_connection-details) | Details needed to connect to the compute instance |
| <a name="output_service-account-email"></a> [service-account-email](#output\_service-account-email) | Service account email |
<!-- END_TF_DOCS -->
