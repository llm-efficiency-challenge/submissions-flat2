<!-- BEGIN_TF_DOCS -->
## Terraform Documentation
Below is the automatically generated Terraform documentation.



### Resources

| Name | Type |
|------|------|
| [google_compute_address.static_ip](https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/compute_address) | resource |
| [google_compute_disk.default](https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/compute_disk) | resource |

### Inputs

| Name | Description | Type | Default | Required |
|------|-------------|------|---------|:--------:|
| <a name="input_image"></a> [image](#input\_image) | Image to use for the VM. See <https://cloud.google.com/compute/docs/images> | `string` | `"debian-cloud/debian-11"` | no |
| <a name="input_persistent_disk_size_gb"></a> [persistent\_disk\_size\_gb](#input\_persistent\_disk\_size\_gb) | Size of the persistent disk in GB | `number` | `100` | no |
| <a name="input_persistent_disk_type"></a> [persistent\_disk\_type](#input\_persistent\_disk\_type) | Type of the persistent disk | `string` | `"pd-balanced"` | no |
| <a name="input_prefix"></a> [prefix](#input\_prefix) | Prefix for all resources | `string` | n/a | yes |
| <a name="input_username"></a> [username](#input\_username) | Username with which you will log into the VM | `string` | n/a | yes |
| <a name="input_zone"></a> [zone](#input\_zone) | Zone to deploy the VM in | `string` | `"europe-west4-a"` | no |

### Outputs

| Name | Description |
|------|-------------|
| <a name="output_disk-id"></a> [disk-id](#output\_disk-id) | ID of the disk |
| <a name="output_public-ip"></a> [public-ip](#output\_public-ip) | Public that can be assigned to a compute instance. |
<!-- END_TF_DOCS -->
