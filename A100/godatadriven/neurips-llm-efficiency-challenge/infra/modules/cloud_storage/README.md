# Cloud storage

Adds cloud storage buckets for files and one for logs.

<!-- BEGIN_TF_DOCS -->
## Terraform Documentation
Below is the automatically generated Terraform documentation.



### Resources

| Name | Type |
|------|------|
| [google_storage_bucket.data](https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/storage_bucket) | resource |
| [google_storage_bucket.data-logs](https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/storage_bucket) | resource |
| [google_storage_bucket.evaluation-results](https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/storage_bucket) | resource |

### Inputs

| Name | Description | Type | Default | Required |
|------|-------------|------|---------|:--------:|
| <a name="input_prefix"></a> [prefix](#input\_prefix) | n/a | `string` | n/a | yes |
| <a name="input_region"></a> [region](#input\_region) | n/a | `string` | n/a | yes |

### Outputs

| Name | Description |
|------|-------------|
| <a name="output_data-bucket-name"></a> [data-bucket-name](#output\_data-bucket-name) | Name of the bucket that can be used to store data. |
| <a name="output_evaluation-bucket-name"></a> [evaluation-bucket-name](#output\_evaluation-bucket-name) | Name of the bucket that can be used to store evaluation results. |
<!-- END_TF_DOCS -->
