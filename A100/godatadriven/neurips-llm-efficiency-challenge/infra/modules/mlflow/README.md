<!-- BEGIN_TF_DOCS -->
## Terraform Documentation
Below is the automatically generated Terraform documentation.



### Resources

| Name | Type |
|------|------|
| [google_compute_instance.compute](https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/compute_instance) | resource |
| [google_project_iam_member.sql_admin](https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/project_iam_member) | resource |
| [google_service_account.mlflow](https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/service_account) | resource |
| [google_sql_database.mlflow](https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/sql_database) | resource |
| [google_storage_bucket.mlflow](https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/storage_bucket) | resource |
| [google_storage_bucket_iam_member.member](https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/storage_bucket_iam_member) | resource |
| [google_project.project](https://registry.terraform.io/providers/hashicorp/google/latest/docs/data-sources/project) | data source |

### Inputs

| Name | Description | Type | Default | Required |
|------|-------------|------|---------|:--------:|
| <a name="input_extra_network_tags"></a> [extra\_network\_tags](#input\_extra\_network\_tags) | Additional network tags to add to the VM | `list(string)` | `[]` | no |
| <a name="input_machine_type"></a> [machine\_type](#input\_machine\_type) | Machine type to use for the VM | `string` | `"e2-small"` | no |
| <a name="input_prefix"></a> [prefix](#input\_prefix) | Prefix for all resources | `string` | n/a | yes |
| <a name="input_region"></a> [region](#input\_region) | Region to deploy the storage bucket in | `string` | n/a | yes |
| <a name="input_sql_connection_string"></a> [sql\_connection\_string](#input\_sql\_connection\_string) | Connection string to the SQL instance | `string` | n/a | yes |
| <a name="input_sql_instance_name"></a> [sql\_instance\_name](#input\_sql\_instance\_name) | Name of the SQL instance in which the database will be created | `string` | n/a | yes |
| <a name="input_static_ip"></a> [static\_ip](#input\_static\_ip) | Static IP address to assign to the VM | `string` | n/a | yes |
| <a name="input_zone"></a> [zone](#input\_zone) | Zone to deploy the VM in | `string` | `"europe-west4-a"` | no |

### Outputs

| Name | Description |
|------|-------------|
| <a name="output_mlflow-artifact-bucket-name"></a> [mlflow-artifact-bucket-name](#output\_mlflow-artifact-bucket-name) | Name of the bucket to use for MLflow artifacts. |
| <a name="output_mlflow_service_account_mail"></a> [mlflow\_service\_account\_mail](#output\_mlflow\_service\_account\_mail) | Email of the service account to use for MLflow |
<!-- END_TF_DOCS -->
