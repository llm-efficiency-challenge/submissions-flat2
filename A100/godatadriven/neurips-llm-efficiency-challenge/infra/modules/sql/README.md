<!-- BEGIN_TF_DOCS -->
## Terraform Documentation
Below is the automatically generated Terraform documentation.



### Resources

| Name | Type |
|------|------|
| [google_secret_manager_secret.credentials](https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/secret_manager_secret) | resource |
| [google_secret_manager_secret_version.credentials](https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/secret_manager_secret_version) | resource |
| [google_sql_database_instance.main](https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/sql_database_instance) | resource |
| [google_sql_user.user](https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/sql_user) | resource |
| [random_password.password](https://registry.terraform.io/providers/hashicorp/random/latest/docs/resources/password) | resource |
| [random_pet.database](https://registry.terraform.io/providers/hashicorp/random/latest/docs/resources/pet) | resource |

### Inputs

| Name | Description | Type | Default | Required |
|------|-------------|------|---------|:--------:|
| <a name="input_prefix"></a> [prefix](#input\_prefix) | Prefix for all resources | `string` | n/a | yes |
| <a name="input_region"></a> [region](#input\_region) | Region to deploy the SQL server in | `string` | `"europe-west4"` | no |
| <a name="input_username"></a> [username](#input\_username) | SQL username | `string` | n/a | yes |
| <a name="input_whitelist_ips"></a> [whitelist\_ips](#input\_whitelist\_ips) | List of IP addresses to whitelist | `list(string)` | n/a | yes |

### Outputs

| Name | Description |
|------|-------------|
| <a name="output_sql-instance-name"></a> [sql-instance-name](#output\_sql-instance-name) | Name of the postgres SQL instance |
| <a name="output_sql-output-ip"></a> [sql-output-ip](#output\_sql-output-ip) | IP address of the postgres SQL instance |
<!-- END_TF_DOCS -->
