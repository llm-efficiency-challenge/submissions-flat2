See [examples](https://github.com/terraform-google-modules/terraform-google-kubernetes-engine/blob/v27.0.0/examples/simple_autopilot_private/main.tf)

<!-- BEGIN_TF_DOCS -->
## Terraform Documentation
Below is the automatically generated Terraform documentation.

### Modules

| Name | Source | Version |
|------|--------|---------|
| <a name="module_gcp-network"></a> [gcp-network](#module\_gcp-network) | terraform-google-modules/network/google | >= 4.0.1 |
| <a name="module_gke"></a> [gke](#module\_gke) | terraform-google-modules/kubernetes-engine/google//modules/beta-autopilot-private-cluster | n/a |
| <a name="module_workload_identity_existing_gsa"></a> [workload\_identity\_existing\_gsa](#module\_workload\_identity\_existing\_gsa) | terraform-google-modules/kubernetes-engine/google//modules/workload-identity | n/a |

### Resources

| Name | Type |
|------|------|
| [google_project_iam_member.artifact-registry-reader](https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/project_iam_member) | resource |
| [google_project_iam_member.log-writer](https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/project_iam_member) | resource |
| [google_project_iam_member.monitoring-metric-writer](https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/project_iam_member) | resource |
| [google_project_iam_member.monitoring-viewer](https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/project_iam_member) | resource |
| [google_project_iam_member.secret-accessor](https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/project_iam_member) | resource |
| [google_project_iam_member.stackdriver-resource-metadata-writer](https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/project_iam_member) | resource |
| [google_service_account.kubernetes](https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/service_account) | resource |
| [google_project.project](https://registry.terraform.io/providers/hashicorp/google/latest/docs/data-sources/project) | data source |

### Inputs

| Name | Description | Type | Default | Required |
|------|-------------|------|---------|:--------:|
| <a name="input_prefix"></a> [prefix](#input\_prefix) | n/a | `string` | n/a | yes |
| <a name="input_region"></a> [region](#input\_region) | The region to deploy the cluster in | `string` | `"europe-west4"` | no |
| <a name="input_zones"></a> [zones](#input\_zones) | The zones the cluster in | `list(string)` | <pre>[<br>  "europe-west4-a"<br>]</pre> | no |

### Outputs

| Name | Description |
|------|-------------|
| <a name="output_ca_certificate"></a> [ca\_certificate](#output\_ca\_certificate) | The cluster ca certificate (base64 encoded) |
| <a name="output_cluster_name"></a> [cluster\_name](#output\_cluster\_name) | Cluster name |
| <a name="output_kubernetes_endpoint"></a> [kubernetes\_endpoint](#output\_kubernetes\_endpoint) | The cluster endpoint |
| <a name="output_location"></a> [location](#output\_location) | n/a |
| <a name="output_master_kubernetes_version"></a> [master\_kubernetes\_version](#output\_master\_kubernetes\_version) | Kubernetes version of the master |
| <a name="output_network_name"></a> [network\_name](#output\_network\_name) | The name of the VPC being created |
| <a name="output_pods_ip_range"></a> [pods\_ip\_range](#output\_pods\_ip\_range) | The IP range in CIDR notation to use for the pods in this cluster. |
| <a name="output_project_id"></a> [project\_id](#output\_project\_id) | The project ID the cluster is in |
| <a name="output_region"></a> [region](#output\_region) | The region in which the cluster resides |
| <a name="output_service_account"></a> [service\_account](#output\_service\_account) | The service account to default running nodes as if not overridden in `node_pools`. |
| <a name="output_service_account_email"></a> [service\_account\_email](#output\_service\_account\_email) | The service account to default running nodes as if not overridden in `node_pools`. |
| <a name="output_subnet_names"></a> [subnet\_names](#output\_subnet\_names) | The names of the subnet being created |
| <a name="output_zones"></a> [zones](#output\_zones) | List of zones in which the cluster resides |
<!-- END_TF_DOCS -->
