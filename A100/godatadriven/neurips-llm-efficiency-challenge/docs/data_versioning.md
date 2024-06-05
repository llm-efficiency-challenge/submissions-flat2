# Data versioning

We use Data Version Control (DVC) for data versioning.

The DVC dependency is python-based, and hence will be installed when you execute `poetry install`.

## Setting up DVC

Make sure to initialize poetry `poetry shell`

### First-time setup only

1. Initialize DVC `dvc init`
1. Add the remote `dvc remote add -f neurips gs://data-neurips-2023/datasets`
1. Modify the config s.t. bucket objects are version aware `dvc remote modify neurips version_aware true`

### When cloning the repository/setting up DVC for the first time

1. Generate the required credentials file to the ".secrets" folder by executing `cd infra && invoke write-json-key-dvc-storage-admin`
1. Tell DVC where it can find the credentials `dvc remote modify --local neurips credentialpath .secrets/data-bucket-service-account-key.json`
1. Set "neurips" as the default remote `dvc remote default neurips`

## Using DVC

You can find the DVC docs [here](https://dvc.org/doc).

## Useful resources

The following resources are useful for reference purposes:

1. [Using GET to download files from a DVC-tracked repository](https://dvc.org/doc/command-reference/get)
1. [Modifying large datasets](https://dvc.org/doc/user-guide/data-management/modifying-large-datasets)
