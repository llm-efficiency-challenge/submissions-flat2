# %%

import json
import requests

# %%

r = requests.patch(
    "http://34.90.207.177:5000/api/2.0/mlflow/users/update-password",
    data=json.dumps({"username": "admin", "password": "boogaboo"}),
    auth=("admin", "password"),
    headers={"Content-Type": "application/json"},
)

# python3 -c "import json; import requests; r=requests.patch('http://localhost:5000/api/2.0/mlflow/users/update-password', data=json.dumps({'username': 'admin', 'password': 'snoodeldoodel'}), auth=('admin', 'boogaboo'),
# headers={'Content-Type': 'application/json'}); r.raise_for_status()"

# %%

r.raise_for_status()

# %%
