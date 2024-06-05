FROM --platform=linux/amd64 python:3.8-slim-bullseye
RUN pip install --no-cache-dir \
    mlflow-skinny==2.5.0 \
    psycopg2-binary==2.9.6 \
    google-cloud==0.34.0 \
    google-cloud-storage==2.10.0
ENTRYPOINT [ "mlflow" ]
