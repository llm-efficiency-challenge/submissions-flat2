sudo apt-get update
sudo apt-get install -y python3-pip
/usr/bin/pip3 install mlflow==2.5.0 psycopg2-binary==2.9.6 google-cloud==0.34.0 google-cloud-storage==2.10.0 pg8000==1.29.8

mlflow server \
    --host=0.0.0.0 \
    --port=5000 \
    --default-artifact-root=${artifact_bucket_name} \
    --backend-store-uri=${backend_database_uri} #\
    #--app-name basic-auth
