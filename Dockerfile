#C:\Users\wisen\youtube_project>docker build -f Dockerfile -t youtube_project .
# docker images  -- to see how many images i have 
#docker tag youtube_project your_dockerhub_username/youtube_project
#docker login   if you are not logged in 
#docker push your_dockerhub_username/youtube_project

FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    mlflow \
    boto3 \
    psycopg2-binary

EXPOSE 5000

#### There is a process change from mlflow check website that is why i hardcorded
##### the allowed hosts to the public IP address , I will change this later 
##### please see website changes https://mlflow.org/docs/latest/self-hosting/security/network/
####  i  got such an error if the website changes were not implemented 
### Invalid Host header - possible DNS rebinding attack detected

CMD mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --allowed-hosts "54.193.111.158:5000" \
    --cors-allowed-origins "*" \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root s3://rentpredictionyusuf/mlflow-artifacts


