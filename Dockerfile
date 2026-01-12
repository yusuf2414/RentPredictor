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

CMD mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --serve-artifacts \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root s3://rentpredictionyusuf/models \
    --gunicorn-opts "--timeout 120 --forwarded-allow-ips='*'"