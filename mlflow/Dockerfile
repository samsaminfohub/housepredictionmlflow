FROM python:3.9-slim

RUN pip install mlflow==2.5.0 psycopg2-binary

EXPOSE 5000

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--default-artifact-root", "/mlflow/artifacts"]