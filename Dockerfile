FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt && \
    pip install dvc[s3]

COPY . .

CMD ["python", "-m", "pytest", "src/tests/"]