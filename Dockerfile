FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY deploy/ deploy/

CMD ["python", "deploy/run_worker.py"]
