FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

COPY requirements-retrieval.txt requirements-visual.txt requirements-api.txt ./
RUN python -m pip install --no-cache-dir -r requirements-api.txt

COPY src ./src
COPY scripts/launch_api.py ./scripts/launch_api.py
COPY data/manifests ./data/manifests

EXPOSE 8000

CMD ["python", "scripts/launch_api.py", "--host", "0.0.0.0", "--port", "8000"]
