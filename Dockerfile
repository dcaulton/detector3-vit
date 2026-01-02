FROM python:3.10-slim

WORKDIR /app
COPY src/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install transformers torch torchvision timm

COPY src/ .

ENV PYTHONUNBUFFERED=1
CMD ["python", "-u", "app.py"]
