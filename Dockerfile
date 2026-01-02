FROM python:3.10-slim

WORKDIR /app
COPY src/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
RUN pip install transformers

COPY src/ .

ENV PYTHONUNBUFFERED=1
CMD ["python", "-u", "app.py"]
