FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY app ./app

ENV PYTHONUNBUFFERED=1 \
    SURYA_INFERENCE_BACKEND=vllm \
    SURYA_INFERENCE_URL=http://surya-vllm:8000/v1 \
    SERVER_PORT=7860 \
    OLLAMA_BASE_URL=http://host.docker.internal:11434 \
    QWEN_MODEL=qwen2.5:3b

EXPOSE 7860

CMD ["python", "-m", "app.main"]
