FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Download model from Hugging Face
RUN python -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download(repo_id='Paras-tripathi/deepfake-detector', \
    filename='best_model.pth', local_dir='models')"

# Expose port
EXPOSE 7860

# Run FastAPI
CMD ["uvicorn", "app.fastapi_app:app", "--host", "0.0.0.0", "--port", "7860"]