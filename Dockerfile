FROM python:3.12-slim-bookworm

# Install required system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

# Download and install UV (optional, but keeping it since you included it)
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the PATH
ENV PATH="/root/.local/bin/:$PATH"

# Set working directory
WORKDIR /app

RUN mkdir -p /data

# Copy application file
COPY app.py /app

RUN pip install --upgrade pip

# Install FastAPI and Uvicorn directly
RUN pip install --no-cache-dir fastapi uvicorn pydantic openai python-dotenv requests numpy pandas datetime python-dateutil sentence-transformers beautifulsoup4 markdown pillow gitpython
# Expose the FastAPI port
EXPOSE 8000

# Run FastAPI using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
