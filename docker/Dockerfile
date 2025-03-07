# Use the official Python 3.9 slim base image
FROM --platform=linux/amd64 python:3.11-slim

# Set environment variables to non-interactive to avoid prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install required packages
RUN apt-get update && \
    apt-get install -y curl jq git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the application files
COPY . /app

# Create and activate a virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies from requirements.txt
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install nltk && \
    pip install openai==0.28 && \
    python -m nltk.downloader punkt
    
# to generate tarfile and place in a folder
RUN python3 setup.py sdist
# Expose port 8080
EXPOSE 8080

# Run the FastAPI application with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
