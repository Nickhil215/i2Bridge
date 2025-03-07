# Use the official Python 3.11 slim base image
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

# Copy the requirements.txt file explicitly
COPY requirements.txt /app/requirements.txt

# Install dependencies from requirements.txt
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# To generate tarfile and place it in a folder
RUN python3 setup.py sdist

# Expose port 8080
EXPOSE 8080

# Run the FastAPI application with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]