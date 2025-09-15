# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables to prevent Python from writing .pyc files
# and to ensure output is sent straight to the terminal.
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies. This is the crucial step.
# We install ffmpeg for audio processing and libmagic1 for file type detection.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Run collectstatic to gather all static files for production serving
RUN python manage.py collectstatic --no-input

# The port the container will listen on (gunicorn will bind to this)
EXPOSE 8000