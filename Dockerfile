# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /code

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies (adding build-essential for some C-extensions if needed)
RUN apt-get update && apt-get install -y build-essential \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get clean

# Copy the rest of your application code
COPY . .

# Expose port 8000 for FastAPI
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]