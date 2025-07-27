# Use a slim Python base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies and pre-download the model to cache it.
# This is the key step that makes the container work without internet access.
RUN pip install --no-cache-dir -r requirements.txt && \
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy the application code into the container
COPY main.py .

# Set a default command (optional, can be overridden)
CMD ["python", "main.py", "--help"]