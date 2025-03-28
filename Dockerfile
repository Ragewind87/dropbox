# Use a base image with CUDA for GPU acceleration
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Install Python and pip
RUN apt-get update && apt-get install -y python3-pip

# Install PyTorch and Hugging Face Transformers
RUN pip3 install torch transformers

# Copy your Python script into the container
COPY score.py /app/score.py

# Set working directory
WORKDIR /app

# Command to run the model inference
CMD ["python3", "score.py"]
