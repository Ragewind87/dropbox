FROM python:3.8-slim

# Install dependencies
RUN pip install torch==2.0.1 transformers==4.36.2 accelerate==0.25.0

# Copy script
COPY generate_chats.py /app/generate_chats_mixtral.py

# Set working dir
WORKDIR /app

# Run with unbuffered output
CMD ["python3", "-u", "generate_chats_mixtral.py"]