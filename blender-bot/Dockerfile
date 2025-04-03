FROM python:3.8-slim

# Install dependencies
RUN pip install torch transformers

# Copy your script
COPY generate_chats.py /app/generate_chats.py
WORKDIR /app

# Default command
CMD ["python3", "generate_chats.py"]