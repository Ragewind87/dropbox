FROM python:3.8-slim

RUN pip install torch==2.0.1 transformers==4.36.2 accelerate==0.25.0

ARG HF_TOKEN
ENV HF_TOKEN=$HF_TOKEN

# Pre-download with explicit torch import and fresh download
RUN python3 -c "import torch; from transformers import AutoTokenizer, AutoModelForCausalLM; \
    AutoTokenizer.from_pretrained('mistralai/Mixtral-8x7B-v0.1', token='$HF_TOKEN', force_download=True); \
    AutoModelForCausalLM.from_pretrained('mistralai/Mixtral-8x7B-v0.1', torch_dtype=torch.float16, token='$HF_TOKEN', force_download=True)"

COPY generate_chats_mixtral.py /app/generate_chats_mixtral.py
WORKDIR /app
CMD ["python3", "-u", "generate_chats_mixtral.py"]