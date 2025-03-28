FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04
RUN apt-get update && apt-get install -y python3-pip
RUN pip3 install torch sentence-transformers
COPY score.py /app/score.py
WORKDIR /app
CMD ["python3", "score.py"]