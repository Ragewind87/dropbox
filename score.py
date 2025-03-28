from sentence_transformers import SentenceTransformer
import torch
import json
import sys

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2').to('cuda')

def process_input():
    for line in sys.stdin:
        data = json.loads(line)
        messages = data["messages"]
        topics = data["topics"]
        msg_emb = model.encode(messages, convert_to_tensor=True, device='cuda')
        topic_emb = model.encode(topics, convert_to_tensor=True, device='cuda')
        scores = torch.nn.functional.cosine_similarity(msg_emb.unsqueeze(1), topic_emb.unsqueeze(0), dim=-1)
        probs = torch.softmax(scores, dim=1).tolist()
        print(json.dumps(probs))
        sys.stdout.flush()

if __name__ == "__main__":
    process_input()