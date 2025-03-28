from sentence_transformers import SentenceTransformer
import torch
import json
import sys

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

def process_input():
    for line in sys.stdin:
        data = json.loads(line)
        messages = data["messages"]
        topics = data["topics"]
        
        # Generate embeddings
        msg_emb = model.encode(messages, convert_to_tensor=True)
        topic_emb = model.encode(topics, convert_to_tensor=True)
        
        # Compute similarity scores
        scores = torch.nn.functional.cosine_similarity(msg_emb.unsqueeze(1), topic_emb.unsqueeze(0), dim=-1)
        probs = torch.softmax(scores, dim=1).tolist()
        
        # Pair each message with its top topic
        results = []
        for msg, prob_list in zip(messages, probs):
            max_prob = max(prob_list)
            top_topic = topics[prob_list.index(max_prob)]
            results.append({
                "message": msg,
                "top_topic": top_topic,
                "probability": max_prob
            })
        
        # Output structured JSON
        print(json.dumps(results, indent=2))
        sys.stdout.flush()

if __name__ == "__main__":
    process_input()