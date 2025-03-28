import json
import sys
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

model_name = "distilbert-base-uncased"  # Replace with DistilBART model
model = DistilBertForSequenceClassification.from_pretrained(model_name)
tokenizer = DistilBertTokenizer.from_pretrained(model_name)

def process_input():
    for line in sys.stdin:
        data = json.loads(line)
        messages = data["messages"]
        topics = data["topics"]
        
        inputs = tokenizer(messages, padding=True, truncation=True, return_tensors="pt")

        # Get model output
        with torch.no_grad():
            logits = model(**inputs).logits

        # Apply softmax to get probabilities
        probs = torch.softmax(logits, dim=-1).tolist()

        # Pair each message with its topics
        results = []
        for msg, prob_list in zip(messages, probs):
            selected_topics = [
                topics[i] for i, prob in enumerate(prob_list) if prob >= 0.3
            ]
            selected_probs = [
                prob_list[i] for i, prob in enumerate(prob_list) if prob >= 0.3
            ]
            
            results.append({
                "message": msg,
                "topic_probs": {selected_topics[i]: selected_probs[i] for i in range(len(selected_topics))}
            })
        
        # Output structured JSON
        print(json.dumps(results, indent=2))
        sys.stdout.flush()

if __name__ == "__main__":
    process_input()
