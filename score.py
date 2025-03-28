import json
import sys
from transformers import BartTokenizer, BartForSequenceClassification
import torch

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "facebook/bart-large-mnli"  # Model name
model = BartForSequenceClassification.from_pretrained(model_name).to(device)  # Move model to device
tokenizer = BartTokenizer.from_pretrained(model_name)

def process_input():
    for line in sys.stdin:
        data = json.loads(line)
        messages = data["messages"]
        topics = data["topics"]

        # Tokenize input messages and move to device
        inputs = tokenizer(messages, padding=True, truncation=True, return_tensors="pt").to(device)

        # Get model output
        with torch.no_grad():
            logits = model(**inputs).logits

        # Apply softmax to get probabilities
        probs = torch.softmax(logits, dim=-1).tolist()

        # Pair each message with its topics based on probabilities
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
