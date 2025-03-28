import json
import sys
from transformers import pipeline

# Load zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def process_input():
    for line in sys.stdin:
        data = json.loads(line)
        messages = data["messages"]  # List of messages
        topics = data["topics"]      # List of topic labels

        results = []
        for msg in messages:
            classification = classifier(msg, topics, multi_label=True)
            scores = classification["scores"]
            labels = classification["labels"]

            # Filter only topics with a probability >= 0.3
            topic_probs = {labels[i]: scores[i] for i in range(len(labels)) if scores[i] >= 0.3}

            results.append({
                "message": msg,
                "topic_probs": topic_probs
            })

        # Output structured JSON
        print(json.dumps(results, indent=2))
        sys.stdout.flush()

if __name__ == "__main__":
    process_input()
