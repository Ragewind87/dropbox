import json
import sys
from transformers import BartTokenizer, BartForSequenceClassification
import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "facebook/bart-large-mnli"
model = BartForSequenceClassification.from_pretrained(model_name).to(device)
tokenizer = BartTokenizer.from_pretrained(model_name)

def process_input():
    start_time = time.time()

    results = []
    raw_data = "".join(sys.stdin.read())
    data = json.loads(raw_data)
    messages = data["messages"]
    topics = data["topics"]

    for msg in messages:
        topic_probs = {}
        for topic in topics:
            input_text = f"{msg} entails {topic}"
            inputs = tokenizer(input_text, padding=True, truncation=True, return_tensors="pt").to(device)
            with torch.no_grad():
                logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            entailment_prob = probs[0][2].item()
            topic_probs[topic] = entailment_prob

        sorted_topics = sorted(topic_probs.items(), key=lambda x: x[1], reverse=True)[:3]
        top_topic_probs = {topic: round(prob, 2) for topic, prob in sorted_topics}

        results.append({
            "message": msg,
            "topic_probs": top_topic_probs
        })

    total_time = time.time() - start_time
    output = {
        "results": results,
        "total_time_seconds": round(total_time, 2),
        "total_messages_processed": len(messages),
        "total_topics_processed": len(topics)
    }
    print(json.dumps(output, indent=2))
    sys.stdout.flush()

if __name__ == "__main__":
    process_input()