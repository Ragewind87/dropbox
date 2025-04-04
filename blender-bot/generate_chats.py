from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import random

model_name = "facebook/blenderbot-400M-distill"  # Or "facebook/bart-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

participants = ["Alice", "Bob", "Charlie"]
topics = ["sports", "politics", "technology"]

def generate_conversation(participants, topics, turns_per_topic=5):
    history = []
    for topic in topics:
        starter = random.choice(participants)
        initial_message = f"{starter}: Let's talk about {topic}."
        history.append(initial_message)
        current_speaker_idx = participants.index(starter)
        for _ in range(turns_per_topic - 1):
            current_speaker_idx = (current_speaker_idx + 1) % len(participants)
            next_speaker = participants[current_speaker_idx]
            input_text = " ".join(history)
            inputs = tokenizer(input_text, return_tensors="pt").to(device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            new_message = f"{next_speaker}: {response}"
            history.append(new_message)
    return history

conversation = generate_conversation(participants, topics)
for message in conversation:
    print(message)