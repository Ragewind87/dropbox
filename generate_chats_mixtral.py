from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import random

model_name = "mistralai/Mixtral-8x7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

participants = ["Alice", "Bob", "Charlie", "Dave"]
topics = ["sports", "politics", "tech", "movies", "food", "travel", "music", "games", "books", "art"]

def generate_conversation(participants, topics, turns_per_topic=10, num_conversations=25):
    all_conversations = []
    for conv_id in range(num_conversations):
        history = []
        conv_topics = random.sample(topics, len(topics))
        for topic in conv_topics:
            starter = random.choice(participants)
            initial_message = f"{starter}: Let's discuss {topic} today."
            history.append(initial_message)
            current_speaker_idx = participants.index(starter)
            for _ in range(turns_per_topic - 1):
                current_speaker_idx = (current_speaker_idx + 1) % len(participants)
                next_speaker = participants[current_speaker_idx]
                input_text = " ".join(history[-5:])
                inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.9,
                    top_p=0.95,
                    pad_token_id=tokenizer.eos_token_id
                )
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                new_message = f"{next_speaker}: {response}"
                history.append(new_message)
        all_conversations.append(history)
    return all_conversations

conversations = generate_conversation(participants, topics, turns_per_topic=10, num_conversations=25)
with open("/app/mixtral_chat_data.txt", "w") as f:
    for i, conv in enumerate(conversations):
        f.write(f"Conversation {i+1}:\n")
        f.write("\n".join(conv) + "\n\n")