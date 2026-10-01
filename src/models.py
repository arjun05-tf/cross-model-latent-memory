import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "gpt2"

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        output_hidden_states=True
    )
    model.eval()

    return tokenizer, model

def extract_hidden_states(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    return outputs.hidden_states


def extract_memory(hidden_states, layer=-1):
    hidden_state=hidden_states[layer]

    #hidden_state: [batch, sequence_length, hidden_size]
    memory = hidden_state.mean(dim=1)

    return memory

if __name__ == "__main__":
    tokenizer, model = load_model()

    text = "Alice lives in Berlin."

    hidden_states = extract_hidden_states(model, tokenizer, text)
    memory = extract_memory(hidden_states)

    # print(f"Number of hidden-state layers:  {len(hidden_states)}")
    # for i, hidden_state in enumerate(hidden_states):
    #     print(f"Layer {i}: {hidden_state.shape}")

    print(f"Memory shape: {memory.shape}")
    print("Memory: ", memory)


    # print("Model loaded")
    # print(f"Hidden size: {model.config.hidden_size}")
    # print(f"Layers: {model.config.num_hidden_layers}")
