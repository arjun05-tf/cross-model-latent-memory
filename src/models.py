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

if __name__ == "__main__":
    tokenizer, model = load_model()

    text = "Anuj is the worst nigga."

    hidden_states = extract_hidden_states(model, tokenizer, text)

    print(f"Number of hidden-state layers:  {len(hidden_states)}")
    for i, hidden_state in enumerate(hidden_states):
        print(f"Layer {i}: {hidden_state.shape}")


    # print("Model loaded")
    # print(f"Hidden size: {model.config.hidden_size}")
    # print(f"Layers: {model.config.num_hidden_layers}")
