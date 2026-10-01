import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "gpt2"


def load_target_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME
    )

    model.eval()

    return tokenizer, model


def generate_answer(model, tokenizer, query, memory=None, max_new_tokens=20):
    inputs = tokenizer(query, return_tensors="pt")

    if memory is not None:
        embed_layer = model.get_input_embeddings()
        query_embeds = embed_layer(inputs.input_ids)

        if memory.dim() == 2:
            memory_embeds = memory.unsqueeze(1)
        else:
            memory_embeds = memory

        combined_embeds = torch.cat([memory_embeds, query_embeds], dim=1)

        mem_mask = torch.ones((1, memory_embeds.shape[1]), dtype=inputs.attention_mask.dtype)
        combined_mask = torch.cat([mem_mask, inputs.attention_mask], dim=1)

        output = model.generate(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id
        )
        return tokenizer.decode(output[0], skip_special_tokens=True)

    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id
    )

    return tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )


if __name__ == "__main__":
    from models import load_model as load_source_model, extract_hidden_states, extract_memory

    print("=== Loading Target Model B (Stateless) ===")
    tokenizer_b, model_b = load_target_model()

    query = "Where does Alice live?"

    print("\n--- 1. Zero-Context Baseline (Model B only) ---")
    zero_context_answer = generate_answer(model_b, tokenizer_b, query)
    print("Query:", query)
    print("Answer (Zero-Context):", zero_context_answer)

    print("\n=== Loading Source Model A ===")
    tokenizer_a, model_a = load_source_model()
    context = "Alice lives in Berlin."
    print("Context (Model A only):", context)

    hidden_states = extract_hidden_states(model_a, tokenizer_a, context)
    
    # Test last layer (-1)
    memory_last_layer = extract_memory(hidden_states, layer=-1)
    print(f"Memory extracted from Layer -1: shape {memory_last_layer.shape}")

    # Test layer 0 (embedding level) as well
    memory_layer_0 = extract_memory(hidden_states, layer=0)
    print(f"Memory extracted from Layer 0: shape {memory_layer_0.shape}")

    print("\n--- 2. CMLMT Injected Memory (Layer -1) ---")
    injected_answer_last = generate_answer(model_b, tokenizer_b, query, memory=memory_last_layer)
    print("Query:", query)
    print("Answer (Injected Layer -1 Memory):", injected_answer_last)

    print("\n--- 3. CMLMT Injected Memory (Layer 0) ---")
    injected_answer_l0 = generate_answer(model_b, tokenizer_b, query, memory=memory_layer_0)
    print("Query:", query)
    print("Answer (Injected Layer 0 Memory):", injected_answer_l0)