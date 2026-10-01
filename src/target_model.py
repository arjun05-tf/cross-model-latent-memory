from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "gpt2"


def load_target_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME
    )

    model.eval()

    return tokenizer, model


def generate_answer(model, tokenizer, query):
    inputs = tokenizer(query, return_tensors="pt")

    output = model.generate(
        **inputs,
        max_new_tokens=20
    )

    return tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )


if __name__ == "__main__":
    tokenizer, model = load_target_model()

    query = "Where does Alice live?"

    answer = generate_answer(
        model,
        tokenizer,
        query
    )

    print("Query:", query)
    print("Answer:", answer)