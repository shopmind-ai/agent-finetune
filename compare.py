from inference import generate_base, generate_with_finetuned
from utils.usage import add


def compare_models(prompt: str) -> dict:
    """Run the same prompt through both base and fine-tuned models, return side-by-side."""
    base = generate_base(prompt)
    finetuned = generate_with_finetuned(prompt)
    return {
        "prompt":             prompt,
        "base_response":      base["response"],
        "finetuned_response": finetuned["response"],
        "base_model":         base["model"],
        "finetuned_model":    finetuned["model"],
        "usage":              add(base["usage"], finetuned["usage"]),
    }
