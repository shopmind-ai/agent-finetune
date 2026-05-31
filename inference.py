import os
from config import FINETUNED_MODEL_PATH, get_llm, _ECOMMERCE_SYSTEM_PROMPT
from utils.usage import from_response, zero


def mlx_load(model_path: str):
    from mlx_lm import load
    return load(model_path)


def mlx_generate(model, tokenizer, prompt: str, max_tokens: int = 256) -> str:
    from mlx_lm import generate
    return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens)


def generate_base(prompt: str) -> dict:
    """Generate with base Claude — no e-commerce specialization."""
    from langchain_core.messages import HumanMessage
    llm = get_llm()
    response = llm.invoke([HumanMessage(content=prompt)])
    return {
        "response": response.content,
        "model":    "base",
        "usage":    from_response(response),
    }


def generate_with_finetuned(prompt: str) -> dict:
    """Use fine-tuned local model if available, else fall back to Claude + e-commerce prompt."""
    model_path = FINETUNED_MODEL_PATH

    if model_path and os.path.exists(model_path):
        model, tokenizer = mlx_load(model_path)
        text = mlx_generate(
            model, tokenizer,
            prompt=f"Human: {prompt}\n\nAssistant:",
            max_tokens=256,
        )
        return {"response": text, "model": "finetuned", "usage": zero()}
    else:
        from langchain_core.messages import SystemMessage, HumanMessage
        llm = get_llm()
        response = llm.invoke([
            SystemMessage(content=_ECOMMERCE_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])
        return {
            "response": response.content,
            "model":    "base_fallback",
            "usage":    from_response(response),
        }
