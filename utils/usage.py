from typing import TypedDict

_INPUT_COST_PER_M = 3.0
_OUTPUT_COST_PER_M = 15.0


class TokenUsage(TypedDict):
    input_tokens: int
    output_tokens: int


def from_response(response) -> TokenUsage:
    meta = getattr(response, "usage_metadata", None) or {}
    return {
        "input_tokens":  meta.get("input_tokens", 0),
        "output_tokens": meta.get("output_tokens", 0),
    }


def add(a: TokenUsage, b: TokenUsage) -> TokenUsage:
    return {
        "input_tokens":  a["input_tokens"]  + b["input_tokens"],
        "output_tokens": a["output_tokens"] + b["output_tokens"],
    }


def zero() -> TokenUsage:
    return {"input_tokens": 0, "output_tokens": 0}


def cost_usd(usage: TokenUsage) -> float:
    return (
        usage["input_tokens"]  / 1_000_000 * _INPUT_COST_PER_M +
        usage["output_tokens"] / 1_000_000 * _OUTPUT_COST_PER_M
    )
