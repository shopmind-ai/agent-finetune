from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from utils.startup import check_and_init, log_job

_MLX_COMMAND = (
    "mlx_lm.lora "
    "--model Qwen/Qwen2.5-0.5B-Instruct "
    "--train "
    "--data ./data "
    "--iters 500 "
    "--steps-per-eval 100 "
    "--val-batches 5"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    check_and_init()
    yield


app = FastAPI(title="ShopMind Fine-tune Agent", version="1.0.0", lifespan=lifespan)


class InvokeRequest(BaseModel):
    action: str          # "prepare_data" | "compare" | "generate"
    prompt: str = ""


@app.post("/invoke")
def invoke(req: InvokeRequest):
    if req.action == "prepare_data":
        from data_prep import extract_training_data, save_training_data
        samples = extract_training_data()
        count = save_training_data(samples)
        log_job("prepare_data", data_count=count)
        return {
            "data_count":  count,
            "file_path":   "data/train.jsonl",
            "sample":      samples[:3],
            "mlx_command": _MLX_COMMAND,
            "usage":       {"input_tokens": 0, "output_tokens": 0},
        }

    elif req.action == "compare":
        if not req.prompt:
            return {"error": "prompt is required for compare action"}
        from compare import compare_models
        result = compare_models(req.prompt)
        log_job("compare", prompt=req.prompt,
                base_response=result["base_response"],
                finetuned_response=result["finetuned_response"],
                usage=result["usage"])
        return result

    elif req.action == "generate":
        if not req.prompt:
            return {"error": "prompt is required for generate action"}
        from inference import generate_with_finetuned
        result = generate_with_finetuned(req.prompt)
        log_job("generate", prompt=req.prompt,
                finetuned_response=result["response"], usage=result["usage"])
        return result

    else:
        return {"error": f"Unknown action: {req.action}. Use 'prepare_data', 'compare', or 'generate'."}


@app.get("/health")
def health():
    return {"status": "ok", "service": "agent-finetune"}
