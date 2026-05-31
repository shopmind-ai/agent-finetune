import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    with patch("utils.startup.check_and_init"), \
         patch("utils.startup.log_job"):
        import importlib
        import main as m
        importlib.reload(m)
        yield TestClient(m.app)


def test_prepare_data_action(client):
    with patch("data_prep.extract_training_data", return_value=[
        {"text": "Human: 退货？\n\nAssistant: 7天内可退。"},
        {"text": "Human: 物流？\n\nAssistant: 3-5天。"},
    ]), patch("data_prep.save_training_data", return_value=2):
        resp = client.post("/invoke", json={"action": "prepare_data"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["data_count"] == 2
    assert "mlx_command" in body
    assert "mlx_lm.lora" in body["mlx_command"]


def test_compare_action(client):
    with patch("compare.compare_models", return_value={
        "prompt": "碳板跑鞋？",
        "base_response": "通用回答",
        "finetuned_response": "专业回答",
        "base_model": "base",
        "finetuned_model": "base_fallback",
        "usage": {"input_tokens": 300, "output_tokens": 90},
    }):
        resp = client.post("/invoke", json={"action": "compare", "prompt": "碳板跑鞋？"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["base_response"] == "通用回答"
    assert body["finetuned_response"] == "专业回答"


def test_compare_missing_prompt_returns_error(client):
    resp = client.post("/invoke", json={"action": "compare"})
    assert "error" in resp.json()


def test_generate_action(client):
    with patch("inference.generate_with_finetuned", return_value={
        "response": "专业跑鞋推荐：碳板中底适合竞速。",
        "model": "base_fallback",
        "usage": {"input_tokens": 150, "output_tokens": 60},
    }):
        resp = client.post("/invoke", json={"action": "generate", "prompt": "推荐跑鞋"})
    assert resp.status_code == 200
    assert resp.json()["model"] == "base_fallback"


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["service"] == "agent-finetune"
