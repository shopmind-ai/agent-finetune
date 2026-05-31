from unittest.mock import patch, MagicMock


def test_generate_base_uses_claude():
    from inference import generate_base
    mock_resp = MagicMock()
    mock_resp.content = "跑步鞋要考虑支撑性和缓震。"
    mock_resp.usage_metadata = {"input_tokens": 100, "output_tokens": 40}
    with patch("inference.get_llm") as mock_llm:
        mock_llm.return_value.invoke.return_value = mock_resp
        result = generate_base("如何选跑步鞋？")
    assert result["model"] == "base"
    assert "跑步鞋" in result["response"]
    assert result["usage"]["input_tokens"] == 100


def test_generate_finetuned_fallback_when_no_model():
    from inference import generate_with_finetuned
    mock_resp = MagicMock()
    mock_resp.content = "ShopMind 专业回答：碳板中底最佳选择。"
    mock_resp.usage_metadata = {"input_tokens": 150, "output_tokens": 50}
    with patch("inference.FINETUNED_MODEL_PATH", ""), \
         patch("inference.get_llm") as mock_llm:
        mock_llm.return_value.invoke.return_value = mock_resp
        result = generate_with_finetuned("如何选跑步鞋？")
    assert result["model"] == "base_fallback"
    assert result["usage"]["input_tokens"] == 150


def test_generate_finetuned_uses_mlx_when_model_exists():
    from inference import generate_with_finetuned
    with patch("inference.FINETUNED_MODEL_PATH", "/fake/model/path"), \
         patch("inference.os.path.exists", return_value=True), \
         patch("inference.mlx_generate") as mock_gen, \
         patch("inference.mlx_load") as mock_load:
        mock_load.return_value = (MagicMock(), MagicMock())
        mock_gen.return_value = "MLX 专业回答：选碳板跑鞋。"
        result = generate_with_finetuned("如何选跑步鞋？")
    assert result["model"] == "finetuned"
    assert "MLX" in result["response"]
    assert result["usage"]["input_tokens"] == 0
