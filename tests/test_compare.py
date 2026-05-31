from unittest.mock import patch


def test_compare_models_returns_both_responses():
    from compare import compare_models
    with patch("compare.generate_base") as mock_base, \
         patch("compare.generate_with_finetuned") as mock_ft:
        mock_base.return_value = {
            "response": "通用跑步鞋建议：选合适尺码。",
            "model": "base",
            "usage": {"input_tokens": 100, "output_tokens": 30}
        }
        mock_ft.return_value = {
            "response": "专业建议：碳板跑鞋适合竞速，缓震跑鞋适合日训。",
            "model": "base_fallback",
            "usage": {"input_tokens": 150, "output_tokens": 60}
        }
        result = compare_models("如何选跑步鞋？")
    assert result["prompt"] == "如何选跑步鞋？"
    assert "通用" in result["base_response"]
    assert "专业" in result["finetuned_response"]
    assert result["base_model"] == "base"
    assert result["finetuned_model"] == "base_fallback"


def test_compare_accumulates_usage():
    from compare import compare_models
    with patch("compare.generate_base") as mock_base, \
         patch("compare.generate_with_finetuned") as mock_ft:
        mock_base.return_value = {"response": "A", "model": "base",
                                   "usage": {"input_tokens": 100, "output_tokens": 30}}
        mock_ft.return_value = {"response": "B", "model": "finetuned",
                                 "usage": {"input_tokens": 200, "output_tokens": 60}}
        result = compare_models("问题？")
    assert result["usage"]["input_tokens"] == 300
    assert result["usage"]["output_tokens"] == 90
