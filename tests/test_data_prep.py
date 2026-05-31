import json
import os
from unittest.mock import patch, MagicMock


def test_extract_training_data_returns_text_samples():
    from data_prep import extract_training_data
    with patch("data_prep.psycopg2.connect") as mock_conn_cls:
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.__enter__ = MagicMock(return_value=mock_cur)
        mock_cur.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cur
        mock_conn_cls.return_value = mock_conn
        mock_cur.fetchall.side_effect = [
            [("跑步鞋", "objection_handling",
              [{"role": "assistant", "content": "贵吗？"},
               {"role": "user", "content": "碳板中底，值得！"}])],
            [],
        ]
        samples = extract_training_data()
    assert len(samples) >= 1
    assert "text" in samples[0]
    assert "Human:" in samples[0]["text"]
    assert "Assistant:" in samples[0]["text"]


def test_save_training_data_writes_jsonl(tmp_path):
    from data_prep import save_training_data
    samples = [
        {"text": "Human: 退货？\n\nAssistant: 7天内可退。"},
        {"text": "Human: 物流？\n\nAssistant: 3-5天到。"},
    ]
    output = str(tmp_path / "train.jsonl")
    count = save_training_data(samples, output)
    assert count == 2
    with open(output, encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["text"] == samples[0]["text"]


def test_format_is_mlx_compatible():
    from data_prep import extract_training_data
    with patch("data_prep.psycopg2.connect") as mock_conn_cls:
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.__enter__ = MagicMock(return_value=mock_cur)
        mock_cur.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cur
        mock_conn_cls.return_value = mock_conn
        mock_cur.fetchall.side_effect = [
            [("商品A", "product_intro",
              [{"role": "assistant", "content": "想了解吗？"},
               {"role": "user", "content": "有什么特点？"}])],
            [],
        ]
        samples = extract_training_data()
    for s in samples:
        assert isinstance(s["text"], str)
        assert "\n\nAssistant:" in s["text"]
