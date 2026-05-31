import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
DATABASE_URL = os.getenv("DATABASE_URL", "")
FINETUNED_MODEL_PATH = os.getenv("FINETUNED_MODEL_PATH", "")

_ECOMMERCE_SYSTEM_PROMPT = """你是 ShopMind 电商平台的专业助手，专注于运动户外品类（跑步装备、健身器材、户外露营）。
回答要专业具体，结合产品特点、使用场景和实际需求，语言简洁实用。"""


def get_llm():
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(model=CLAUDE_MODEL, api_key=ANTHROPIC_API_KEY)
