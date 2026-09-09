"""LLM agent layer on top of the poe2craft solver: knowledge base (RAG), live
prices, trade search, and Claude tool definitions."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_DIR = os.path.join(ROOT, "cache")
DATA_DIR = os.path.join(ROOT, "data")
KB_PATH = os.path.join(DATA_DIR, "kb.sqlite")
USER_AGENT = "poe2craft-agent/0.1 (personal crafting assistant; contact: kholkinkbauman@gmail.com)"
DEFAULT_LEAGUE = os.environ.get("POE2_LEAGUE", "Runes of Aldur")
