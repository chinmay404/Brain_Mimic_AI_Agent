import os
from typing import Optional

try:
	from langchain_groq import ChatGroq
except Exception:
	ChatGroq = None


def get_groq_llm(api_key: Optional[str] = None, model: str = "openai/gpt-oss-120b", temperature: float = 0.0):
	"""
	Factory that returns a LangChain-compatible Groq LLM client (ChatGroq).

	Args:
		api_key: Optional explicit API key. If not provided, reads from env `GROQ_API_KEY`.
		model: Model identifier to use with Groq.
		temperature: Sampling temperature.

	Returns:
		An instance of `ChatGroq` (or raises ImportError if provider not installed).

	Usage:
		from utils.llm_provider import get_groq_llm
		llm = get_groq_llm()
	"""
	if ChatGroq is None:
		raise ImportError("langchain_groq.ChatGroq is not available. Install the provider package.")

	key = api_key or os.getenv("GROQ_API_KEY")
	if not key:
		raise ValueError("GROQ_API_KEY not provided and no api_key argument supplied")

	llm = ChatGroq(model=model, temperature=temperature, api_key=key)
	return llm


__all__ = ["get_groq_llm"]

