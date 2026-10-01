"""LangChain agent with local Ollama and optional cloud providers."""

from __future__ import annotations

import ast
import datetime as dt
import operator
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_community.tools.file_management.read import ReadFileTool
from langchain_community.tools.file_management.write import WriteFileTool
from langchain_tavily import TavilySearch
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from ollama import Client, ResponseError


_OPS: dict[type[ast.AST], Any] = {
	ast.Add: operator.add,
	ast.Sub: operator.sub,
	ast.Mult: operator.mul,
	ast.Div: operator.truediv,
	ast.Pow: operator.pow,
	ast.USub: operator.neg,
}

_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
_MAX_FILE_CHARS = 10000
_TOOL_PREVIEW_CHARS = 200


def _output_path(filename: str) -> Path:
	if not filename or "/" in filename or "\\" in filename or ":" in filename:
		raise ValueError("Use a filename without directories")
	if _OUTPUT_DIR.is_symlink():
		raise ValueError("The outputs directory cannot be a symbolic link")
	path = (_OUTPUT_DIR / filename).resolve()
	if path.parent != _OUTPUT_DIR.resolve():
		raise ValueError("File must stay inside the outputs directory")
	return path


def _safe_eval(expression: str) -> float:
	"""Evaluate simple arithmetic expression safely via AST."""

	def _eval(node: ast.AST) -> float:
		if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
			return float(node.value)
		if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
			return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
		if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
			return _OPS[type(node.op)](_eval(node.operand))
		raise ValueError("Only numbers and + - * / ** are allowed")

	parsed = ast.parse(expression, mode="eval")
	return _eval(parsed.body)


@tool
def get_current_time() -> str:
	"""Get the current local datetime in ISO format."""
	return dt.datetime.now().isoformat(timespec="seconds")


@tool
def calculator(expression: str) -> str:
	"""Calculate arithmetic expressions, e.g. '(12+3)*4/5'."""
	try:
		return str(_safe_eval(expression))
	except Exception as exc:  # Keep tool failure message concise for the agent.
		return f"calculation_error: {exc}"


@tool
def read_text_file(filename: str) -> str:
	"""Read a UTF-8 text file of any extension in the project's outputs directory."""
	try:
		_output_path(filename)
		content = ReadFileTool(root_dir=str(_OUTPUT_DIR)).invoke({"file_path": filename})
		if len(content) > _MAX_FILE_CHARS:
			return "Error: file is too large to read"
		return content
	except (OSError, UnicodeError, ValueError) as exc:
		return f"Error: {exc}"


@tool
def create_text_file(filename: str, content: str) -> str:
	"""Create a UTF-8 text file of any extension in outputs; never overwrite a file."""
	try:
		if len(content) > _MAX_FILE_CHARS:
			return "Error: content is too large to save"
		path = _output_path(filename)
		_OUTPUT_DIR.mkdir(exist_ok=True)
		with path.open("x", encoding="utf-8") as output:
			output.write(content)
		return f"Created outputs/{filename}"
	except (OSError, ValueError) as exc:
		return f"Error: {exc}"


@tool
def rewrite_text_file(filename: str, original_content: str, updated_content: str) -> str:
	"""Rewrite a UTF-8 outputs file of any extension if its text matches original_content."""
	try:
		if len(original_content) > _MAX_FILE_CHARS or len(updated_content) > _MAX_FILE_CHARS:
			return "Error: content is too large to save"
		path = _output_path(filename)
		if path.read_text(encoding="utf-8") != original_content:
			return "Error: file changed since it was read; read it again before rewriting"
		result = WriteFileTool(root_dir=str(_OUTPUT_DIR)).invoke(
			{"file_path": filename, "text": updated_content}
		)
		return f"Updated outputs/{filename}" if result == f"File written successfully to {filename}." else result
	except (OSError, UnicodeError, ValueError) as exc:
		return f"Error: {exc}"


@tool
def replace_text_file(filename: str, old_text: str, new_text: str) -> str:
	"""Replace one unique exact snippet in a UTF-8 outputs file of any extension."""
	try:
		if not old_text:
			return "Error: old_text cannot be empty"
		path = _output_path(filename)
		content = path.read_text(encoding="utf-8")
		if len(content) > _MAX_FILE_CHARS:
			return "Error: file is too large to edit"
		if content.count(old_text) != 1:
			return "Error: old_text must occur exactly once; read the file and choose a unique snippet"
		updated = content.replace(old_text, new_text, 1)
		if len(updated) > _MAX_FILE_CHARS:
			return "Error: content is too large to save"
		result = WriteFileTool(root_dir=str(_OUTPUT_DIR)).invoke({"file_path": filename, "text": updated})
		return f"Updated outputs/{filename}" if result == f"File written successfully to {filename}." else result
	except (OSError, UnicodeError, ValueError) as exc:
		return f"Error: {exc}"


def _supports_thinking(model_name: str, base_url: str) -> bool:
	try:
		return "thinking" in (Client(host=base_url).show(model_name).capabilities or [])
	except (ConnectionError, ResponseError):
		return False


def build_agent(*, checkpointer: BaseCheckpointSaver | None = None, reasoning: bool = False):
	load_dotenv()
	provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
	if provider == "deepseek":
		api_key = os.getenv("DEEPSEEK_API_KEY")
		if not api_key:
			raise RuntimeError("DEEPSEEK_API_KEY is not set")
		model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
		model = init_chat_model(
			model=model_name,
			model_provider="openai",
			api_key=api_key,
			base_url="https://api.deepseek.com",
			temperature=0,
		)
	elif provider == "openai":
		api_key = os.getenv("OPENAI_API_KEY")
		if not api_key:
			raise RuntimeError("OPENAI_API_KEY is not set")
		model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
		model = init_chat_model(
			model=model_name,
			model_provider="openai",
			api_key=api_key,
			temperature=0,
		)
	elif provider == "ollama":
		model_name = os.getenv("OLLAMA_MODEL", "qwen3:4b")
		base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
		model = init_chat_model(
			model=model_name,
			model_provider="ollama",
			base_url=base_url,
			temperature=0,
			reasoning=True if reasoning and _supports_thinking(model_name, base_url) else None,
		)
	else:
		raise RuntimeError("LLM_PROVIDER must be 'ollama', 'openai' or 'deepseek'")

	tools = [get_current_time, calculator, read_text_file, create_text_file, rewrite_text_file, replace_text_file]
	if os.getenv("TAVILY_API_KEY"):
		tools.append(TavilySearch(max_results=5, topic="general"))
	provider_name = {"ollama": "Ollama", "deepseek": "DeepSeek", "openai": "OpenAI"}[provider]
	search_instructions = (
		"涉及时事、近期信息或实时数据时，使用网页搜索，并在回答中附上搜索结果的来源链接。"
		"如果搜索工具调用失败，不要声称已经搜索过。"
		if os.getenv("TAVILY_API_KEY")
		else "当前没有网页搜索功能，不要声称已经上网搜索过。"
	)
	return create_agent(
		model=model,
		tools=tools,
		checkpointer=checkpointer,
		system_prompt=(
			"你是一个务实的助手。"
			f"当前配置的模型提供方是 {provider_name}，模型是 {model_name}。"
			"如果用户问你是什么模型，就回答这个配置的模型名称。"
			"你的工具包括本地时钟和基础四则运算计算器。"
			"你只能在 outputs 文件夹中读取、创建和编辑任意后缀的 UTF-8 文本文件。"
			"编辑文件前先读取文件；局部修改使用 replace_text_file，"
			"只有整篇重写时才使用 rewrite_text_file。只有在用户要求时才修改文件。"
			f"{search_instructions}"
			"需要时调用工具，回答要准确、简洁。"
		),
	)


def _stream_reply(agent, user_input: str, config: dict) -> None:
	section = ""
	answered = False

	def emit(label: str, text: str, *, new_line: bool = False) -> None:
		nonlocal section
		if new_line or section != label:
			text = text.lstrip()
			if not text:
				return
			if section:
				print()
			print(label, end="")
			section = label
		print(text, end="", flush=True)

	for message, _ in agent.stream(
		{"messages": [{"role": "user", "content": user_input}]}, config, stream_mode="messages"
	):
		if message.type == "human":
			continue
		if message.type == "tool":
			emit("[工具结果] ", message.text[:_TOOL_PREVIEW_CHARS], new_line=True)
			continue
		reasoning = message.additional_kwargs.get("reasoning_content")
		if reasoning:
			emit("[思考] ", reasoning)
		for call in getattr(message, "tool_call_chunks", None) or message.tool_calls:
			if call.get("name"):
				emit("[调用工具] ", call["name"], new_line=True)
		if message.text.strip():
			answered = True
		emit("Agent> ", message.text)
	if not answered:
		emit("Agent> ", "(no response)", new_line=True)
	print()


def main() -> None:
	agent = build_agent(checkpointer=InMemorySaver(), reasoning=True)
	thread_config = {"configurable": {"thread_id": str(uuid4())}}

	print("LangChain agent started. Type 'new' for a new conversation or 'exit' to quit.")
	while True:
		user_input = input("You> ").strip()
		if not user_input:
			continue
		if user_input.lower() in {"exit", "quit"}:
			print("Bye.")
			break
		if user_input.lower() == "new":
			thread_config = {"configurable": {"thread_id": str(uuid4())}}
			print("New conversation started.")
			continue

		_stream_reply(agent, user_input, thread_config)


if __name__ == "__main__":
	try:
		main()
	except KeyboardInterrupt:
		print("\nInterrupted.")
	except Exception as exc:
		print(f"Startup error: {exc}")