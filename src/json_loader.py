from src.model.functions_definition import FunctionDefinition
from src.model.prompt import PromptCase
import json

def load_function_definitions(path: str):
	try:
		with open(path, 'r', encoding="utf-8") as f:
			data = json.load(f)
		return [FunctionDefinition(**item) for item in data]
	except FileNotFoundError:
		raise RuntimeError(f"File not found: {path}")
	except json.JSONDecodeError:
		raise RuntimeError(f"Invalid JSON file in {path}")

def load_test_prompts(path: str):
	try:
		with open(path, 'r', encoding="utf-8") as f:
			data = json.load(f)
		return [PromptCase(**item) for item in data]
	except FileNotFoundError:
		raise RuntimeError(f"File not fount: {path}")
	except json.JSONDecodeError:
		raise RuntimeError(f"Invalid JSON file in {path}")