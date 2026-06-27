from src.model.functions_definition import FunctionDefinition
from src.model.prompt import Prompt
import json

def load_function_definition(path: str):
	try:
		with open(path, 'r', encoding="utf-8") as f:
			data = json.load(f)
		return [FunctionDefinition(**item) for item in data]
	except FileNotFoundError:
		raise RuntimeError(f"File not found: {path}")
	except json.JSONDecodeError:
		raise RuntimeError(f"Invalid JSON file in {path}")

def load_prompt(path: str):
	try:
		with open(path, 'r', encoding="utf-8") as f:
			data = json.load(f)
		return [Prompt(**item) for item in data]
	except FileNotFoundError:
		raise RuntimeError(f"File not fount: {path}")
	except json.JSONDecodeError:
		raise RuntimeError(f"Invalid JSON file in {path}")