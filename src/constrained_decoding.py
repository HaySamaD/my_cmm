from llm_sdk import Small_LLM_Model
import json
from typing import Set, Any


def get_approve_valid_token(logits: Any, valid_ids: Set[int]) -> int:
    """Returns the valid token ID that maximizes the logits value."""
    return max(valid_ids, key=lambda i: logits[i])


def extract_complete_json(text: str) -> Any:
    """Extracts the first matched valid balancing closure of {...}."""
    start = text.find("{")
    if start == -1:
        return None

    count = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            count += 1
        if text[i] == "}":
            count -= 1
        if count == 0:
            return text[start:i+1]

    return None


def build_json_valid_ids(vocab: dict) -> Set[int]:
    """Filters vocabulary to only allow secure structural JSON characters."""
    safe_json = set(
        'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        '0123456789*_.,:-+/\'!?()[]{}"ĠĊ'
    )
    valid = set()
    for token_str, token_id in vocab.items():
        if token_str and all(c in safe_json for c in token_str):
            valid.add(token_id)
    return valid


def load_vocabulary(model: Small_LLM_Model) -> Any:
    vocab_path = model.get_path_to_tokenizer_file()
    with open(vocab_path, 'r', encoding="utf-8") as f:
        toke_data = json.load(f)
    raw_vocab = toke_data.get("model", {}).get("vocab", {})
    return raw_vocab


def build_system_prompt(functions: Any) -> str:
    lines = [
        "STRICT SYSTEM RULE: Use ONLY a matching "
        "function from the list below.",
        "If NO function matches the user's inten (even if types match), "
        "set name: \"none\".",
        "Never use an unrelated function for a different task.",
        "",
        "Available functions:",

    ]
    for fn in functions:
        param = ", ".join(
            f"{name}: {info.type}"
            for name, info in fn.parameters.items()
        )
        lines.append(f"  -{fn.name}({param}): {fn.description}")

    lines.append('\nOutput ONLY valid JSON: {"name": "<fn>", "args: {<args>}}')

    return "\n".join(lines)
