from llm_sdk import Small_LLM_Model
import json
import numpy as np
from typing import Set, Any


def get_approve_valid_token(logits: Any,
                            current_text: str,
                            vocab: dict
                            ) -> int:
    """
    Updated dynamic structural status monitor:
      - Prevents the formatting of numbers inside texts
        (by filtering out quotation marks when expecting numerical values).
      - The form allows writing whole numbers
        (such as 16 and 144) without restricting them to the number 1.
      - Fully supports the Arabic language within free texts.
    """
    logits_array = np.array(logits, dtype=np.float32)
    masked_logits = np.full_like(logits_array, -float('inf'))

    unescaped_quotes = current_text.count('"') - current_text.count('\\"')
    is_inside_string = (unescaped_quotes % 2 == 1)

    valid_ids = []

    if is_inside_string:
        for token_str, token_id in vocab.items():
            if not token_str:
                continue
            if '\n' in token_str or '\r' in token_str:
                continue
            if token_str.startswith('"') and current_text.endswith('"'):
                continue
            valid_ids.append(token_id)
    else:
        stripped = current_text.strip()

        if not current_text:
            for token_str, token_id in vocab.items():
                if token_str and token_str.lstrip().startswith('{'):
                    valid_ids.append(token_id)

        elif '"name"' not in stripped:
            for token_str, token_id in vocab.items():
                if token_str and any(c in 'name": ' for c in token_str):
                    valid_ids.append(token_id)

        elif stripped.endswith('{"name":'):
            for token_str, token_id in vocab.items():
                if token_str and '"' in token_str:
                    valid_ids.append(token_id)

        elif '"name"' in stripped and '"args"' not in stripped:
            for token_str, token_id in vocab.items():
                if not token_str or '""' in token_str:
                    continue
                has_valid_char = any(
                    c in ',"args: ' for c in token_str) or '{' in token_str
                if has_valid_char:
                    valid_ids.append(token_id)

        elif stripped.endswith(':'):
            is_numeric_field = (
                "add_numbers" in stripped or "square_root" in stripped)

            for token_str, token_id in vocab.items():
                if not token_str:
                    continue
                if is_numeric_field and '"' in token_str:
                    continue
                if any(c in '{"0123456789-. ' for c in token_str):
                    valid_ids.append(token_id)

        elif stripped.endswith('{') or stripped.endswith(','):
            for token_str, token_id in vocab.items():
                if token_str and ('"' in token_str or
                                  any(c in ' \n\t' for c in token_str)):
                    valid_ids.append(token_id)
        else:
            for token_str, token_id in vocab.items():
                if token_str and any(c in '0123456789.,}} \n\t' for c in token_str):
                    valid_ids.append(token_id)

    if not valid_ids:
        return int(np.argmax(logits_array))

    masked_logits[valid_ids] = logits_array[valid_ids]
    return int(np.argmax(masked_logits))


def build_json_valid_ids(vocab: dict) -> Set[int]:
    return set(vocab.values())


def extract_complete_json(text: str) -> Any:
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
            return text
    return None


def load_vocabulary(model: Small_LLM_Model) -> Any:
    vocab_path = model.get_path_to_tokenizer_file()
    with open(vocab_path, 'r', encoding="utf-8") as f:
        toke_data = json.load(f)
    raw_vocab = toke_data.get("model", {}).get("vocab", {})
    return raw_vocab


def build_system_prompt(functions: Any) -> str:
    lines = [
        "STRICT SYSTEM RULE: Use ONLY a matching function from the list below.",
        "If NO function matches the user's intent (even if types match), set name: \"none\".",
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
    lines.append('\nOutput ONLY valid JSON: {"name": "<fn>", "args": {<args>}}')
    return "\n".join(lines)
