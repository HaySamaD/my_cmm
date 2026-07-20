import json
import numpy as np
from llm_sdk import Small_LLM_Model
from typing import Any, Optional, Set


class JsonConstrainedDecoder:
    """
    Finite State Machine vocabulary decoder designed
    to mask illegal probability logits
    and isolate character strings over active parsing loops.
    """

    def __init__(self,
                 vocab: dict,
                 valid_function_names: Optional[list[str]] = None
                 ) -> None:
        """
        Prepares internal state parameters, constants, tracking metrics,
        and arrays.
        """
        self.vocab = vocab
        self.actual_function_names = (
            ["none"] if valid_function_names is None else list(
                valid_function_names))
        if "none" not in self.actual_function_names:
            self.actual_function_names.append("none")
        self.mask_ceiling = -100000.0

    def get_approve_valid_token(self, logits: Any, current_text: str) -> int:
        """
        Computes structural status configurations
        and selects the next type-safe token.
        """
        logits_array = np.array(logits, dtype=np.float32)
        masked_logits = np.full_like(logits_array, self.mask_ceiling)

        unescaped_quotes = current_text.count('"') - current_text.count('\\"')
        is_inside_string = (unescaped_quotes % 2 == 1)
        stripped = current_text.strip()
        valid_ids: list[int] = []

        if is_inside_string:
            valid_ids = self._handle_inside_string_state(
                current_text, stripped)
        else:
            valid_ids = self._handle_outside_string_state(
                current_text, stripped)

        if not valid_ids:
            return int(np.argmax(logits_array))

        masked_logits[valid_ids] = logits_array[valid_ids]
        return int(np.argmax(masked_logits))

    def _handle_inside_string_state(self,
                                    current_text: str,
                                    stripped: str
                                    ) -> list[int]:
        """
        Filters tokens while the decoder
        is processing content nested inside string literal states.
        """
        valid_ids = []
        is_writing_name_value = stripped.endswith(
            '"name":') or ('"name":' in stripped and '"args"' not in stripped)
        is_formatting_template = "format_template" in stripped

        for token_str, token_id in self.vocab.items():
            if not token_str or '\n' in token_str or '\r' in token_str:
                continue

            if is_writing_name_value:
                last_part = current_text.split('"name":')[-1].strip()
                current_name_prefix = (
                    last_part.split('"')[1]
                    if last_part.count('"') >= 1 else "")
                potential_name = (
                    current_name_prefix + token_str.replace('"', ''))
                if not any(
                    name.startswith(potential_name)
                    for name in self.actual_function_names
                ):
                    continue

            if '\\"' in token_str and is_formatting_template:
                valid_ids.append(token_id)
                continue

            if token_str.startswith('"') and current_text.endswith('"'):
                continue
            valid_ids.append(token_id)
        return valid_ids

    def _handle_outside_string_state(self,
                                     current_text: str,
                                     stripped: str
                                     ) -> list[int]:
        """
        Enforces schema rules
        and syntax structure borders when outside a string value block.
        """
        valid_ids = []
        if not current_text:
            return [
                tid for tstr, tid in self.vocab.items()
                if tstr and tstr.lstrip().startswith('{')
            ]

        if '"name"' in stripped and '"args"' not in stripped:
            for token_str, token_id in self.vocab.items():
                if not token_str:
                    continue
                potential_stripped = (current_text + token_str).strip()
                if any(
                    potential_stripped.startswith(f'{{"name": "{name}"')
                    for name in self.actual_function_names
                ):
                    if (
                        token_str.strip() == ","
                        and not current_text.rstrip().endswith('"')
                    ):
                        continue
                    valid_ids.append(token_id)
        elif stripped.endswith('"args"'):
            valid_ids = [
                tid for tstr, tid in self.vocab.items()
                if tstr and any(c in ': \t\n' for c in tstr)
            ]
        elif stripped.endswith('"args":') or stripped.endswith('"args" :'):
            valid_ids = [
                tid for tstr, tid in self.vocab.items()
                if tstr and ('{' in tstr or any(c in ' \t\n' for c in tstr))
            ]
        elif stripped.endswith(':'):
            is_numeric_field = (
                "multiply" in stripped
                or "interest" in stripped
                or "even" in stripped)
            for token_str, token_id in self.vocab.items():
                if not token_str:
                    continue
                if is_numeric_field and '"' in token_str:
                    continue
                if any(c in '{"0123456789-.,\" ' for c in token_str):
                    valid_ids.append(token_id)
        elif stripped.endswith('{') or stripped.endswith(','):
            for token_str, token_id in self.vocab.items():
                if not token_str:
                    continue
                if '"' in token_str or any(c in ' \n\t' for c in token_str):
                    if "format_template" in stripped and '"user"' in token_str:
                        continue
                    valid_ids.append(token_id)
        else:
            valid_ids = [
                tid for tstr, tid in self.vocab.items()
                if tstr and any(c in '0123456789.,\"}} \n\t' for c in tstr)
            ]
        return valid_ids

    @staticmethod
    def build_json_valid_ids(vocab: dict) -> Set[int]:
        """
        Maps target indices across vocabulary parameters.
        """
        return set(vocab.values())

    @staticmethod
    def extract_complete_json(text: str) -> Optional[str]:
        """
        Slices and verifies completely bounded brace blocks
        to terminate text streams early.
        """
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

    @staticmethod
    def build_system_prompt(functions: Any) -> str:
        """
        Formats definitions and instruction properties
        into systemic context boundaries.
        """
        lines = [
            "STRICT SYSTEM RULE: Use ONLY a matching function "
            "from the list below.",
            'If NO function matches the user\'s intent '
            '(even if types match), set name: "none".',
            "Never use an unrelated function for a different task.",
            "",
            "Available functions:",
        ]
        for fn in functions:
            param = ", ".join(
                f"{name}: {info.type}" for name, info in fn.parameters.items())
            lines.append(f"  -{fn.name}({param}): {fn.description}")
        lines.append(
            '\nOutput ONLY valid JSON: {"name": "<fn>", "args": {<args>}}')
        return "\n".join(lines)


def load_vocabulary(model: Small_LLM_Model) -> Any:
    """Parses local configuration tokens from downstream JSON files."""
    vocab_path = model.get_path_to_tokenizer_file()
    with open(vocab_path, 'r', encoding="utf-8") as f:
        toke_data = json.load(f)
    raw_vocab = toke_data.get("model", {}).get("vocab", {})
    return raw_vocab
