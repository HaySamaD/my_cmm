import os
import time
import json
import argparse
from typing import Any, cast
from rich.markup import escape
from rich.console import Console
from llm_sdk import Small_LLM_Model
from src.json_loader import load_function_definitions, load_test_prompts
from src.constrained_decoding import JsonConstrainedDecoder, load_vocabulary

console = Console()


class ConstrainedDecodingApp:
    """
    Main orchestration class for loading dependencies and handling the
    constrained text decoding execution pipeline.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        """
        Initializes components, loads schemas,
        test prompts, and compiles parameters.
        """
        self.args = args
        self.model = Small_LLM_Model(model_name=args.model)
        self.functions = load_function_definitions(args.functions_definition)
        self.prompts = load_test_prompts(args.input)
        self.max_tokens = args.max_tokens
        if not self.functions or not self.prompts:
            raise RuntimeError("Initialization files are empty or missing.")
        self.system_prompt = JsonConstrainedDecoder.build_system_prompt(
            self.functions)
        vocab = load_vocabulary(self.model)
        valid_names = [fn.name for fn in self.functions]
        self.decoder = JsonConstrainedDecoder(
            vocab=vocab, valid_function_names=valid_names)

    def execute_pipeline(self) -> None:
        """
        Iterates through the testing prompts
        and performs state-constrained sampling loops.
        """
        console.print("\n[bold yellow]Processing prompts...[/bold yellow]\n")
        all_results = []
        start_time = time.time()

        for index, p in enumerate(self.prompts, 1):
            prompt = p.prompt
            console.print(
                f"[bold cyan]Processing prompt {index}/{len(self.prompts)}:"
                f"[/bold cyan] {prompt}")

            full_prompt = (
                f"{self.system_prompt}\n\nUser prompt: {prompt}\nAssistant:")
            generated_ids = self.model.encode(full_prompt)[0].tolist()
            all_generated = self.model.encode('{"name": "')[0].tolist()
            parsed = {"name": "none", "args": {}}
            token_count = 0

            print("  -> Generating: ", end="", flush=True)
            print('{"name": "', end="", flush=True)

            while True:
                if token_count >= self.max_tokens:
                    console.print("\n  -> [bold red][Error] Maximum "
                                  "token limit reached (Loop safety "
                                  "break)[/bold red]")
                    break

                current_text = self.model.decode(all_generated)
                logits = self.model.get_logits_from_input_ids(
                    generated_ids + all_generated)

                next_id = self.decoder.get_approve_valid_token(
                    logits, current_text)
                all_generated.append(next_id)
                token_count += 1

                print(self.model.decode([next_id]), end="", flush=True)

                text = self.model.decode(all_generated)
                clean_json = JsonConstrainedDecoder.extract_complete_json(text)
                if clean_json:
                    try:
                        parsed = json.loads(clean_json)
                        break
                    except Exception:
                        pass
            print()

            if parsed.get("name", "none") != "none":
                escaped_args = escape(str(parsed['args']))
                console.print("  -> [bold green]✔ Success:[/bold green] "
                              f"{parsed['name']}({escaped_args})")
            else:
                console.print("  -> [bold red]❌ [Error] Could not generate "
                              "function call[/bold red]")
            print()

            extracted_args = cast(dict[Any, Any], parsed.get("args", {}))
            cleaned_parameters: dict[str, Any] = {}

            current_fn_def = next((
                fn for fn in self.functions if fn.name == parsed.get("name")),
                None)

            for key, value in extracted_args.items():
                if current_fn_def and key not in current_fn_def.parameters:
                    continue

                expected_type = (
                    current_fn_def.parameters[key].type
                    if current_fn_def else "string")

                if expected_type == "integer":
                    try:
                        cleaned_parameters[key] = int(float(str(value)))
                    except ValueError:
                        cleaned_parameters[key] = value
                elif expected_type == "number":
                    try:
                        cleaned_parameters[key] = float(str(value))
                    except ValueError:
                        cleaned_parameters[key] = value
                else:
                    cleaned_parameters[key] = value

            all_results.append({
                "prompt": prompt,
                "name": parsed.get("name", "none"),
                "parameters": cleaned_parameters
            })

        self._save_results(all_results, time.time() - start_time)

    def _save_results(self,
                      results: list[dict[str, Any]],
                      total_time: float
                      ) -> None:
        """
        Filters non-matching function entries
        and stores structured answers inside output JSON path.
        """
        all_parsed_result = [r for r in results if r["name"] != "none"]
        os.makedirs(os.path.dirname(self.args.output), exist_ok=True)
        with open(self.args.output, 'w', encoding="utf-8") as f:
            json.dump(all_parsed_result, f, ensure_ascii=False, indent=2)

        console.print("The result is saved in [cyan]"
                      f"{self.args.output}[/cyan]")
        console.print("Total time: [bold yellow]"
                      f"{total_time:.2f}[/bold yellow] seconds\n")


def main() -> None:
    """Application entrypoint parsing raw command line shell options."""
    console.print("\n[bold purple]=== Program Start ===[/bold purple]\n")
    pars = argparse.ArgumentParser(
     description="Constrained Decoding Pipeline for Safe JSON Function Calling"
    )
    pars.add_argument("--input",
                      type=str,
                      default="data/input/function_calling_tests.json",
                      help="Path to input test cases."
                      )
    pars.add_argument("--functions_definition",
                      type=str,
                      default="data/input/functions_definition.json",
                      help="Path to function schema definition configurations."
                      )
    pars.add_argument("--output",
                      type=str,
                      default="data/output/function_calling_results.json",
                      help="Path to dump schema valid evaluations."
                      )
    pars.add_argument("--model",
                      type=str,
                      default="Qwen/Qwen3-0.6B",
                      help="Local path or Hugging Face Hub tracking name tag."
                      )
    pars.add_argument("--max_tokens",
                      type=int,
                      default=50,
                      help="Maximum tokens to generate the answer"
                      )

    app = ConstrainedDecodingApp(pars.parse_args())
    app.execute_pipeline()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram Stopped.")
    except Exception as e:
        print(f"Error: {e}")
