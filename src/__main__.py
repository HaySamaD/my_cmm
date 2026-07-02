import os
import time
import json
import argparse
from rich.console import Console
from typing import Any, cast
from llm_sdk import Small_LLM_Model
from src.json_loader import load_function_definitions, load_test_prompts
from src.constrained_decoding import (build_system_prompt,
                                      load_vocabulary,
                                      build_json_valid_ids,
                                      extract_complete_json,
                                      get_approve_valid_token)

console = Console()


def parsing_args() -> argparse.Namespace:
    pars = argparse.ArgumentParser(
        description="Constrained Decoding Pipeline for "
        "Safe JSON Function Calling"
    )
    pars.add_argument("--input",
                      type=str,
                      default="data/input/function_calling_tests.json")
    pars.add_argument("--functions_definition",
                      type=str,
                      default="data/input/functions_definition.json")
    pars.add_argument("--output",
                      type=str,
                      default="data/output/function_calling_results.json")
    pars.add_argument("--model",
                      type=str,
                      default="Qwen/Qwen3-0.6B")
    return pars.parse_args()


def main() -> None:
    console.print("\n[bold purple]=== Program Start ===[/bold purple]\n")
    args = parsing_args()

    print("Loading functions...")
    functions = load_function_definitions(args.functions_definition)
    if not functions:
        raise RuntimeError("No function definition found")

    prompts = load_test_prompts(args.input)
    if not prompts:
        raise RuntimeError("No function calling tests (prompt) found")

    print("Building system prompt...")
    system = build_system_prompt(functions)

    print(f"Loading model {args.model}...")
    try:
        model = Small_LLM_Model(model_name=args.model)
    except OSError:
        raise RuntimeError(f"Model {args.model} does not exist")

    print("Building valid token IDs...")
    vocab = load_vocabulary(model)
    valid_ids = build_json_valid_ids(vocab)

    all_results = []
    start_time = time.time()

    console.print("\n[bold yellow]Processing prompts...[/bold yellow]\n")

    for index, p in enumerate(prompts, 1):
        prompt = p.prompt
        console.print(
            "[bold cyan]Processing prompt "
            f"{index}/{len(prompts)}:[/bold cyan] {prompt}")

        full_prompt = f"{system}\n\nUser prompt: {prompt}\nAssistant:"
        input_ids = model.encode(full_prompt)
        generated_ids = input_ids[0].tolist()

        all_generated = []
        clean_json = None
        parsed = {"name": "none", "args": {}}

        all_generated.extend(model.encode('{"name": "')[0].tolist())

        print("  -> Generating: ", end="", flush=True)
        print('{"name": "', end="", flush=True)

        for _ in range(50):
            logits = model.get_logits_from_input_ids(
                generated_ids + all_generated)
            next_id = get_approve_valid_token(logits, valid_ids)
            all_generated.append(next_id)

            print(model.decode([next_id]), end="", flush=True)

            text = model.decode(all_generated)
            clean_json = extract_complete_json(text)
            if clean_json:
                try:
                    parsed = json.loads(clean_json)
                    break
                except Exception:
                    pass
        print()

        if parsed.get("name", "none") != "none":
            console.print(
                "  -> [bold green]✅ "
                f"{parsed['name']}({parsed['args']})[/bold green]")
        else:
            console.print(
                "  -> [bold red]❌ [Error] Could not generate "
                "function call[/bold red]")
        print()

        extracted_args = cast(dict[Any, Any], parsed.get("args", {}))
        cleaned_parameters = {}
        for key, value in extracted_args.items():
            if type(value) in (int, float):
                cleaned_parameters[key] = float(value)
            else:
                cleaned_parameters[key] = value

        all_results.append({
            "prompt": prompt,
            "name": parsed.get("name", "none"),
            "parameters": cleaned_parameters
        })

    total_time = time.time() - start_time
    all_parsed_result = [result
                         for result in all_results if result["name"] != "none"]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding="utf-8") as f:
        json.dump(all_parsed_result, f, ensure_ascii=False, indent=2)

    console.print(f"The result is saved in [cyan]{args.output}[/cyan]")
    console.print("[bold green]Completed[/bold green]")
    console.print(
        f"Total time: [bold yellow]{total_time:.2f}[/bold yellow] seconds\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram Stopped.")
    except Exception as e:
        print(f"Error: {e}")
