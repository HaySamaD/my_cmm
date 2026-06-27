import os
import time
import json
import argparse
from src.json_loadr import load_function_definition, load_prompt
from src.constrained_decoding import build_system_prompt, load_vocabulary, build_json_valid_ids, extract_complete_json, get_approve_valid_token
from llm_sdk import Small_LLM_Model

def parsing_args():
	pars = argparse.ArgumentParser(
		description="Constrained Decoding Pipeline for Safe JSON Function Calling"
	)

	pars.add_argument(
		"--input",
		type=str,
		default="data/input/function_calling_tests.json"
	)

	pars.add_argument(
		"--functions_definition",
		type=str,
		default="data/input/functions_definition.json"
	)

	pars.add_argument(
		"--output",
		type=str,
		default="data/output/functions_output.json"
	)

	pars.add_argument(
		"--model",
		type=str,
		default="Qwen/Qwen3-0.6B"
	)

	return pars.parse_args()

def main():
	print("Program Start")
	args = parsing_args()

	print("Loading functions...")
	functions = load_function_definition(args.functions_definition)
	if not functions:
		raise RuntimeError(f"No function definition found")
	prompts = load_prompt(args.input)
	if not prompts:
		raise RuntimeError(f"No function calling tests (prompt) found")

	print("Building system prompt...")
	system = build_system_prompt(functions)

	print(f"loading model {args.model}...")
	try:
		model = Small_LLM_Model(model_name=args.model)
	except OSError:
		raise RuntimeError(f"Model {args.model} not exist")

	print("Building valid token IDs...")
	vocab = load_vocabulary(model)
	valid_ids = build_json_valid_ids(vocab)

	all_results = []

	start_time = time.time()
	print("Processing prompts...\n")
	for p in prompts:
		prompt = p.prompt
		print(f"Processing prompt {prompt}")
		full_prompt = f"{system}\n\nUser prompt: {prompt}\nAssistant:"
		input_ids = model.encode(full_prompt)
		generated_ids = input_ids[0].tolist()

		all_generated = []

		clean_json = None
		parsed = {"name": "none", "args": {}}
		all_generated.extend(model.encode('{"name": "')[0].tolist())
		for _ in range(50):
			logits = model.get_logits_from_input_ids(generated_ids + all_generated)
			next_id = get_approve_valid_token(logits, valid_ids)
			all_generated.append(next_id)
			text = model.decode(all_generated)
			# print(text)
			clean_json = extract_complete_json(text)
			if clean_json:
				try:
					parsed = json.loads(clean_json)
					break
				except Exception:
					pass

		all_results.append({
			"prompt": prompt,
			"name": parsed.get("name", "none"),
			"args": parsed.get("args", {})
		})

		if parsed.get("name", "none") != "none":
			print(f"  -> ✅ {parsed['name']}({parsed['args']})")
		else:
			print(f"  -> ❌ [Error] Could not generate function call")

	total_time = time.time() - start_time
	all_parsed_result = [result for result in all_results if result["name"] != "none"]

	os.makedirs(os.path.dirname(args.output), exist_ok=True)
	with open(args.output, 'w', encoding="utf-8") as f:
		json.dump(all_parsed_result, f, ensure_ascii=False, indent=2)
	print(f"The result is saved in {args.output}")
	print("Completed")
	print(f"Total time: {total_time:.2f} second")


if __name__ == "__main__":
	try:
		main()
	except KeyboardInterrupt as e:
		print(f"Program Stoped.")
	except Exception as e:
		print(f"Error: {e}")