import argparse
import json
import math
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_DIR = REPO_ROOT / "evaluation"
if str(EVALUATION_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATION_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Evo2 generation with a vLLM-like serving metric block. "
            "The input dataset is a JSONL file with prompt and output_tokens fields."
        )
    )
    parser.add_argument("--model", default="evo2_7b", help="Evo2 model name")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="Prepared JSONL prompt file",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=16,
        help="Number of prompts to benchmark. <=0 uses every JSONL row.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for result files",
    )
    parser.add_argument(
        "--result-json",
        type=Path,
        default=None,
        help="Aggregate result JSON path. Defaults to <output-dir>/benchmark.json.",
    )
    parser.add_argument(
        "--detailed-json",
        type=Path,
        default=None,
        help="Per-request result JSON path. Defaults to <output-dir>/detailed.json.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Evo2 generation temperature",
    )
    parser.add_argument("--top-k", type=int, default=1, help="Evo2 top_k")
    parser.add_argument("--top-p", type=float, default=0.0, help="Evo2 top_p")
    parser.add_argument(
        "--force-prompt-threshold",
        type=int,
        default=None,
        help="Optional force_prompt_threshold passed to Evo2.generate",
    )
    parser.add_argument(
        "--num-warmups",
        type=int,
        default=0,
        help="Number of initial requests to run before measured requests",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "Number of same-length prompts to generate together. Values greater "
            "than 1 require prompts in each batch to have the same length and "
            "output_tokens."
        ),
    )
    parser.add_argument(
        "--label",
        default="evo2",
        help="Result label written to the aggregate JSON",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "prompt" not in item:
                raise ValueError(f"{path}:{line_number} is missing 'prompt'")
            if "output_tokens" not in item:
                raise ValueError(f"{path}:{line_number} is missing 'output_tokens'")
            item["output_tokens"] = int(item["output_tokens"])
            rows.append(item)
    return rows


def percentile(values: list[float], percentile_value: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile_value / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[int(rank)]
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def mean_ms(values: list[float]) -> float:
    return (statistics.mean(values) * 1000.0) if values else 0.0


def median_ms(values: list[float]) -> float:
    return (statistics.median(values) * 1000.0) if values else 0.0


def std_ms(values: list[float]) -> float:
    return (statistics.pstdev(values) * 1000.0) if len(values) > 1 else 0.0


def p99_ms(values: list[float]) -> float:
    return percentile(values, 99.0) * 1000.0


def synchronize_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        return


def load_evo2_model(model_name: str):
    try:
        from evo2_runtime import preload_cudnn_libraries

        preload_cudnn_libraries()
        from evo2 import Evo2
    except Exception as exc:
        raise RuntimeError(
            "Evo2 is not available. Run the benchmark from an environment with "
            "the repo's evo2 dependency group installed."
        ) from exc
    return Evo2(model_name)


def run_one_request(model, request: dict, args: argparse.Namespace) -> dict:
    from vortex.model.generation import Generator

    prompt = request["prompt"]
    output_tokens = int(request["output_tokens"])
    step_times = []
    generator = Generator(
        model.model,
        model.tokenizer,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
    )

    def token_callback(_step_index):
        synchronize_cuda()
        step_times.append(time.perf_counter())

    synchronize_cuda()
    start_perf = time.perf_counter()
    start_wall = time.time()
    try:
        output_ids, _logits, _inference_params = generator.generate(
            device="cuda:0",
            input_string=prompt,
            num_tokens=output_tokens,
            cached_generation=True,
            print_generation=False,
            verbose=False,
            stop_at_eos=False,
            force_prompt_threshold=args.force_prompt_threshold,
            token_callback=token_callback,
        )
        synchronize_cuda()
        end_perf = time.perf_counter()
        generated_batch = list(model.tokenizer.detokenize_batch(output_ids))
        generated_text = generated_batch[0] if generated_batch else ""
        actual_output_len = len(generated_text)
        if actual_output_len == 0:
            actual_output_len = output_tokens
        ttft = step_times[0] - start_perf if step_times else 0.0
        itls = [
            step_times[index] - step_times[index - 1]
            for index in range(1, len(step_times))
        ]
        latency = end_perf - start_perf
        return {
            "success": True,
            "request_id": request.get("request_id"),
            "prompt_len": len(prompt),
            "output_len": actual_output_len,
            "expected_output_len": output_tokens,
            "ttft": ttft,
            "itl": itls,
            "latency": latency,
            "start_time": start_wall,
            "generated_text": generated_text,
            "error": "",
            "metadata": request.get("metadata", {}),
        }
    except Exception as exc:
        synchronize_cuda()
        end_perf = time.perf_counter()
        return {
            "success": False,
            "request_id": request.get("request_id"),
            "prompt_len": len(prompt),
            "output_len": 0,
            "expected_output_len": output_tokens,
            "ttft": 0.0,
            "itl": [],
            "latency": end_perf - start_perf,
            "start_time": start_wall,
            "generated_text": "",
            "error": repr(exc),
            "metadata": request.get("metadata", {}),
        }


def run_batch_requests(
    model,
    requests: list[dict],
    args: argparse.Namespace,
) -> list[dict]:
    if len(requests) == 1:
        return [run_one_request(model, requests[0], args)]

    from vortex.model.generation import Generator, prepare_batch

    output_tokens = int(requests[0]["output_tokens"])
    prompt_lengths = {len(request["prompt"]) for request in requests}
    output_lengths = {int(request["output_tokens"]) for request in requests}
    if len(prompt_lengths) != 1 or len(output_lengths) != 1:
        return [run_one_request(model, request, args) for request in requests]

    step_times = []
    generator = Generator(
        model.model,
        model.tokenizer,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
    )

    def token_callback(_step_index):
        synchronize_cuda()
        step_times.append(time.perf_counter())

    prompts = [request["prompt"] for request in requests]
    synchronize_cuda()
    start_perf = time.perf_counter()
    start_wall = time.time()
    try:
        input_ids, _lengths = prepare_batch(
            prompts,
            model.tokenizer,
            prepend_bos=False,
            device="cuda:0",
        )
        output_ids, _logits, _inference_params = generator.generate(
            device="cuda:0",
            input_ids=input_ids,
            num_tokens=output_tokens,
            cached_generation=True,
            print_generation=False,
            verbose=False,
            stop_at_eos=False,
            force_prompt_threshold=args.force_prompt_threshold,
            token_callback=token_callback,
        )
        synchronize_cuda()
        end_perf = time.perf_counter()
        generated_texts = list(model.tokenizer.detokenize_batch(output_ids))
        ttft = step_times[0] - start_perf if step_times else 0.0
        itls = [
            step_times[index] - step_times[index - 1]
            for index in range(1, len(step_times))
        ]
        latency = end_perf - start_perf
        details = []
        for index, request in enumerate(requests):
            generated_text = (
                generated_texts[index] if index < len(generated_texts) else ""
            )
            actual_output_len = len(generated_text) or output_tokens
            details.append(
                {
                    "success": True,
                    "request_id": request.get("request_id"),
                    "prompt_len": len(request["prompt"]),
                    "output_len": actual_output_len,
                    "expected_output_len": output_tokens,
                    "ttft": ttft,
                    "itl": itls,
                    "latency": latency,
                    "start_time": start_wall,
                    "generated_text": generated_text,
                    "error": "",
                    "metadata": request.get("metadata", {}),
                    "batch_size": len(requests),
                }
            )
        return details
    except Exception as exc:
        synchronize_cuda()
        end_perf = time.perf_counter()
        return [
            {
                "success": False,
                "request_id": request.get("request_id"),
                "prompt_len": len(request["prompt"]),
                "output_len": 0,
                "expected_output_len": int(request["output_tokens"]),
                "ttft": 0.0,
                "itl": [],
                "latency": end_perf - start_perf,
                "start_time": start_wall,
                "generated_text": "",
                "error": repr(exc),
                "metadata": request.get("metadata", {}),
                "batch_size": len(requests),
            }
            for request in requests
        ]


def summarize_results(
    requests: list[dict],
    details: list[dict],
    duration: float,
    args: argparse.Namespace,
) -> dict:
    successes = [item for item in details if item["success"]]
    failures = [item for item in details if not item["success"]]
    ttfts = [item["ttft"] for item in successes]
    itls = [latency for item in successes for latency in item["itl"]]
    tpots = [
        (item["latency"] - item["ttft"]) / (item["output_len"] - 1)
        for item in successes
        if item["output_len"] > 1
    ]
    e2els = [item["latency"] for item in successes]
    total_input = sum(item["prompt_len"] for item in successes)
    total_output = sum(item["output_len"] for item in successes)
    completed = len(successes)
    safe_duration = duration if duration > 0 else 1e-12

    result = {
        "date": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "backend": "evo2",
        "endpoint_type": "evo2",
        "label": args.label,
        "model_id": args.model,
        "tokenizer_id": args.model,
        "num_prompts": len(requests),
        "batch_size": max(1, args.batch_size),
        "duration": duration,
        "completed": completed,
        "failed": len(failures),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "request_throughput": completed / safe_duration,
        "request_goodput": None,
        "output_throughput": total_output / safe_duration,
        "total_token_throughput": (total_input + total_output) / safe_duration,
        "input_lens": [item["prompt_len"] for item in details],
        "output_lens": [item["output_len"] for item in details],
        "ttfts": [item["ttft"] for item in details],
        "itls": [item["itl"] for item in details],
        "start_times": [item["start_time"] for item in details],
        "generated_texts": [item["generated_text"] for item in details],
        "errors": [item["error"] for item in details],
        "mean_ttft_ms": mean_ms(ttfts),
        "median_ttft_ms": median_ms(ttfts),
        "std_ttft_ms": std_ms(ttfts),
        "p99_ttft_ms": p99_ms(ttfts),
        "mean_tpot_ms": mean_ms(tpots),
        "median_tpot_ms": median_ms(tpots),
        "std_tpot_ms": std_ms(tpots),
        "p99_tpot_ms": p99_ms(tpots),
        "mean_itl_ms": mean_ms(itls),
        "median_itl_ms": median_ms(itls),
        "std_itl_ms": std_ms(itls),
        "p99_itl_ms": p99_ms(itls),
        "mean_e2el_ms": mean_ms(e2els),
        "median_e2el_ms": median_ms(e2els),
        "std_e2el_ms": std_ms(e2els),
        "p99_e2el_ms": p99_ms(e2els),
        "max_output_tokens_per_s": 0.0,
        "max_concurrent_requests": min(max(1, args.batch_size), completed),
        "rtfx": 0.0,
    }
    return result


def print_metric_block(result: dict) -> None:
    print("{s:{c}^{n}}".format(s=" Serving Benchmark Result ", n=50, c="="))
    print("{:<40} {:<10}".format("Successful requests:", result["completed"]))
    print("{:<40} {:<10}".format("Failed requests:", result["failed"]))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", result["duration"]))
    print("{:<40} {:<10}".format("Total input tokens:", result["total_input_tokens"]))
    print(
        "{:<40} {:<10}".format(
            "Total generated tokens:", result["total_output_tokens"]
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Request throughput (req/s):", result["request_throughput"]
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Output token throughput (tok/s):", result["output_throughput"]
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Total token throughput (tok/s):", result["total_token_throughput"]
        )
    )
    print("{s:{c}^{n}}".format(s="Time to First Token", n=50, c="-"))
    print("{:<40} {:<10.2f}".format("Mean TTFT (ms):", result["mean_ttft_ms"]))
    print("{:<40} {:<10.2f}".format("Median TTFT (ms):", result["median_ttft_ms"]))
    print("{:<40} {:<10.2f}".format("P99 TTFT (ms):", result["p99_ttft_ms"]))
    print("{s:{c}^{n}}".format(s="Time per Output Token (excl. 1st token)", n=50, c="-"))
    print("{:<40} {:<10.2f}".format("Mean TPOT (ms):", result["mean_tpot_ms"]))
    print("{:<40} {:<10.2f}".format("Median TPOT (ms):", result["median_tpot_ms"]))
    print("{:<40} {:<10.2f}".format("P99 TPOT (ms):", result["p99_tpot_ms"]))
    print("{s:{c}^{n}}".format(s="Inter-token Latency", n=50, c="-"))
    print("{:<40} {:<10.2f}".format("Mean ITL (ms):", result["mean_itl_ms"]))
    print("{:<40} {:<10.2f}".format("Median ITL (ms):", result["median_itl_ms"]))
    print("{:<40} {:<10.2f}".format("P99 ITL (ms):", result["p99_itl_ms"]))
    print("=" * 50)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_json = args.result_json or (args.output_dir / "benchmark.json")
    detailed_json = args.detailed_json or (args.output_dir / "detailed.json")

    requests = read_jsonl(args.dataset_path)
    if args.num_prompts > 0:
        requests = requests[: args.num_prompts]
    if not requests:
        raise ValueError("No requests to benchmark")

    model = load_evo2_model(args.model)

    warmups = requests[: args.num_warmups]
    batch_size = max(1, args.batch_size)
    for batch_start in range(0, len(warmups), batch_size):
        _ = run_batch_requests(
            model,
            warmups[batch_start : batch_start + batch_size],
            args,
        )

    measured_requests = requests
    benchmark_start = time.perf_counter()
    details = []
    for batch_start in range(0, len(measured_requests), batch_size):
        details.extend(
            run_batch_requests(
                model,
                measured_requests[batch_start : batch_start + batch_size],
                args,
            )
        )
    synchronize_cuda()
    duration = time.perf_counter() - benchmark_start

    result = summarize_results(measured_requests, details, duration, args)
    print_metric_block(result)

    with detailed_json.open("w", encoding="utf-8") as handle:
        json.dump(details, handle, indent=2)
    with result_json.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print(f"Result JSON: {result_json}")
    print(f"Detailed JSON: {detailed_json}")

    if result["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
