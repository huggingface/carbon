import argparse
import json
import math
import statistics
import time
from datetime import datetime
from pathlib import Path


SPECIAL_TOKENS = [
    "<oov>",
    "<s>",
    "</s>",
    "<pad>",
    "<mask>",
    "<bog>",
    "<eog>",
    "<bok>",
    "<eok>",
    "<+>",
    "<->",
    "<cds>",
    "<pseudo>",
    "<tRNA>",
    "<rRNA>",
    "<ncRNA>",
    "<miscRNA>",
    "<mam>",
    "<vrt>",
    "<inv>",
    "<pln>",
    "<fng>",
    "<prt>",
    "<arc>",
    "<bct>",
    "<mit>",
    "<plt>",
    "<plm>",
    "<vir>",
    "<sp0>",
    "<sp1>",
    "<sp2>",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark GENERator generation through native Transformers with a "
            "vLLM-like serving metric block."
        )
    )
    parser.add_argument(
        "--model",
        default="GenerTeam/GENERator-v2-eukaryote-1.2b-base",
        help="Hugging Face model id or local path.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="Prepared JSONL prompt file.",
    )
    parser.add_argument("--num-prompts", type=int, default=16)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for result files.",
    )
    parser.add_argument("--result-json", type=Path, default=None)
    parser.add_argument("--detailed-json", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bp-per-token", type=int, default=6)
    parser.add_argument("--kmer-size", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.00001)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--num-warmups", type=int, default=0)
    parser.add_argument("--label", default="generator-transformers")
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


class KmerTokenizer:
    def __init__(self, k: int):
        import itertools
        import re

        self.k = k
        self.special_tokens = SPECIAL_TOKENS
        self.vocab = {
            token: index
            for index, token in enumerate(
                self.special_tokens
                + ["".join(kmer) for kmer in itertools.product("ATCG", repeat=k)]
            )
        }
        self.ids_to_tokens = {index: token for token, index in self.vocab.items()}
        self.special_token_pattern = re.compile(
            "|".join(re.escape(token) for token in self.special_tokens)
        )
        self.dna_pattern = re.compile(f"[A-Z]{{{self.k}}}|[A-Z]+")
        self.bos_token_id = self.vocab["<s>"]
        self.eos_token_id = self.vocab["</s>"]
        self.pad_token_id = self.vocab["<pad>"]
        self.unk_token_id = self.vocab["<oov>"]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def tokenize(self, text: str) -> list[str]:
        tokens = []
        pos = 0
        while pos < len(text):
            special_match = self.special_token_pattern.match(text, pos)
            if special_match:
                tokens.append(special_match.group())
                pos = special_match.end()
                continue
            dna_match = self.dna_pattern.match(text, pos)
            if dna_match:
                tokens.append(dna_match.group())
                pos = dna_match.end()
                continue
            tokens.append(text[pos])
            pos += 1
        return tokens

    def encode(self, text: str, add_bos_token: bool = True) -> list[int]:
        token_ids = [
            self.vocab.get(token, self.unk_token_id) for token in self.tokenize(text)
        ]
        if add_bos_token:
            return [self.bos_token_id] + token_ids
        return token_ids

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        tokens = []
        for token_id in token_ids:
            token = self.ids_to_tokens.get(int(token_id), "<oov>")
            if skip_special_tokens and token in self.special_tokens:
                continue
            tokens.append(token)
        return "".join(tokens)


def load_model(args: argparse.Namespace):
    import torch
    from transformers import AutoModelForCausalLM

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        dtype=dtype_map[args.dtype],
        low_cpu_mem_usage=True,
    )
    model.to(args.device)
    model.eval()
    return model


def make_batch(
    requests: list[dict],
    tokenizer: KmerTokenizer,
    device: str,
) -> tuple:
    import torch

    encoded = [tokenizer.encode(request["prompt"], add_bos_token=True) for request in requests]
    max_len = max(len(item) for item in encoded)
    input_ids = []
    attention_mask = []
    for item in encoded:
        pad_len = max_len - len(item)
        input_ids.append([tokenizer.pad_token_id] * pad_len + item)
        attention_mask.append([0] * pad_len + [1] * len(item))
    return (
        torch.tensor(input_ids, dtype=torch.long, device=device),
        torch.tensor(attention_mask, dtype=torch.long, device=device),
        [len(item) for item in encoded],
    )


def run_batch(
    model,
    requests: list[dict],
    tokenizer: KmerTokenizer,
    args: argparse.Namespace,
) -> list[dict]:
    import torch

    output_tokens = int(requests[0]["output_tokens"])
    output_lengths = {int(request["output_tokens"]) for request in requests}
    if len(output_lengths) != 1:
        raise ValueError("All requests in a batch must have the same output_tokens")

    input_ids, attention_mask, prompt_lens = make_batch(requests, tokenizer, args.device)
    synchronize_cuda()
    start_perf = time.perf_counter()
    start_wall = time.time()
    with torch.inference_mode():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=output_tokens,
            min_new_tokens=output_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_k=args.top_k,
            use_cache=True,
            eos_token_id=None,
            pad_token_id=tokenizer.pad_token_id,
        )
    synchronize_cuda()
    end_perf = time.perf_counter()
    latency = end_perf - start_perf

    details = []
    for index, request in enumerate(requests):
        prompt_len = prompt_lens[index]
        generated_ids = generated[index, input_ids.shape[1] :].tolist()
        generated_ids = generated_ids[:output_tokens]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        details.append(
            {
                "success": True,
                "request_id": request.get("request_id"),
                "prompt_len": prompt_len,
                "output_len": len(generated_ids),
                "expected_output_len": output_tokens,
                "output_bp": len(generated_ids) * args.bp_per_token,
                "ttft": 0.0,
                "itl": [],
                "latency": latency,
                "start_time": start_wall,
                "generated_text": generated_text,
                "error": "",
                "metadata": request.get("metadata", {}),
                "batch_size": len(requests),
            }
        )
    return details


def summarize_results(
    requests: list[dict],
    details: list[dict],
    duration: float,
    args: argparse.Namespace,
) -> dict:
    successes = [item for item in details if item["success"]]
    failures = [item for item in details if not item["success"]]
    ttfts = [item["ttft"] for item in successes if item["ttft"] > 0]
    itls = [latency for item in successes for latency in item["itl"]]
    e2els = [item["latency"] for item in successes]
    total_input = sum(item["prompt_len"] for item in successes)
    total_output = sum(item["output_len"] for item in successes)
    completed = len(successes)
    safe_duration = duration if duration > 0 else 1e-12

    return {
        "date": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "backend": "transformers",
        "endpoint_type": "transformers",
        "label": args.label,
        "model_id": args.model,
        "tokenizer_id": "builtin-kmer",
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
        "bp_per_token": args.bp_per_token,
        "output_bp_throughput": (total_output * args.bp_per_token) / safe_duration,
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
        "mean_tpot_ms": 0.0,
        "median_tpot_ms": 0.0,
        "std_tpot_ms": 0.0,
        "p99_tpot_ms": 0.0,
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
            "Output bp throughput (bp/s):", result["output_bp_throughput"]
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Total token throughput (tok/s):", result["total_token_throughput"]
        )
    )
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

    tokenizer = KmerTokenizer(args.kmer_size)
    model = load_model(args)

    batch_size = max(1, args.batch_size)
    warmups = requests[: args.num_warmups]
    for batch_start in range(0, len(warmups), batch_size):
        _ = run_batch(
            model,
            warmups[batch_start : batch_start + batch_size],
            tokenizer,
            args,
        )

    benchmark_start = time.perf_counter()
    details = []
    try:
        for batch_start in range(0, len(requests), batch_size):
            details.extend(
                run_batch(
                    model,
                    requests[batch_start : batch_start + batch_size],
                    tokenizer,
                    args,
                )
            )
    except Exception as exc:
        details.append(
            {
                "success": False,
                "request_id": None,
                "prompt_len": 0,
                "output_len": 0,
                "expected_output_len": 0,
                "output_bp": 0,
                "ttft": 0.0,
                "itl": [],
                "latency": 0.0,
                "start_time": time.time(),
                "generated_text": "",
                "error": repr(exc),
                "metadata": {},
                "batch_size": batch_size,
            }
        )
    synchronize_cuda()
    duration = time.perf_counter() - benchmark_start

    result = summarize_results(requests, details, duration, args)
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
