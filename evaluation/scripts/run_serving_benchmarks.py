import argparse
import csv
import json
import math
import os
import random
import re
import shlex
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EVO2_BENCHMARK_SCRIPT = REPO_ROOT / "evaluation" / "scripts" / "benchmark_evo2_serving.py"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "scratch" / "serving_benchmarks"
DEFAULT_CARBON_MODEL = "HuggingFaceBio/Carbon-3B"
DEFAULT_CARBON_DRAFT_MODEL = "HuggingFaceBio/Carbon-500M"
DEFAULT_GENERATOR_MODEL = "GenerTeam/GENERator-v2-eukaryote-3b-base"
DEFAULT_EVO2_MODEL = "evo2_7b"
SUMMARY_FIELDS = [
    "run_name",
    "status",
    "backend",
    "model",
    "draft_model",
    "num_speculative_tokens",
    "gpu_id",
    "port",
    "prompt_file",
    "run_dir",
    "result_json",
    "completed",
    "failed",
    "duration",
    "total_input_tokens",
    "total_output_tokens",
    "request_throughput",
    "output_throughput",
    "total_token_throughput",
    "mean_ttft_ms",
    "mean_tpot_ms",
    "mean_itl_ms",
    "p99_ttft_ms",
    "p99_tpot_ms",
    "p99_itl_ms",
    "skip_reason",
    "error",
]


@dataclass
class RunSpec:
    name: str
    backend: str
    model: str
    prompt_file: Path
    run_dir: Path
    port: int | None = None
    served_model_name: str | None = None
    draft_model: str | None = None
    num_speculative_tokens: int | None = None
    speculative_config: dict | None = None
    server_extra_args: list[str] = field(default_factory=list)
    requires_probe: bool = False
    skip_on_probe_failure: bool = True
    skip_reason: str = ""
    gpu_id: str = ""
    metadata: dict = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare and run serving benchmarks for Carbon, speculative Carbon, "
            "Evo2, and optionally GENERator."
        )
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--run-id",
        default=None,
        help="Output run id. Defaults to a UTC timestamp.",
    )
    parser.add_argument("--data-repo", default="GenerTeam/sequence-recovery")
    parser.add_argument("--data-config", default="eukaryote")
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-prompts", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--input-bp", type=int, default=1080)
    parser.add_argument("--output-bp", type=int, default=1080)
    parser.add_argument("--bp-per-token", type=int, default=6)
    parser.add_argument("--carbon-model", default=DEFAULT_CARBON_MODEL)
    parser.add_argument(
        "--carbon-revision",
        default=None,
        help="Target Carbon model revision passed to vLLM --revision.",
    )
    parser.add_argument(
        "--carbon-code-revision",
        default=None,
        help="Target Carbon code revision passed to vLLM --code-revision.",
    )
    parser.add_argument(
        "--carbon-tokenizer-revision",
        default=None,
        help="Target Carbon tokenizer revision passed to vLLM --tokenizer-revision.",
    )
    parser.add_argument("--carbon-draft-model", default=DEFAULT_CARBON_DRAFT_MODEL)
    parser.add_argument(
        "--carbon-draft-revision",
        default=None,
        help="Draft Carbon model revision recorded in --speculative-config.",
    )
    parser.add_argument(
        "--carbon-draft-code-revision",
        default=None,
        help="Draft Carbon code revision recorded in --speculative-config.",
    )
    parser.add_argument(
        "--carbon-speculative-tokens",
        type=int,
        nargs="+",
        default=[2, 4, 8],
    )
    parser.add_argument("--evo2-model", default=DEFAULT_EVO2_MODEL)
    parser.add_argument("--generator-model", default=DEFAULT_GENERATOR_MODEL)
    parser.add_argument(
        "--generator",
        choices=["auto", "always", "never"],
        default="auto",
        help="Whether to include the GENERator vLLM benchmark.",
    )
    parser.add_argument("--skip-carbon", action="store_true")
    parser.add_argument("--skip-speculative", action="store_true")
    parser.add_argument("--skip-evo2", action="store_true")
    parser.add_argument(
        "--gpu-ids",
        default=None,
        help="Comma-separated physical GPU IDs. Defaults to visible GPUs.",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=8,
        help="Maximum number of one-GPU benchmark workers to run concurrently.",
    )
    parser.add_argument("--base-port", type=int, default=8100)
    parser.add_argument("--server-start-timeout", type=float, default=900.0)
    parser.add_argument("--probe-timeout", type=float, default=120.0)
    parser.add_argument("--benchmark-timeout", type=float, default=None)
    parser.add_argument("--vllm-dtype", default="bfloat16")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument(
        "--vllm-max-model-len",
        type=int,
        default=512,
        help="Maximum model length for vLLM servers.",
    )
    parser.add_argument(
        "--vllm-extra-arg",
        action="append",
        default=[],
        help="Extra argument passed to every vLLM serve command. Repeatable.",
    )
    parser.add_argument(
        "--bench-extra-arg",
        action="append",
        default=[],
        help="Extra argument passed to every vLLM bench serve command. Repeatable.",
    )
    parser.add_argument(
        "--skip-spec-vocab-check",
        action="store_true",
        help="Skip the target/draft vocab compatibility preflight.",
    )
    parser.add_argument("--evo2-temperature", type=float, default=1.0)
    parser.add_argument("--evo2-top-k", type=int, default=1)
    parser.add_argument("--evo2-top-p", type=float, default=0.0)
    parser.add_argument("--evo2-batch-size", type=int, default=1)
    parser.add_argument("--evo2-force-prompt-threshold", type=int, default=None)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sanitize_path_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    return sanitized or "run"


def normalize_hf_repo_id(repo_id: str) -> str:
    prefix = "hf://datasets/"
    if repo_id.startswith(prefix):
        return repo_id[len(prefix) :]
    return repo_id


def normalize_evo2_model_name(model_name: str) -> str:
    prefix = "arcinstitute/"
    if model_name.startswith(prefix):
        return model_name[len(prefix) :]
    return model_name


def evo2_run_name(model_name: str) -> str:
    normalized_model_name = normalize_evo2_model_name(model_name).replace("_", "-")
    return sanitize_path_component(normalized_model_name)


def model_run_name(model_name: str) -> str:
    return sanitize_path_component(model_name.rsplit("/", 1)[-1].lower())


def load_sequence_recovery_rows(args: argparse.Namespace) -> list[dict]:
    from datasets import load_dataset

    repo_id = normalize_hf_repo_id(args.data_repo)
    try:
        dataset = load_dataset(repo_id, args.data_config, split=args.split)
    except Exception:
        parquet_path = f"hf://datasets/{repo_id}/{args.data_config}/{args.split}.parquet"
        dataset = load_dataset("parquet", data_files={args.split: parquet_path}, split=args.split)
    return list(dataset)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def prepare_prompt_files(args: argparse.Namespace, run_dir: Path) -> dict:
    rows = load_sequence_recovery_rows(args)
    min_len = args.input_bp + args.output_bp
    eligible = [
        (index, row)
        for index, row in enumerate(rows)
        if isinstance(row.get("sequence"), str) and len(row["sequence"]) >= min_len
    ]
    if len(eligible) < args.num_prompts:
        raise ValueError(
            f"Need {args.num_prompts} sequences of at least {min_len} bp, "
            f"but only found {len(eligible)} eligible rows"
        )

    rng = random.Random(args.seed)
    selected = rng.sample(eligible, args.num_prompts)
    effective_sixmer_input_bp = (args.input_bp // args.bp_per_token) * args.bp_per_token
    if effective_sixmer_input_bp <= 0:
        raise ValueError("--input-bp must be at least --bp-per-token for vLLM runs")
    sixmer_output_tokens = math.ceil(args.output_bp / args.bp_per_token)
    sixmer_effective_output_bp = sixmer_output_tokens * args.bp_per_token

    carbon_rows = []
    generator_rows = []
    evo2_rows = []
    sample_metadata = []

    for ordinal, (source_index, row) in enumerate(selected):
        sequence = row["sequence"].upper()
        heldout_target = sequence[-args.output_bp :]
        full_context = sequence[-(args.input_bp + args.output_bp) : -args.output_bp]
        sixmer_context = full_context[-effective_sixmer_input_bp:]
        request_id = f"sample-{ordinal:04d}"
        common_metadata = {
            "request_id": request_id,
            "source_index": int(source_index),
            "data_type": row.get("type"),
            "requested_input_bp": int(args.input_bp),
            "requested_output_bp": int(args.output_bp),
            "target_sequence": heldout_target,
        }
        carbon_rows.append(
            {
                "request_id": request_id,
                "prompt": "<dna>" + sixmer_context,
                "output_tokens": sixmer_output_tokens,
                "metadata": {
                    **common_metadata,
                    "prompt_family": "carbon",
                    "prompt_prefix": "<dna>",
                    "effective_input_bp": effective_sixmer_input_bp,
                    "effective_output_bp": sixmer_effective_output_bp,
                },
            }
        )
        generator_rows.append(
            {
                "request_id": request_id,
                "prompt": sixmer_context,
                "output_tokens": sixmer_output_tokens,
                "metadata": {
                    **common_metadata,
                    "prompt_family": "generator",
                    "prompt_prefix": "",
                    "effective_input_bp": effective_sixmer_input_bp,
                    "effective_output_bp": sixmer_effective_output_bp,
                },
            }
        )
        evo2_rows.append(
            {
                "request_id": request_id,
                "prompt": full_context,
                "output_tokens": args.output_bp,
                "metadata": {
                    **common_metadata,
                    "prompt_family": "evo2",
                    "prompt_prefix": "",
                    "effective_input_bp": args.input_bp,
                    "effective_output_bp": args.output_bp,
                },
            }
        )
        sample_metadata.append(
            {
                **common_metadata,
                "sequence_len_bp": len(sequence),
                "sixmer_context": sixmer_context,
                "evo2_context": full_context,
            }
        )

    prompts_dir = run_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    carbon_path = prompts_dir / "carbon_prompts.jsonl"
    generator_path = prompts_dir / "generator_prompts.jsonl"
    evo2_path = prompts_dir / "evo2_prompts.jsonl"
    write_jsonl(carbon_path, carbon_rows)
    write_jsonl(generator_path, generator_rows)
    write_jsonl(evo2_path, evo2_rows)

    metadata = {
        "data_repo": args.data_repo,
        "data_config": args.data_config,
        "split": args.split,
        "num_prompts": args.num_prompts,
        "seed": args.seed,
        "requested_input_bp": args.input_bp,
        "requested_output_bp": args.output_bp,
        "bp_per_token": args.bp_per_token,
        "sixmer_effective_input_bp": effective_sixmer_input_bp,
        "sixmer_output_tokens": sixmer_output_tokens,
        "sixmer_effective_output_bp": sixmer_effective_output_bp,
        "evo2_input_tokens": args.input_bp,
        "evo2_output_tokens": args.output_bp,
        "prompt_files": {
            "carbon": str(carbon_path),
            "generator": str(generator_path),
            "evo2": str(evo2_path),
        },
        "samples": sample_metadata,
    }
    metadata_path = prompts_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    metadata["metadata_path"] = str(metadata_path)
    return metadata


def parse_gpu_ids(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def resolve_gpu_ids(args: argparse.Namespace) -> list[str]:
    if args.gpu_ids:
        return parse_gpu_ids(args.gpu_ids)

    env_value = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if env_value and env_value != "NoDevFiles":
        return parse_gpu_ids(env_value)

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            output = subprocess.check_output(
                [
                    nvidia_smi,
                    "--query-gpu=index",
                    "--format=csv,noheader",
                ],
                text=True,
            )
            gpu_ids = [line.strip() for line in output.splitlines() if line.strip()]
            if gpu_ids:
                return gpu_ids
        except Exception:
            pass

    try:
        import torch

        return [str(index) for index in range(torch.cuda.device_count())]
    except Exception:
        return []


def vllm_executable() -> str:
    return shutil.which("vllm") or "vllm"


def build_vllm_server_command(args: argparse.Namespace, spec: RunSpec) -> list[str]:
    command = [
        vllm_executable(),
        "serve",
        spec.model,
        "--host",
        "127.0.0.1",
        "--port",
        str(spec.port),
        "--served-model-name",
        spec.served_model_name or spec.model,
        "--trust-remote-code",
        "--dtype",
        args.vllm_dtype,
        "--tensor-parallel-size",
        "1",
        "--gpu-memory-utilization",
        str(args.vllm_gpu_memory_utilization),
        "--max-model-len",
        str(args.vllm_max_model_len),
    ]
    if spec.speculative_config:
        speculative_config = json.dumps(spec.speculative_config, separators=(",", ":"))
        command.extend(["--speculative-config", speculative_config])
    command.extend(spec.server_extra_args)
    command.extend(args.vllm_extra_arg)
    return command


def build_vllm_bench_command(args: argparse.Namespace, spec: RunSpec) -> list[str]:
    command = [
        vllm_executable(),
        "bench",
        "serve",
        "--backend",
        "openai",
        "--base-url",
        f"http://127.0.0.1:{spec.port}",
        "--endpoint",
        "/v1/completions",
        "--model",
        spec.model,
        "--served-model-name",
        spec.served_model_name or spec.model,
        "--dataset-name",
        "custom",
        "--dataset-path",
        str(spec.prompt_file),
        "--custom-output-len",
        "-1",
        "--num-prompts",
        str(args.num_prompts),
        "--no-oversample",
        "--skip-chat-template",
        "--ignore-eos",
        "--temperature",
        "0",
        "--save-result",
        "--save-detailed",
        "--result-dir",
        str(spec.run_dir),
        "--result-filename",
        "benchmark.json",
        "--seed",
        str(args.seed),
        "--disable-shuffle",
        "--trust-remote-code",
        "--label",
        spec.name,
        "--metadata",
        f"run_name={spec.name}",
        f"requested_input_bp={args.input_bp}",
        f"requested_output_bp={args.output_bp}",
    ]
    if spec.num_speculative_tokens is not None:
        command.extend(["num_speculative_tokens=" + str(spec.num_speculative_tokens)])
    command.extend(args.bench_extra_arg)
    return command


def build_evo2_command(args: argparse.Namespace, spec: RunSpec) -> list[str]:
    command = [
        sys.executable,
        str(EVO2_BENCHMARK_SCRIPT),
        "--model",
        spec.model,
        "--dataset-path",
        str(spec.prompt_file),
        "--num-prompts",
        str(args.num_prompts),
        "--output-dir",
        str(spec.run_dir),
        "--result-json",
        str(spec.run_dir / "benchmark.json"),
        "--detailed-json",
        str(spec.run_dir / "detailed.json"),
        "--temperature",
        str(args.evo2_temperature),
        "--top-k",
        str(args.evo2_top_k),
        "--top-p",
        str(args.evo2_top_p),
        "--batch-size",
        str(args.evo2_batch_size),
        "--label",
        spec.name,
    ]
    if args.evo2_force_prompt_threshold is not None:
        command.extend(
            ["--force-prompt-threshold", str(args.evo2_force_prompt_threshold)]
        )
    return command


def carbon_server_extra_args(args: argparse.Namespace) -> list[str]:
    extra_args = []
    if args.carbon_revision:
        extra_args.extend(["--revision", args.carbon_revision])
    if args.carbon_code_revision:
        extra_args.extend(["--code-revision", args.carbon_code_revision])
    if args.carbon_tokenizer_revision:
        extra_args.extend(["--tokenizer-revision", args.carbon_tokenizer_revision])
    return extra_args


def carbon_speculative_config(args: argparse.Namespace, token_count: int) -> dict:
    config = {
        "model": args.carbon_draft_model,
        "num_speculative_tokens": token_count,
        "draft_tensor_parallel_size": 1,
    }
    if args.carbon_draft_revision:
        config["revision"] = args.carbon_draft_revision
    if args.carbon_draft_code_revision:
        config["code_revision"] = args.carbon_draft_code_revision
    return config


def check_speculative_vocab_compatibility(
    target_model: str,
    draft_model: str,
    output_path: Path,
) -> tuple[bool, str]:
    from transformers import AutoConfig, AutoTokenizer

    details = {
        "target_model": target_model,
        "draft_model": draft_model,
        "checks": {},
    }
    try:
        target_config = AutoConfig.from_pretrained(target_model, trust_remote_code=True)
        draft_config = AutoConfig.from_pretrained(draft_model, trust_remote_code=True)
        target_vocab = getattr(target_config, "vocab_size", None)
        draft_vocab = getattr(draft_config, "vocab_size", None)
        details["checks"]["config_vocab_size"] = {
            "target": target_vocab,
            "draft": draft_vocab,
            "matches": target_vocab == draft_vocab,
        }

        target_tokenizer = AutoTokenizer.from_pretrained(
            target_model, trust_remote_code=True
        )
        draft_tokenizer = AutoTokenizer.from_pretrained(
            draft_model, trust_remote_code=True
        )
        target_tokenizer_len = len(target_tokenizer)
        draft_tokenizer_len = len(draft_tokenizer)
        details["checks"]["tokenizer_length"] = {
            "target": target_tokenizer_len,
            "draft": draft_tokenizer_len,
            "matches": target_tokenizer_len == draft_tokenizer_len,
        }

        ok = target_vocab == draft_vocab and target_tokenizer_len == draft_tokenizer_len
        details["ok"] = ok
        reason = (
            "target and draft vocab sizes match"
            if ok
            else "target and draft vocab sizes do not match"
        )
    except Exception as exc:
        ok = False
        reason = f"vocab compatibility check failed: {exc!r}"
        details["ok"] = False
        details["error"] = reason

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(details, handle, indent=2)
    return ok, reason


def build_run_specs(
    args: argparse.Namespace,
    run_dir: Path,
    prompt_metadata: dict,
) -> list[RunSpec]:
    prompt_files = prompt_metadata["prompt_files"]
    specs = []
    spec_preflight_ok = True
    spec_preflight_reason = ""
    if (
        not args.dry_run
        and not args.prepare_only
        and not args.skip_carbon
        and not args.skip_speculative
        and not args.skip_spec_vocab_check
        and args.carbon_speculative_tokens
    ):
        spec_preflight_ok, spec_preflight_reason = check_speculative_vocab_compatibility(
            args.carbon_model,
            args.carbon_draft_model,
            run_dir / "speculative_vocab_preflight.json",
        )

    if not args.skip_carbon:
        carbon_name = model_run_name(args.carbon_model)
        carbon_extra_args = carbon_server_extra_args(args)
        carbon_metadata = {
            "prompt_family": "carbon",
            "carbon_revision": args.carbon_revision or "",
            "carbon_code_revision": args.carbon_code_revision or "",
            "carbon_tokenizer_revision": args.carbon_tokenizer_revision or "",
            "carbon_draft_revision": args.carbon_draft_revision or "",
            "carbon_draft_code_revision": args.carbon_draft_code_revision or "",
        }
        specs.append(
            RunSpec(
                name=f"{carbon_name}-vllm",
                backend="vllm",
                model=args.carbon_model,
                prompt_file=Path(prompt_files["carbon"]),
                run_dir=run_dir / f"{carbon_name}-vllm",
                served_model_name=f"{carbon_name}-vllm",
                server_extra_args=carbon_extra_args,
                metadata=carbon_metadata,
            )
        )
        if not args.skip_speculative:
            for token_count in args.carbon_speculative_tokens:
                skip_reason = "" if spec_preflight_ok else spec_preflight_reason
                specs.append(
                    RunSpec(
                        name=f"{carbon_name}-spec-{token_count}",
                        backend="vllm",
                        model=args.carbon_model,
                        prompt_file=Path(prompt_files["carbon"]),
                        run_dir=run_dir / f"{carbon_name}-spec-{token_count}",
                        served_model_name=f"{carbon_name}-spec-{token_count}",
                        draft_model=args.carbon_draft_model,
                        num_speculative_tokens=token_count,
                        speculative_config=carbon_speculative_config(
                            args, token_count
                        ),
                        server_extra_args=carbon_extra_args,
                        skip_reason=skip_reason,
                        metadata=carbon_metadata,
                    )
                )

    if not args.skip_evo2:
        normalized_evo2_model = normalize_evo2_model_name(args.evo2_model)
        evo2_name = evo2_run_name(args.evo2_model)
        specs.append(
            RunSpec(
                name=evo2_name,
                backend="evo2",
                model=normalized_evo2_model,
                prompt_file=Path(prompt_files["evo2"]),
                run_dir=run_dir / evo2_name,
                metadata={
                    "prompt_family": "evo2",
                    "requested_model": args.evo2_model,
                    "normalized_model": normalized_evo2_model,
                },
            )
        )

    if args.generator != "never":
        specs.append(
            RunSpec(
                name="generator-v2-eukaryote-3b-vllm",
                backend="vllm",
                model=args.generator_model,
                prompt_file=Path(prompt_files["generator"]),
                run_dir=run_dir / "generator-v2-eukaryote-3b-vllm",
                served_model_name="generator-v2-eukaryote-3b-vllm",
                requires_probe=True,
                skip_on_probe_failure=args.generator == "auto",
                metadata={"prompt_family": "generator"},
            )
        )

    for index, spec in enumerate(specs):
        if spec.backend == "vllm":
            spec.port = args.base_port + index
    return specs


def read_file_tail(path: Path, max_chars: int = 4000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:]


def wait_for_vllm_ready(
    proc: subprocess.Popen,
    base_url: str,
    timeout: float,
    stderr_path: Path,
) -> None:
    deadline = time.time() + timeout
    url = base_url.rstrip("/") + "/v1/models"
    last_error = ""
    while time.time() < deadline:
        return_code = proc.poll()
        if return_code is not None:
            tail = read_file_tail(stderr_path)
            raise RuntimeError(
                f"vLLM server exited with code {return_code} before readiness. {tail}"
            )
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if 200 <= response.status < 300:
                    return
        except Exception as exc:
            last_error = repr(exc)
        time.sleep(2.0)
    raise TimeoutError(f"Timed out waiting for {url}. Last error: {last_error}")


def post_json(url: str, payload: dict, timeout: float) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = response.read().decode("utf-8")
    return json.loads(data)


def first_prompt(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                return json.loads(line)
    raise ValueError(f"No prompts found in {path}")


def run_generator_probe(args: argparse.Namespace, spec: RunSpec) -> None:
    prompt = first_prompt(spec.prompt_file)
    payload = {
        "model": spec.served_model_name or spec.model,
        "prompt": prompt["prompt"],
        "max_tokens": min(4, int(prompt["output_tokens"])),
        "temperature": 0,
    }
    response = post_json(
        f"http://127.0.0.1:{spec.port}/v1/completions",
        payload,
        timeout=args.probe_timeout,
    )
    if not response.get("choices"):
        raise RuntimeError(f"Completion probe returned no choices: {response}")


def terminate_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=30)


def metric_row_from_result(spec: RunSpec, result: dict, status: str) -> dict:
    row = base_summary_row(spec, status=status)
    for key in [
        "completed",
        "failed",
        "duration",
        "total_input_tokens",
        "total_output_tokens",
        "request_throughput",
        "output_throughput",
        "total_token_throughput",
        "mean_ttft_ms",
        "mean_tpot_ms",
        "mean_itl_ms",
        "p99_ttft_ms",
        "p99_tpot_ms",
        "p99_itl_ms",
    ]:
        row[key] = result.get(key, "")
    row["result_json"] = str(spec.run_dir / "benchmark.json")
    return row


def base_summary_row(spec: RunSpec, status: str) -> dict:
    return {
        "run_name": spec.name,
        "status": status,
        "backend": spec.backend,
        "model": spec.model,
        "draft_model": spec.draft_model or "",
        "num_speculative_tokens": spec.num_speculative_tokens or "",
        "gpu_id": spec.gpu_id,
        "port": spec.port or "",
        "prompt_file": str(spec.prompt_file),
        "run_dir": str(spec.run_dir),
        "result_json": "",
        "completed": "",
        "failed": "",
        "duration": "",
        "total_input_tokens": "",
        "total_output_tokens": "",
        "request_throughput": "",
        "output_throughput": "",
        "total_token_throughput": "",
        "mean_ttft_ms": "",
        "mean_tpot_ms": "",
        "mean_itl_ms": "",
        "p99_ttft_ms": "",
        "p99_tpot_ms": "",
        "p99_itl_ms": "",
        "skip_reason": spec.skip_reason,
        "error": "",
    }


def run_vllm_spec(args: argparse.Namespace, spec: RunSpec) -> dict:
    spec.run_dir.mkdir(parents=True, exist_ok=True)
    if spec.skip_reason:
        row = base_summary_row(spec, "skipped")
        row["skip_reason"] = spec.skip_reason
        return row

    server_cmd = build_vllm_server_command(args, spec)
    bench_cmd = build_vllm_bench_command(args, spec)
    command_manifest = {
        "server": server_cmd,
        "benchmark": bench_cmd,
        "cuda_visible_devices": spec.gpu_id,
    }
    with (spec.run_dir / "commands.json").open("w", encoding="utf-8") as handle:
        json.dump(command_manifest, handle, indent=2)

    if args.dry_run:
        row = base_summary_row(spec, "dry_run")
        row["result_json"] = str(spec.run_dir / "benchmark.json")
        return row

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = spec.gpu_id
    server_stdout_path = spec.run_dir / "server_stdout.log"
    server_stderr_path = spec.run_dir / "server_stderr.log"
    bench_stdout_path = spec.run_dir / "bench_stdout.log"
    bench_stderr_path = spec.run_dir / "bench_stderr.log"

    with server_stdout_path.open("w", encoding="utf-8") as server_stdout:
        with server_stderr_path.open("w", encoding="utf-8") as server_stderr:
            proc = subprocess.Popen(
                server_cmd,
                cwd=REPO_ROOT,
                env=env,
                stdout=server_stdout,
                stderr=server_stderr,
                text=True,
            )
            try:
                wait_for_vllm_ready(
                    proc,
                    f"http://127.0.0.1:{spec.port}",
                    args.server_start_timeout,
                    server_stderr_path,
                )
                if spec.requires_probe:
                    try:
                        run_generator_probe(args, spec)
                    except Exception as exc:
                        status = "skipped" if spec.skip_on_probe_failure else "failed"
                        row = base_summary_row(spec, status)
                        row["skip_reason"] = (
                            f"GENERATOR compatibility probe failed: {exc!r}"
                        )
                        row["error"] = "" if spec.skip_on_probe_failure else repr(exc)
                        return row

                with bench_stdout_path.open("w", encoding="utf-8") as bench_stdout:
                    with bench_stderr_path.open("w", encoding="utf-8") as bench_stderr:
                        completed = subprocess.run(
                            bench_cmd,
                            cwd=REPO_ROOT,
                            env=env,
                            stdout=bench_stdout,
                            stderr=bench_stderr,
                            text=True,
                            timeout=args.benchmark_timeout,
                        )
                if completed.returncode != 0:
                    row = base_summary_row(spec, "failed")
                    row["error"] = (
                        f"vLLM bench exited with code {completed.returncode}. "
                        f"{read_file_tail(bench_stderr_path)}"
                    )
                    return row

                result_path = spec.run_dir / "benchmark.json"
                with result_path.open("r", encoding="utf-8") as handle:
                    result = json.load(handle)
                return metric_row_from_result(spec, result, "completed")
            except Exception as exc:
                status = (
                    "skipped"
                    if spec.requires_probe and spec.skip_on_probe_failure
                    else "failed"
                )
                row = base_summary_row(spec, status)
                if status == "skipped":
                    row["skip_reason"] = (
                        f"GENERATOR compatibility startup failed: {exc!r}"
                    )
                else:
                    row["error"] = repr(exc)
                return row
            finally:
                terminate_process(proc)


def run_evo2_spec(args: argparse.Namespace, spec: RunSpec) -> dict:
    spec.run_dir.mkdir(parents=True, exist_ok=True)
    command = build_evo2_command(args, spec)
    with (spec.run_dir / "commands.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {"benchmark": command, "cuda_visible_devices": spec.gpu_id},
            handle,
            indent=2,
        )
    if args.dry_run:
        row = base_summary_row(spec, "dry_run")
        row["result_json"] = str(spec.run_dir / "benchmark.json")
        return row

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = spec.gpu_id
    stdout_path = spec.run_dir / "evo2_stdout.log"
    stderr_path = spec.run_dir / "evo2_stderr.log"
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout:
            with stderr_path.open("w", encoding="utf-8") as stderr:
                completed = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    env=env,
                    stdout=stdout,
                    stderr=stderr,
                    text=True,
                    timeout=args.benchmark_timeout,
                )
        if completed.returncode != 0:
            row = base_summary_row(spec, "failed")
            row["error"] = (
                f"Evo2 benchmark exited with code {completed.returncode}. "
                f"{read_file_tail(stderr_path)}"
            )
            return row
        result_path = spec.run_dir / "benchmark.json"
        with result_path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
        return metric_row_from_result(spec, result, "completed")
    except Exception as exc:
        row = base_summary_row(spec, "failed")
        row["error"] = repr(exc)
        return row


def run_spec(args: argparse.Namespace, spec: RunSpec) -> dict:
    if spec.backend == "vllm":
        return run_vllm_spec(args, spec)
    if spec.backend == "evo2":
        return run_evo2_spec(args, spec)
    raise ValueError(f"Unsupported backend: {spec.backend}")


def run_gpu_queue(args: argparse.Namespace, gpu_id: str, specs: list[RunSpec]) -> list[dict]:
    rows = []
    for spec in specs:
        spec.gpu_id = gpu_id
        print(f"[gpu {gpu_id}] starting {spec.name}", flush=True)
        row = run_spec(args, spec)
        rows.append(row)
        print(f"[gpu {gpu_id}] finished {spec.name}: {row['status']}", flush=True)
    return rows


def write_summary(run_dir: Path, rows: list[dict]) -> None:
    json_path = run_dir / "summary.json"
    csv_path = run_dir / "summary.csv"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})


def write_dry_run_script(
    args: argparse.Namespace,
    run_dir: Path,
    specs: list[RunSpec],
) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Dry-run commands generated by run_serving_benchmarks.py",
    ]
    for spec in specs:
        gpu = spec.gpu_id or "0"
        lines.append("")
        lines.append(f"# {spec.name}")
        if spec.backend == "vllm":
            lines.append(
                f"CUDA_VISIBLE_DEVICES={shlex.quote(gpu)} "
                + shlex.join(build_vllm_server_command(args, spec))
            )
            lines.append(
                f"CUDA_VISIBLE_DEVICES={shlex.quote(gpu)} "
                + shlex.join(build_vllm_bench_command(args, spec))
            )
        elif spec.backend == "evo2":
            lines.append(
                f"CUDA_VISIBLE_DEVICES={shlex.quote(gpu)} "
                + shlex.join(build_evo2_command(args, spec))
            )
    path = run_dir / "dry_run_commands.sh"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def assign_specs_to_gpus(
    specs: list[RunSpec],
    gpu_ids: list[str],
    max_parallel: int,
) -> dict[str, list[RunSpec]]:
    active_gpus = gpu_ids[: min(max_parallel, len(gpu_ids), len(specs))]
    assignments = {gpu_id: [] for gpu_id in active_gpus}
    for index, spec in enumerate(specs):
        gpu_id = active_gpus[index % len(active_gpus)]
        spec.gpu_id = gpu_id
        assignments[gpu_id].append(spec)
    return assignments


def main() -> None:
    args = parse_args()
    if args.num_prompts <= 0:
        raise ValueError("--num-prompts must be positive")
    if args.max_parallel <= 0:
        raise ValueError("--max-parallel must be positive")
    if args.bp_per_token <= 0:
        raise ValueError("--bp-per-token must be positive")

    run_id = args.run_id or datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_root / sanitize_path_component(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    prompt_metadata = prepare_prompt_files(args, run_dir)
    print(f"Wrote prompt metadata to {prompt_metadata['metadata_path']}")
    if args.prepare_only:
        write_summary(run_dir, [])
        print(f"Prepare-only complete: {run_dir}")
        return

    specs = build_run_specs(args, run_dir, prompt_metadata)
    if not specs:
        raise ValueError("No benchmark runs selected")

    gpu_ids = resolve_gpu_ids(args)
    if not gpu_ids:
        if args.dry_run:
            gpu_ids = ["0"]
        else:
            raise RuntimeError("No GPUs found. Pass --gpu-ids to select devices.")

    assignments = assign_specs_to_gpus(specs, gpu_ids, args.max_parallel)
    write_dry_run_script(args, run_dir, specs)

    if args.dry_run:
        rows = []
        for spec in specs:
            rows.append(run_spec(args, spec))
        write_summary(run_dir, rows)
        print(f"Dry-run complete. Commands: {run_dir / 'dry_run_commands.sh'}")
        print(f"Summary: {run_dir / 'summary.csv'}")
        return

    all_rows = []
    with ThreadPoolExecutor(max_workers=len(assignments)) as executor:
        futures = {
            executor.submit(run_gpu_queue, args, gpu_id, queue_specs): gpu_id
            for gpu_id, queue_specs in assignments.items()
        }
        for future in as_completed(futures):
            all_rows.extend(future.result())

    all_rows.sort(key=lambda row: [spec.name for spec in specs].index(row["run_name"]))
    write_summary(run_dir, all_rows)
    print(f"Wrote summary CSV to {run_dir / 'summary.csv'}")
    print(f"Wrote summary JSON to {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
