#!/usr/bin/env python3
"""
Run SWE-bench inference against an OpenAI-compatible API (e.g. vLLM).
Generates predictions JSONL compatible with the official SWE-bench harness.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import secrets
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import httpx
from datasets import load_dataset, load_from_disk
from tqdm.auto import tqdm

from swebench.inference.make_datasets.utils import extract_diff


LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def normalize_dataset_name(name: str) -> str:
    lower = name.lower()
    if lower in {
        "swe-bench/swe-bench_oracle",
        "swe-bench_oracle",
        "swebench_oracle",
        "oracle",
    }:
        return "princeton-nlp/SWE-bench_oracle"
    return name


def load_env_file(path: str) -> None:
    env_path = Path(path)
    if not env_path.exists():
        raise FileNotFoundError(f"env file not found: {env_path}")
    for line in env_path.read_text().splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        if "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        os.environ.setdefault(key, value)


def load_dataset_split(name_or_path: str, split: str):
    path = Path(name_or_path)
    if path.exists():
        if path.is_dir() and (path / split).exists():
            return load_from_disk(path / split)
        dataset = load_from_disk(path)
        if hasattr(dataset, "keys"):
            return dataset[split]
        return dataset
    return load_dataset(name_or_path, split=split)


def split_system_user(text: str) -> tuple[str, str]:
    text = f"{text}\n\n"
    parts = text.split("\n", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return "", text


def strip_think(text: Optional[str]) -> Optional[str]:
    if not text:
        return text
    return THINK_TAG_RE.sub("", text)


def sanitize_patch(patch: Optional[str]) -> str:
    if not patch:
        return ""
    patch = patch.strip()
    if not patch:
        return ""
    # Trim leading text before diff markers
    idx = patch.find("diff --git ")
    if idx < 0:
        idx = patch.find("--- a/")
    if idx > 0:
        patch = patch[idx:]
    # Strip fenced code blocks
    if patch.startswith("```"):
        lines = patch.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        patch = "\n".join(lines).strip()
    # Remove any stray fence lines
    patch = "\n".join(
        line for line in patch.splitlines() if not line.strip().startswith("```")
    ).strip()
    if not patch:
        return ""
    # Basic diff sanity check
    if "diff --git " not in patch and not ("--- a/" in patch and "+++ b/" in patch):
        return ""
    if not patch.endswith("\n"):
        patch += "\n"
    return patch


def parse_model_args(arg: Optional[str]) -> Dict[str, Any]:
    if not arg:
        return {}
    result: Dict[str, Any] = {}
    for item in arg.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid model arg: {item}")
        key, raw = item.split("=", 1)
        key = key.strip()
        raw = raw.strip()
        lower = raw.lower()
        if lower in {"true", "false"}:
            value: Any = lower == "true"
        else:
            try:
                value = int(raw)
            except ValueError:
                try:
                    value = float(raw)
                except ValueError:
                    value = raw
        result[key] = value
    return result


def read_existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    existing = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            instance_id = data.get("instance_id")
            if instance_id:
                existing.add(instance_id)
    return existing


def make_output_path(base: str, run_idx: int, runs: int) -> Path:
    base_path = Path(base)
    if runs <= 1:
        return base_path
    suffix = f"-run{run_idx}"
    if base_path.stem.endswith("-predictions"):
        stem = base_path.stem[: -len("-predictions")]
        name = f"{stem}{suffix}-predictions{base_path.suffix}"
    else:
        name = f"{base_path.stem}{suffix}{base_path.suffix}"
    return base_path.with_name(name)


def build_payload(
    model: str,
    system: str,
    user: str,
    temperature: Optional[float],
    top_p: Optional[float],
    max_tokens: Optional[int],
    seed: Optional[int],
    model_args: Dict[str, Any],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if seed is not None:
        payload["seed"] = seed
    payload.update(model_args)
    return payload


def post_with_retry(
    client: httpx.Client,
    api_base: str,
    payload: Dict[str, Any],
    timeout_s: float,
    max_retries: int,
    retry_backoff_s: float,
) -> Dict[str, Any]:
    attempt = 0
    while True:
        try:
            response = client.post(
                f"{api_base.rstrip('/')}/chat/completions",
                json=payload,
                timeout=timeout_s,
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status not in (429, 500, 502, 503, 504):
                raise
            attempt += 1
        except (httpx.TimeoutException, httpx.RequestError):
            attempt += 1

        if attempt > max_retries:
            raise
        sleep_s = retry_backoff_s * (2 ** (attempt - 1))
        sleep_s *= 0.5 + random.random()
        time.sleep(sleep_s)


def gather_instance_ids(path: Optional[str], inline_ids: Optional[List[str]]) -> Optional[set[str]]:
    ids: set[str] = set()
    if inline_ids:
        ids.update(inline_ids)
    if path:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                value = line.strip()
                if value:
                    ids.add(value)
    return ids if ids else None


def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--env_file", type=str, default=None)
    pre_args, _ = pre.parse_known_args()
    if pre_args.env_file:
        load_env_file(pre_args.env_file)

    parser = argparse.ArgumentParser(
        description="Run SWE-bench inference with an OpenAI-compatible API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env_file", type=str, default=None, help="Optional .env file")
    parser.add_argument(
        "--api_base",
        type=str,
        default=os.environ.get("VLLM_API_BASE", "http://127.0.0.1:8000/v1"),
        help="OpenAI-compatible API base URL",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=os.environ.get("VLLM_API_KEY") or os.environ.get("OPENAI_API_KEY"),
        help="API key (if required by the server)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("VLLM_MODEL", "minimax-nvfp4"),
        help="Model name served by the API",
    )
    parser.add_argument(
        "--prompt_dataset",
        type=str,
        default=os.environ.get("PROMPT_DATASET", "SWE-bench/SWE-bench_oracle"),
        help='Prompt dataset (must contain "text" field)',
    )
    parser.add_argument(
        "--prompt_split",
        type=str,
        default=os.environ.get("PROMPT_SPLIT", "test"),
        help="Prompt dataset split",
    )
    parser.add_argument(
        "--eval_dataset",
        type=str,
        default=os.environ.get("EVAL_DATASET"),
        help="Optional eval dataset for filtering instance IDs (e.g. SWE-bench/SWE-bench_Verified)",
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default=os.environ.get("EVAL_SPLIT", "test"),
        help="Eval dataset split",
    )
    parser.add_argument(
        "--instance_ids",
        nargs="+",
        type=str,
        default=None,
        help="Optional instance IDs to run (space separated)",
    )
    parser.add_argument(
        "--instance_ids_file",
        type=str,
        default=None,
        help="Optional file with instance IDs (one per line)",
    )
    parser.add_argument(
        "--strict_eval_ids",
        action="store_true",
        help="Error if prompt dataset is missing eval instance IDs",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.environ.get("OUTPUT_DIR", "preds"),
        help="Output directory",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=int(os.environ.get("RUNS", "1")),
        help="Number of runs (multi-run uses different seeds)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("WORKERS", "8")),
        help="Number of concurrent workers",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.environ.get("TEMPERATURE", "0.0")),
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=float(os.environ.get("TOP_P", "1.0")),
        help="Top-p sampling",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=int(os.environ.get("MAX_TOKENS", "4096")),
        help="Max tokens for generation (<=0 to omit)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None if os.environ.get("SEED") in {None, ""} else int(os.environ["SEED"]),
        help="Random seed (per-run seed is derived if runs>1)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.environ.get("TIMEOUT", "600")),
        help="Request timeout in seconds",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=int(os.environ.get("MAX_RETRIES", "3")),
        help="Max retries per request",
    )
    parser.add_argument(
        "--retry_backoff",
        type=float,
        default=float(os.environ.get("RETRY_BACKOFF", "2.0")),
        help="Exponential backoff base seconds",
    )
    parser.add_argument(
        "--model_args",
        type=str,
        default=os.environ.get("MODEL_ARGS"),
        help="Extra model args as comma-separated key=value",
    )
    parser.add_argument(
        "--max_instances",
        type=int,
        default=None,
        help="Optional cap on number of instances",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle prompts before generation",
    )
    parser.add_argument(
        "--on_error",
        type=str,
        choices=["empty", "skip", "raise"],
        default=os.environ.get("ON_ERROR", "empty"),
        help="How to handle generation errors",
    )
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        help="Resume from existing output file",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Do not resume from existing output",
    )
    parser.set_defaults(resume=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it exists",
    )
    parser.add_argument(
        "--strip_think",
        dest="strip_think",
        action="store_true",
        help="Strip <think> blocks from outputs",
    )
    parser.add_argument(
        "--no-strip-think",
        dest="strip_think",
        action="store_false",
        help="Do not strip <think> blocks",
    )
    parser.set_defaults(strip_think=os.environ.get("STRIP_THINK", "true").lower() != "false")
    parser.add_argument(
        "--save_full_output",
        dest="save_full_output",
        action="store_true",
        help="Save full model output in predictions",
    )
    parser.add_argument(
        "--no-save-full-output",
        dest="save_full_output",
        action="store_false",
        help="Do not save full model output in predictions",
    )
    parser.set_defaults(
        save_full_output=os.environ.get("SAVE_FULL_OUTPUT", "true").lower() != "false"
    )
    parser.add_argument(
        "--strict_patch",
        dest="strict_patch",
        action="store_true",
        help="Drop malformed patches that lack diff headers",
    )
    parser.add_argument(
        "--no-strict-patch",
        dest="strict_patch",
        action="store_false",
        help="Keep raw patch content even if diff headers are missing",
    )
    parser.set_defaults(
        strict_patch=os.environ.get("STRICT_PATCH", "true").lower() != "false"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_args = parse_model_args(args.model_args)
    max_tokens = None if args.max_tokens is not None and args.max_tokens <= 0 else args.max_tokens

    prompt_dataset_name = normalize_dataset_name(args.prompt_dataset)
    prompt_dataset = load_dataset_split(prompt_dataset_name, args.prompt_split)
    if "instance_id" not in prompt_dataset.column_names:
        raise ValueError(
            f'Prompt dataset "{prompt_dataset_name}" missing "instance_id" field.'
        )
    if "text" not in prompt_dataset.column_names:
        raise ValueError(
            f'Prompt dataset "{prompt_dataset_name}" missing "text" field. '
            "Use an inference dataset like SWE-bench_oracle or SWE-bench_bm25_*."
        )
    prompt_ids = set(prompt_dataset["instance_id"])

    eval_ids: Optional[set[str]] = None
    if args.eval_dataset:
        eval_dataset = load_dataset_split(args.eval_dataset, args.eval_split)
        if "instance_id" not in eval_dataset.column_names:
            raise ValueError(
                f'Eval dataset "{args.eval_dataset}" missing "instance_id" field.'
            )
        eval_ids = set(eval_dataset["instance_id"])
        missing = eval_ids - prompt_ids
        if missing:
            message = f"Prompt dataset missing {len(missing)} eval instance IDs."
            if args.strict_eval_ids:
                raise ValueError(message)
            LOG.warning(message)

    selected_ids = gather_instance_ids(args.instance_ids_file, args.instance_ids)
    if eval_ids is not None:
        selected_ids = eval_ids if selected_ids is None else (selected_ids & eval_ids)

    prompts: List[dict] = []
    for item in prompt_dataset:
        instance_id = item["instance_id"]
        if selected_ids is not None and instance_id not in selected_ids:
            continue
        prompts.append(item)

    if args.shuffle:
        random.shuffle(prompts)
    if args.max_instances is not None:
        prompts = prompts[: args.max_instances]

    LOG.info("Prepared %d prompts", len(prompts))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.output_file:
        safe_model = args.model.replace("/", "-")
        base = output_dir / f"{safe_model}-predictions.jsonl"
    else:
        base = Path(args.output_file)

    if args.runs > 1:
        LOG.info("Multi-run mode enabled (runs=%d).", args.runs)

    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    limits = httpx.Limits(
        max_connections=max(1, args.workers),
        max_keepalive_connections=max(1, args.workers),
    )

    for run_idx in range(1, args.runs + 1):
        output_path = make_output_path(str(base), run_idx, args.runs)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists() and args.overwrite:
            output_path.unlink()
        elif output_path.exists() and not args.resume:
            raise SystemExit(
                f"Output {output_path} exists. Use --resume or --overwrite."
            )

        existing_ids = read_existing_ids(output_path) if args.resume else set()
        remaining = [p for p in prompts if p["instance_id"] not in existing_ids]

        if args.runs > 1:
            if args.seed is None:
                run_seed = secrets.randbits(32)
            else:
                run_seed = args.seed + (run_idx - 1)
        else:
            run_seed = args.seed

        LOG.info(
            "Run %d/%d | seed=%s | output=%s | remaining=%d",
            run_idx,
            args.runs,
            run_seed,
            output_path,
            len(remaining),
        )

        if not remaining:
            continue

        def task(record: dict) -> Dict[str, Any]:
            system, user = split_system_user(record["text"])
            payload = build_payload(
                model=args.model,
                system=system,
                user=user,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=max_tokens,
                seed=run_seed,
                model_args=model_args,
            )
            data = post_with_retry(
                client,
                args.api_base,
                payload,
                args.timeout,
                args.max_retries,
                args.retry_backoff,
            )
            content = data["choices"][0]["message"]["content"]
            if args.strip_think:
                content = strip_think(content)
            patch = extract_diff(content)
            if args.strict_patch:
                patch = sanitize_patch(patch)
            result = {
                "instance_id": record["instance_id"],
                "model_name_or_path": args.model,
                "model_patch": patch,
            }
            if args.save_full_output:
                result["full_output"] = content
            return result

        error_count = 0
        with httpx.Client(headers=headers, limits=limits) as client:
            with open(output_path, "a", encoding="utf-8") as f:
                with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
                    future_to_record = {
                        executor.submit(task, record): record for record in remaining
                    }
                    with tqdm(
                        total=len(remaining),
                        desc=f"Generating (run {run_idx})",
                        ncols=100,
                    ) as pbar:
                        for future in as_completed(future_to_record):
                            record = future_to_record[future]
                            result: Optional[Dict[str, Any]] = None
                            try:
                                result = future.result()
                            except Exception as exc:
                                error_count += 1
                                LOG.warning(
                                    "Error on %s: %s",
                                    record["instance_id"],
                                    exc,
                                )
                                if args.on_error == "raise":
                                    raise
                                if args.on_error == "empty":
                                    result = {
                                        "instance_id": record["instance_id"],
                                        "model_name_or_path": args.model,
                                        "model_patch": "",
                                    }
                                    if args.save_full_output:
                                        result["full_output"] = ""
                            if result is not None:
                                f.write(json.dumps(result) + "\n")
                                f.flush()
                            pbar.update(1)

        if error_count:
            LOG.warning("Run %d finished with %d errors", run_idx, error_count)


if __name__ == "__main__":
    main()
