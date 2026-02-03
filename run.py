#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple


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


def pred_file_for_run(base: str, idx: int, runs: int) -> str:
    if runs <= 1:
        return base
    stem = base[:-5] if base.endswith(".jsonl") else base
    if stem.endswith("-predictions"):
        prefix = stem[: -len("-predictions")]
        return f"{prefix}-run{idx}-predictions.jsonl"
    return f"{stem}-run{idx}.jsonl"


def discover_latest_run_root(base_dir: Path, safe_model: str) -> Optional[Path]:
    candidates = sorted(
        base_dir.glob(f"{safe_model}__*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def discover_prediction_files(
    predictions_path: str,
    output_dir: Path,
    safe_model: str,
    run_id: Optional[str],
) -> Tuple[List[str], Optional[Path]]:
    if predictions_path:
        path = Path(predictions_path)
        if path.is_dir():
            files = sorted(path.glob("*.jsonl"))
            return [str(p) for p in files], path
        return [str(path)], path.parent

    if run_id:
        run_root = output_dir / f"{safe_model}__{run_id}"
        if run_root.exists():
            files = sorted(run_root.glob("*-run*-predictions.jsonl"))
            if files:
                return [str(p) for p in files], run_root
            single = run_root / f"{safe_model}-predictions.jsonl"
            if single.exists():
                return [str(single)], run_root

    latest = discover_latest_run_root(output_dir, safe_model)
    if latest:
        files = sorted(latest.glob("*-run*-predictions.jsonl"))
        if files:
            return [str(p) for p in files], latest
        single = latest / f"{safe_model}-predictions.jsonl"
        if single.exists():
            return [str(single)], latest

    fallback = output_dir / f"{safe_model}-predictions.jsonl"
    if fallback.exists():
        return [str(fallback)], output_dir
    return [], None


def derive_run_id(base_run_id: str, pred_file: str, idx: int, total: int) -> str:
    if total <= 1:
        return base_run_id
    name = Path(pred_file).stem
    if "-run" in name:
        suffix = name.split("-run")[-1]
        if suffix.isdigit():
            return f"{base_run_id}-run{suffix}"
    return f"{base_run_id}-run{idx}"


def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--env-file", type=str, default=None)
    pre_args, _ = pre.parse_known_args()
    if pre_args.env_file:
        load_env_file(pre_args.env_file)

    parser = argparse.ArgumentParser(
        description="Run SWE-bench inference + evaluation + plotting.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env-file", type=str, default=None, help="Optional .env file")

    parser.add_argument("--pred", action="store_true", help="Run prediction stage")
    parser.add_argument("--eval", action="store_true", help="Run evaluation stage")
    parser.add_argument("--plot", action="store_true", help="Run plotting stage")

    # Inference
    parser.add_argument(
        "--api-base",
        type=str,
        default=os.environ.get("API_BASE") or os.environ.get("VLLM_API_BASE") or "http://127.0.0.1:8000/v1",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("API_KEY") or os.environ.get("VLLM_API_KEY") or "",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("MODEL_NAME") or os.environ.get("VLLM_MODEL") or "minimax-nvfp4",
    )
    parser.add_argument("--prompt-dataset", type=str, default=os.environ.get("PROMPT_DATASET", "princeton-nlp/SWE-bench_oracle"))
    parser.add_argument("--prompt-split", type=str, default=os.environ.get("PROMPT_SPLIT", "test"))
    parser.add_argument("--eval-dataset", type=str, default=os.environ.get("EVAL_DATASET", "SWE-bench/SWE-bench_Verified"))
    parser.add_argument("--eval-split", type=str, default=os.environ.get("EVAL_SPLIT", "test"))
    parser.add_argument("--output-dir", type=str, default=os.environ.get("OUTPUT_DIR", "preds"))
    parser.add_argument("--output-file", type=str, default=os.environ.get("OUTPUT_FILE", ""))
    parser.add_argument("-N", "--runs", type=int, default=int(os.environ.get("RUNS", "1")))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("WORKERS", "8")))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("TEMPERATURE", "0.0")))
    parser.add_argument("--top-p", type=float, default=float(os.environ.get("TOP_P", "1.0")))
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("MAX_TOKENS", "4096")))
    parser.add_argument("--seed", type=str, default=os.environ.get("SEED", ""))
    parser.add_argument("--timeout", type=float, default=float(os.environ.get("TIMEOUT", "600")))
    parser.add_argument("--max-retries", type=int, default=int(os.environ.get("MAX_RETRIES", "0")))
    parser.add_argument("--retry-backoff", type=float, default=float(os.environ.get("RETRY_BACKOFF", "2.0")))
    parser.add_argument("--on-error", type=str, default=os.environ.get("ON_ERROR", "empty"))
    parser.add_argument("--resume", action="store_true", default=os.environ.get("RESUME", "true").lower() != "false")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--save-full-output", action="store_true", default=os.environ.get("SAVE_FULL_OUTPUT", "true").lower() != "false")
    parser.add_argument("--no-save-full-output", dest="save_full_output", action="store_false")
    parser.add_argument("--model-args", type=str, default=os.environ.get("MODEL_ARGS", ""))
    parser.add_argument("--max-instances", type=str, default=os.environ.get("MAX_INSTANCES", ""))
    parser.add_argument("--shuffle", action="store_true", default=os.environ.get("SHUFFLE", "false").lower() == "true")
    parser.add_argument("--instance-ids", nargs="+", type=str, default=None)
    parser.add_argument("--instance-ids-file", type=str, default=os.environ.get("INSTANCE_IDS_FILE", ""))
    parser.add_argument("--strip-think", action="store_true", default=os.environ.get("STRIP_THINK", "true").lower() != "false")
    parser.add_argument("--no-strip-think", dest="strip_think", action="store_false")
    parser.add_argument("--strict-patch", action="store_true", default=os.environ.get("STRICT_PATCH", "false").lower() == "true")
    parser.add_argument("--no-strict-patch", dest="strict_patch", action="store_false")

    # Evaluation
    parser.add_argument("--predictions-path", type=str, default=os.environ.get("PREDICTIONS_PATH", ""))
    parser.add_argument("--max-workers", type=int, default=int(os.environ.get("MAX_WORKERS", "4")))
    parser.add_argument("--open-file-limit", type=int, default=int(os.environ.get("OPEN_FILE_LIMIT", "4096")))
    parser.add_argument("--test-timeout", type=int, default=int(os.environ.get("TEST_TIMEOUT", "1800")))
    parser.add_argument("--force-rebuild", type=str, default=os.environ.get("FORCE_REBUILD", "false"))
    parser.add_argument("--cache-level", type=str, default=os.environ.get("CACHE_LEVEL", "env"))
    parser.add_argument("--clean", type=str, default=os.environ.get("CLEAN", "false"))
    parser.add_argument("--run-id", type=str, default=os.environ.get("RUN_ID", ""))
    parser.add_argument("--namespace", type=str, default=os.environ.get("NAMESPACE", "swebench"))
    parser.add_argument("--instance-image-tag", type=str, default=os.environ.get("INSTANCE_IMAGE_TAG", "latest"))
    parser.add_argument("--env-image-tag", type=str, default=os.environ.get("ENV_IMAGE_TAG", "latest"))
    parser.add_argument("--rewrite-reports", type=str, default=os.environ.get("REWRITE_REPORTS", "false"))
    parser.add_argument("--modal", type=str, default=os.environ.get("MODAL", "false"))

    # Plot
    parser.add_argument("--plot-dir", type=str, default=os.environ.get("PLOT_DIR", "plots"))
    parser.add_argument("--reports", nargs="+", type=str, default=None)

    return parser.parse_args()


def build_inference_cmd(args: argparse.Namespace, output_file: str) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        "swebench.inference.run_openai_compatible",
        "--api_base",
        args.api_base,
        "--model",
        args.model,
        "--prompt_dataset",
        args.prompt_dataset,
        "--prompt_split",
        args.prompt_split,
        "--eval_dataset",
        args.eval_dataset,
        "--eval_split",
        args.eval_split,
        "--output_dir",
        args.output_dir,
        "--output_file",
        output_file,
        "--runs",
        str(args.runs),
        "--workers",
        str(args.workers),
        "--temperature",
        str(args.temperature),
        "--top_p",
        str(args.top_p),
        "--max_tokens",
        str(args.max_tokens),
        "--timeout",
        str(args.timeout),
        "--max_retries",
        str(args.max_retries),
        "--retry_backoff",
        str(args.retry_backoff),
        "--on_error",
        args.on_error,
    ]
    if args.api_key:
        cmd += ["--api_key", args.api_key]
    if args.seed:
        cmd += ["--seed", str(args.seed)]
    if args.resume:
        cmd += ["--resume"]
    else:
        cmd += ["--no-resume"]
    if args.save_full_output:
        cmd += ["--save_full_output"]
    else:
        cmd += ["--no-save-full-output"]
    if args.model_args:
        cmd += ["--model_args", args.model_args]
    if args.max_instances:
        cmd += ["--max_instances", str(args.max_instances)]
    if args.shuffle:
        cmd += ["--shuffle"]
    if args.instance_ids:
        cmd += ["--instance_ids", *args.instance_ids]
    if args.instance_ids_file:
        cmd += ["--instance_ids_file", args.instance_ids_file]
    if args.strip_think:
        cmd += ["--strip_think"]
    else:
        cmd += ["--no-strip-think"]
    if args.strict_patch:
        cmd += ["--strict_patch"]
    else:
        cmd += ["--no-strict-patch"]
    return cmd


def build_eval_cmd(args: argparse.Namespace, predictions_path: str, run_id: str) -> List[str]:
    predictions_arg = predictions_path
    if predictions_path != "gold":
        predictions_arg = str(Path(predictions_path).resolve())
    cmd = [
        sys.executable,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        args.eval_dataset,
        "--split",
        args.eval_split,
        "--predictions_path",
        predictions_arg,
        "--max_workers",
        str(args.max_workers),
        "--open_file_limit",
        str(args.open_file_limit),
        "--timeout",
        str(args.test_timeout),
        "--force_rebuild",
        str(args.force_rebuild),
        "--cache_level",
        args.cache_level,
        "--clean",
        str(args.clean),
        "--run_id",
        run_id,
        "--namespace",
        args.namespace,
        "--instance_image_tag",
        args.instance_image_tag,
        "--env_image_tag",
        args.env_image_tag,
        "--rewrite_reports",
        str(args.rewrite_reports),
        "--modal",
        str(args.modal),
    ]
    if args.instance_ids:
        cmd += ["--instance_ids", *args.instance_ids]
    return cmd


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    subprocess_env = os.environ.copy()
    existing_pythonpath = subprocess_env.get("PYTHONPATH", "")
    subprocess_env["PYTHONPATH"] = (
        str(repo_root)
        if not existing_pythonpath
        else f"{repo_root}{os.pathsep}{existing_pythonpath}"
    )
    stages_selected = args.pred or args.eval or args.plot
    run_pred = args.pred or not stages_selected
    run_eval = args.eval or not stages_selected
    run_plot = args.plot or not stages_selected

    safe_model = args.model.replace("/", "-")
    model_tag = args.model.replace("/", "__")
    if not args.run_id and run_pred:
        args.run_id = f"{safe_model}-eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    base_output_dir = Path(args.output_dir)
    run_root: Optional[Path] = None
    if run_pred:
        if not args.run_id:
            raise SystemExit("run-id is required for prediction stage.")
        if args.runs > 1:
            run_root = base_output_dir / f"{safe_model}__{args.run_id}"
            run_root.mkdir(parents=True, exist_ok=True)
            output_dir = run_root
        else:
            output_dir = base_output_dir
            output_dir.mkdir(parents=True, exist_ok=True)
        args.output_dir = str(output_dir)
    else:
        output_dir = base_output_dir

    if args.output_file:
        output_name = Path(args.output_file).name
        output_file = str((output_dir / output_name) if args.runs > 1 else Path(args.output_file))
    else:
        output_file = str(output_dir / f"{safe_model}-predictions.jsonl")

    report_files: List[str] = []

    if run_pred:
        cmd = build_inference_cmd(args, output_file)
        subprocess.run(
            cmd,
            check=True,
            cwd=run_root if run_root else None,
            env=subprocess_env,
        )

    if run_eval:
        if not args.run_id:
            if args.predictions_path:
                args.run_id = f"{safe_model}-eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            else:
                latest = discover_latest_run_root(output_dir, safe_model)
                if latest:
                    args.run_id = latest.name.split("__", 1)[-1]
                    run_root = latest
                else:
                    raise SystemExit(
                        "run-id not set and no run folder found. "
                        "Pass --run-id or --predictions-path."
                    )

        pred_files, discovered_root = discover_prediction_files(
            args.predictions_path or "",
            output_dir,
            safe_model,
            args.run_id,
        )
        if discovered_root and not run_root:
            run_root = discovered_root
        if not pred_files:
            raise SystemExit("No prediction files found for evaluation.")

        total = len(pred_files)
        for idx, pred_file in enumerate(pred_files, start=1):
            run_id = derive_run_id(args.run_id, pred_file, idx, total)
            cmd = build_eval_cmd(args, pred_file, run_id)
            subprocess.run(
                cmd,
                check=True,
                cwd=run_root if run_root else None,
                env=subprocess_env,
            )
            report_name = f"{model_tag}.{run_id}.json"
            report_path = str((run_root / report_name) if run_root else Path(report_name))
            report_files.append(report_path)

    if run_plot:
        plot_dir = Path(args.plot_dir)
        if run_root:
            plot_dir = run_root / plot_dir
        plot_dir.mkdir(parents=True, exist_ok=True)
        reports = args.reports if args.reports else report_files
        if not reports:
            if not run_root:
                latest = discover_latest_run_root(output_dir, safe_model)
                if latest:
                    run_root = latest
            if run_root:
                reports = sorted(
                    str(p) for p in run_root.glob(f"{model_tag}.*.json")
                )
        if not reports:
            raise SystemExit("No report files provided or generated for plotting.")
        cmd = [
            sys.executable,
            "scripts/plot_run_report.py",
            "--reports",
            *reports,
            "--out_dir",
            str(plot_dir),
        ]
        subprocess.run(cmd, check=True, env=subprocess_env)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
