#!/usr/bin/env python3
"""Evaluate a fixed, exploratory residual-removal contrast with retained provenance."""

# Numerical thread limits must precede imports that initialize BLAS.
# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

for variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "1"

import torch

from sarrl.envs import PlanarReachEnv
from sarrl.evaluation import assert_repository_import_root, v13_scenarios
from sarrl.evaluation.provenance import runtime_metadata
from sarrl.evaluation.residual_diagnosis import (
    CASES,
    ZeroResidualPolicy,
    compare_historical,
    evaluate_recorded,
    paired_contrast,
)
from sarrl.rl.networks import SquashedGaussianActor

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / "results/penalty_ablation"
INVENTORY_HASH = "78f359c13e3146d5b383c522d00809149403af92ed06839895ccf24f47a1754f"
AUDIT_HASH = "133a8049918b2d955ec6dc9a6803fd816f63f9035dcd2219ff2320f1d1a0566b"


class FrozenCPUActor:
    """Load only a verified actor, with no critics, optimizers or automatic device choice."""

    def __init__(self, path):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (payload.get("checkpoint_version"), payload["obs_dim"], payload["action_dim"]) != (
            1,
            8,
            2,
        ):
            raise ValueError("unsupported diagnostic checkpoint")
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            self.actor = SquashedGaussianActor(8, 2, tuple(payload["config"]["hidden"]))
        self.actor.load_state_dict(payload["actor"], strict=True)
        self.actor.eval().requires_grad_(False)
        if any(p.device.type != "cpu" for p in self.actor.parameters()):
            raise ValueError("diagnostic actor must stay on CPU")

    def act(self, observation, deterministic=True):
        if not deterministic:
            raise ValueError("diagnostic inference must be deterministic")
        with torch.no_grad():
            action = self.actor.deterministic(
                torch.as_tensor(observation, dtype=torch.float32, device="cpu").unsqueeze(0)
            )
        return action.squeeze(0).numpy()


class FrozenCUDAActor(FrozenCPUActor):
    """Use the retained CPU-load/CUDA-transfer path with explicit float32 inference."""

    def __init__(self, path):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for diagnostic actor inference")
        super().__init__(path)
        self.actor.to("cuda:0")
        self.validate_device()

    def validate_device(self):
        tensors = list(self.actor.parameters()) + list(self.actor.buffers())
        if not tensors or any(
            p.device != torch.device("cuda:0") or p.dtype != torch.float32 for p in tensors
        ):
            raise ValueError("diagnostic actor must use CUDA:0 and float32")

    def act(self, observation, deterministic=True):
        if not deterministic:
            raise ValueError("diagnostic inference must be deterministic")
        self.validate_device()
        with torch.no_grad():
            action = self.actor.deterministic(
                torch.as_tensor(observation, dtype=torch.float32, device="cuda:0").unsqueeze(0)
            )
        return action.squeeze(0).cpu().numpy()


def execution_metadata():
    """Inspect the effective runtime without creating an environment or changing precision."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for diagnostic actor inference")
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise RuntimeError("diagnostic inference requires one intra-op and inter-op thread")
    return {
        "actor_device": "cuda:0",
        "plant_device": "cpu",
        "filter_device": "cpu",
        "actor_dtype": "float32",
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "intraop_threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "thread_environment": {
            k: os.environ[k]
            for k in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, payload):
    """Publish one new JSON file; never replace retained evidence."""
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()


@contextmanager
def exclusive_run(output):
    # The lock survives process death only as an empty file; kernel ownership does not.
    if sys.platform != "linux":
        raise RuntimeError("run the diagnostic runner in the retained Linux environment")
    import fcntl

    output.mkdir(parents=True, exist_ok=True)
    with (output / ".run.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another diagnostic process owns this output") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def build_plan(protocol, amendment):
    execution = execution_metadata()
    assert_repository_import_root(ROOT)
    if sha(OFFICIAL / "checkpoint_inventory.json") != INVENTORY_HASH:
        raise ValueError("frozen checkpoint inventory changed")
    history_path = OFFICIAL / "failure_audit/trajectories.json"
    if sha(history_path) != AUDIT_HASH:
        raise ValueError("retained historical trajectories changed")
    inventory = read(OFFICIAL / "checkpoint_inventory.json")
    records = inventory["records"]
    expected = {
        (c, s) for c in ("P0_inloop_reference", "P1_inloop_half_penalty") for s in range(30, 35)
    }
    if len(records) != 10 or {(r["condition"], r["training_seed"]) for r in records} != expected:
        raise ValueError("expected the ten frozen models")
    models = []
    for record in records:
        path = OFFICIAL / "training" / record["condition"] / f"seed_{record['training_seed']}"
        checkpoint = path / "best.pt"
        if sha(checkpoint) != record["hashes"]["best.pt"]:
            raise ValueError(f"checkpoint hash mismatch: {checkpoint}")
        models.append(
            {
                "condition": record["condition"],
                "training_seed": record["training_seed"],
                "path": str(checkpoint.relative_to(ROOT)),
                "sha256": sha(checkpoint),
            }
        )
    sources = sorted((ROOT / "sarrl").rglob("*.py")) + [Path(__file__), ROOT / "pyproject.toml"]
    return {
        "schema_version": 2,
        "exploratory": True,
        "training": False,
        "episodes": 66,
        "max_physical_steps": 16500,
        "workers": 1,
        "execution": execution,
        "cases": {k: list(v) for k, v in CASES.items()},
        "models": models,
        "inventory_sha256": INVENTORY_HASH,
        "history_sha256": AUDIT_HASH,
        "protocol_sha256": sha(protocol),
        "amendment_sha256": sha(amendment),
        "source_hashes": {str(p.relative_to(ROOT)): sha(p) for p in sources},
        "runtime": runtime_metadata(ROOT),
    }


def cell_specs(plan):
    for scenario, seeds in plan["cases"].items():
        for seed in seeds:
            yield "Z", scenario, seed, None
            for model in plan["models"]:
                label = f"{model['condition']}_train_seed_{model['training_seed']}"
                yield label, scenario, seed, model


def validate_output_path(output):
    output = output.resolve()
    result_root = (ROOT / "results").resolve()
    if output.parent != result_root or not output.name.startswith("residual_diagnosis_"):
        raise ValueError("output must be a new results/residual_diagnosis_* directory")
    return output


def execute(plan, output, protocol, amendment):
    if execution_metadata() != plan["execution"]:
        raise ValueError("execution device/precision differs from preflight")
    for path, key in ((protocol, "protocol_sha256"), (amendment, "amendment_sha256")):
        if sha(path) != plan[key]:
            raise ValueError("protocol or amendment changed since preflight")
    output = validate_output_path(output)
    with exclusive_run(output):
        if (output / "complete.json").exists():
            raise FileExistsError("diagnosis already complete; use its retained results")
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            if read(manifest_path) != plan:
                raise ValueError("resume manifest differs from current sources/runtime/protocol")
        else:
            if set(p.name for p in output.iterdir()) - {".run.lock"}:
                raise ValueError("output is not empty and has no manifest")
            write_new(manifest_path, plan)
        for source, name, key in (
            (protocol, "protocol.md", "protocol_sha256"),
            (amendment, "amendment.md", "amendment_sha256"),
        ):
            frozen = output / name
            if not frozen.exists():
                with frozen.open("xb") as stream:
                    stream.write(source.read_bytes())
            if sha(frozen) != plan[key]:
                raise ValueError(f"retained {name} differs")
        history_rows = read(OFFICIAL / "failure_audit/trajectories.json")
        history = {
            (r["episode"]["controller"], r["episode"]["scenario"], r["episode"]["seed"]): r
            for r in history_rows
        }
        if len(history_rows) != len(history) or len(history) != 60:
            raise ValueError("historical keys are not the 60 unique cases")
        scenario_specs = {s.key: s for s in v13_scenarios()}
        completed = []
        for controller, scenario, seed, model in cell_specs(plan):
            key = (controller, scenario, seed)
            name = f"{controller}__{scenario}__{seed}"
            data_path, marker = output / f"{name}.json", output / f"{name}.complete.json"
            started = output / f"{name}.started.json"
            invalid = output / f"{name}.invalid.json"
            if invalid.exists() or invalid.with_suffix(".json.tmp").exists():
                raise ValueError(f"invalid retained cell needs inspection: {name}")
            if marker.exists():
                completion = read(marker)
                if completion != {"sha256": sha(data_path), "manifest_sha256": sha(manifest_path)}:
                    raise ValueError(f"completed cell hash mismatch: {name}")
                result = read(data_path)
            else:
                # An orphaned output or interrupted temporary file requires an explicit audit.
                if (
                    data_path.exists()
                    or data_path.with_suffix(".json.tmp").exists()
                    or started.exists()
                    or started.with_suffix(".json.tmp").exists()
                ):
                    raise ValueError(f"incomplete retained cell needs inspection: {name}")
                write_new(
                    started,
                    {
                        "key": list(key),
                        "status": "started",
                        "manifest_sha256": sha(manifest_path),
                        "reserved_max_physical_steps": 250,
                    },
                )
                result = None
                try:
                    specification = scenario_specs[scenario]
                    if model is not None and sha(ROOT / model["path"]) != model["sha256"]:
                        raise ValueError("checkpoint changed since preflight")
                    policy = (
                        ZeroResidualPolicy()
                        if model is None
                        else FrozenCUDAActor(ROOT / model["path"])
                    )
                    if model is not None:
                        policy.validate_device()
                    env = PlanarReachEnv(
                        mode="torque",
                        randomization=specification.randomization,
                        fault=specification.fault,
                    )
                    result = evaluate_recorded(
                        policy, env, seed=seed, scenario=scenario, controller=controller
                    )
                    if model is not None:
                        compare_historical(result, history[key])
                    result["identity"] = model
                    write_new(data_path, result)
                    write_new(
                        marker, {"sha256": sha(data_path), "manifest_sha256": sha(manifest_path)}
                    )
                except Exception as exc:
                    write_new(
                        invalid,
                        {
                            "key": list(key),
                            "status": "invalid",
                            "error": str(exc),
                            "result": result,
                            "partial": getattr(exc, "partial", None),
                        },
                    )
                    with (output / "errors.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(
                            json.dumps({"key": key, "status": "invalid", "error": str(exc)}) + "\n"
                        )
                    raise
                print(f"completed {name}", flush=True)
            if (
                result["status"] != "valid"
                or tuple(result["episode"][k] for k in ("controller", "scenario", "seed")) != key
            ):
                raise ValueError(f"retained cell identity/status mismatch: {name}")
            if model is not None:
                compare_historical(result, history[key])
            completed.append(result)
        zero = {
            (r["episode"]["scenario"], r["episode"]["seed"]): r
            for r in completed
            if r["episode"]["controller"] == "Z"
        }
        contrasts = [
            paired_contrast(r, zero[(r["episode"]["scenario"], r["episode"]["seed"])])
            for r in completed
            if r["episode"]["controller"] != "Z"
        ]
        if len(completed) != 66 or len(zero) != 6 or len(contrasts) != 60:
            raise ValueError("incomplete diagnostic matrix")
        if sum(r["episode"]["steps"] for r in completed) > 16500:
            raise ValueError("diagnostic budget exceeded")
        payloads = {
            "contrasts.json": contrasts,
            "summary.json": [
                {
                    "episode": r["episode"],
                    "safety": r["safety"],
                    "summary": r["summary"],
                    "identity": r["identity"],
                }
                for r in completed
            ],
        }
        for name, payload in payloads.items():
            path = output / name
            if path.exists():
                if read(path) != payload:
                    raise ValueError(f"retained aggregate differs: {name}")
            else:
                write_new(path, payload)
        write_new(
            output / "complete.json",
            {
                "episodes": 66,
                "shared_controls": 6,
                "contrasts": 60,
                "hashes": {
                    p.name: sha(p)
                    for p in sorted(output.iterdir())
                    if p.is_file() and p.name != ".run.lock"
                },
            },
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run"))
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="explicitly execute the 66 diagnostic episodes; no training",
    )
    args = parser.parse_args(argv)
    if args.command == "run" and (not args.execute or args.output is None):
        parser.error("run requires --execute and --output")
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    plan = build_plan(args.protocol, args.amendment)
    if args.command == "plan":
        print(json.dumps(plan, indent=2, sort_keys=True))
    else:
        execute(plan, args.output, args.protocol, args.amendment)


if __name__ == "__main__":
    main()
