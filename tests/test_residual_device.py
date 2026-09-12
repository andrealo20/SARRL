"""Device contract checks using synthetic tensors, with physical operations forbidden."""

from copy import deepcopy

import numpy as np
import pytest
import torch

from sarrl.dynamics import PlanarArm
from sarrl.envs import PlanarReachEnv
from sarrl.rl.networks import SquashedGaussianActor
from tools import run_residual_diagnosis as runner


@pytest.fixture(autouse=True)
def forbid_physics(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("no reset or physical step is permitted in device tests")

    monkeypatch.setattr(PlanarReachEnv, "reset", forbidden)
    monkeypatch.setattr(PlanarReachEnv, "step_torque", forbidden)
    monkeypatch.setattr(PlanarArm, "step_rk4", forbidden)


def test_no_cuda_fails_before_checkpoint_plan_or_output(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    missing = tmp_path / "missing"
    with pytest.raises(RuntimeError, match="CUDA is required"):
        runner.FrozenCUDAActor(missing)
    with pytest.raises(RuntimeError, match="CUDA is required"):
        runner.build_plan(missing, missing)
    with pytest.raises(RuntimeError, match="CUDA is required"):
        runner.execute({}, tmp_path / "output", missing, missing)
    assert not (tmp_path / "output").exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_cuda_actor_matches_retained_transfer_path(tmp_path, monkeypatch):
    actor = SquashedGaussianActor(8, 2, (8, 8))
    path = tmp_path / "synthetic.pt"
    torch.save(
        {
            "checkpoint_version": 1,
            "obs_dim": 8,
            "action_dim": 2,
            "config": {"hidden": [8, 8]},
            "actor": actor.state_dict(),
        },
        path,
    )
    monkeypatch.setattr(torch.optim, "Adam", lambda *a, **kw: pytest.fail("optimizer created"))
    cpu = runner.FrozenCPUActor(path)
    reference = deepcopy(cpu.actor).to("cuda:0")
    cpu_rng = torch.get_rng_state().clone()
    gpu_rng = torch.cuda.get_rng_state(0).clone()
    policy = runner.FrozenCUDAActor(path)
    observed_inputs = []
    handle = policy.actor.trunk.register_forward_pre_hook(
        lambda module, inputs: observed_inputs.append((inputs[0].device, inputs[0].dtype))
    )
    for observation in (np.zeros(8), np.linspace(-2, 2, 8), np.ones(8)):
        with torch.no_grad():
            expected = (
                reference.deterministic(
                    torch.as_tensor(observation, dtype=torch.float32, device="cuda:0").unsqueeze(0)
                )[0]
                .cpu()
                .numpy()
            )
        actual = policy.act(observation)
        assert actual.dtype == np.float32
        assert np.array_equal(actual, expected)
    handle.remove()
    assert observed_inputs == [(torch.device("cuda:0"), torch.float32)] * 3
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert torch.equal(gpu_rng, torch.cuda.get_rng_state(0))
    assert all(not p.requires_grad for p in policy.actor.parameters())
    with pytest.raises(ValueError, match="deterministic"):
        policy.act(np.zeros(8), deterministic=False)
    policy.actor.cpu()
    with pytest.raises(ValueError, match="CUDA:0 and float32"):
        policy.act(np.zeros(8))
    policy.actor.to(device="cuda:0", dtype=torch.float64)
    with pytest.raises(ValueError, match="CUDA:0 and float32"):
        policy.validate_device()


def test_metadata_records_effective_device_precision_threads(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "synthetic GPU")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: (8, 9))
    monkeypatch.setattr(torch, "get_num_threads", lambda: 1)
    monkeypatch.setattr(torch, "get_num_interop_threads", lambda: 1)
    info = runner.execution_metadata()
    assert info["actor_device"] == "cuda:0"
    assert info["plant_device"] == info["filter_device"] == "cpu"
    assert info["actor_dtype"] == "float32"
    assert info["gpu_capability"] == [8, 9]
    assert info["torch"] == torch.__version__
    assert info["cuda_runtime"] == torch.version.cuda
    assert info["float32_matmul_precision"] == torch.get_float32_matmul_precision()
    assert info["cuda_matmul_allow_tf32"] == torch.backends.cuda.matmul.allow_tf32
    assert info["cudnn_allow_tf32"] == torch.backends.cudnn.allow_tf32
    assert set(info["thread_environment"].values()) == {"1"}
    monkeypatch.setattr(torch, "get_num_interop_threads", lambda: 2)
    with pytest.raises(RuntimeError, match="one intra-op and inter-op"):
        runner.execution_metadata()


@pytest.mark.parametrize(
    "field", ["gpu_name", "float32_matmul_precision", "actor_device", "interop_threads"]
)
def test_changed_runtime_blocks_resume_before_output(tmp_path, monkeypatch, field):
    saved = {field: "before"}
    monkeypatch.setattr(runner, "execution_metadata", lambda: {field: "after"})
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="differs from preflight"):
        runner.execute({"execution": saved}, output, tmp_path / "missing", tmp_path / "missing")
    assert not output.exists()


def test_changed_amendment_blocks_before_output(tmp_path, monkeypatch):
    protocol, amendment = tmp_path / "protocol.md", tmp_path / "amendment.md"
    protocol.write_text("synthetic protocol")
    amendment.write_text("synthetic amendment")
    plan = {
        "execution": {},
        "protocol_sha256": runner.sha(protocol),
        "amendment_sha256": runner.sha(amendment),
    }
    monkeypatch.setattr(runner, "execution_metadata", lambda: {})
    amendment.write_text("changed amendment")
    with pytest.raises(ValueError, match="changed since preflight"):
        runner.execute(plan, tmp_path / "output", protocol, amendment)
    assert not (tmp_path / "output").exists()
