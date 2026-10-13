"""Cached seven-joint measurements and a pure command delay, outside the controllers."""

from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass

import numpy as np

from sarrl.robots.panda import vector7

NOISE_SEED_MASK = 0x50A07


@dataclass(frozen=True)
class PandaInterfaceConfig:
    position_noise_std_rad: float = 0.0
    velocity_noise_std_rad_s: float = 0.0
    command_delay_steps: int = 0

    def validate(self):
        scales = [self.position_noise_std_rad, self.velocity_noise_std_rad_s]
        if not np.all(np.isfinite(scales)) or np.any(np.asarray(scales) < 0.0):
            raise ValueError("Panda sensor noise scales must be finite and non-negative")
        if type(self.command_delay_steps) is not int or self.command_delay_steps < 0:
            raise ValueError("command delay must be a non-negative integer")


class PandaControlInterface:
    """One measurement per control boundary; only sent commands enter the FIFO."""

    def __init__(self, config: PandaInterfaceConfig, seed: int):
        config.validate()
        self.config = config
        self._rng = np.random.default_rng(seed ^ NOISE_SEED_MASK)
        self._scales = np.array(
            [config.position_noise_std_rad] * 7 + [config.velocity_noise_std_rad_s] * 7
        )
        self._queue = deque(np.zeros(7) for _ in range(config.command_delay_steps))
        self._sample_step = -1
        self._sample = None
        self._noise_hash = hashlib.sha256()
        self._noise_squared = np.zeros(14)
        self._difference_squared = np.zeros(7)
        self.measurements = self.commands = self.mismatched_commands = 0
        self.last_sent = self.last_delivered = None

    def measure(self, q, qd, step: int):
        if (
            type(step) is not int
            or step < 0
            or step not in (self._sample_step, self._sample_step + 1)
        ):
            raise ValueError(
                "measurements must use successive control boundaries, starting at zero"
            )
        if step != self._sample_step:
            state = np.concatenate([vector7(q, "q"), vector7(qd, "qd")])
            noise = self._rng.normal(size=14) * self._scales
            self._sample = state + noise
            self._sample_step = step
            self._noise_hash.update(noise.astype("<f8").tobytes())
            self._noise_squared += noise**2
            self.measurements += 1
        return self._sample[:7].copy(), self._sample[7:].copy()

    def deliver(self, command, torque_limit):
        limit = vector7(torque_limit, "torque_limit")
        if np.any(limit <= 0.0):
            raise ValueError("torque limits must be positive")
        self.last_sent = np.clip(vector7(command, "command"), -limit, limit)
        self._queue.append(self.last_sent.copy())
        self.last_delivered = self._queue.popleft()
        difference = self.last_sent - self.last_delivered
        self.commands += 1
        self.mismatched_commands += int(np.any(difference != 0.0))
        self._difference_squared += difference**2
        return self.last_delivered.copy()

    def diagnostics(self):
        return {
            "measurement_samples": self.measurements,
            "measurement_noise_sha256": self._noise_hash.hexdigest(),
            "measurement_noise_rms": np.sqrt(
                self._noise_squared / max(self.measurements, 1)
            ).tolist(),
            "actuation_mismatch_steps": self.mismatched_commands,
            "sent_delivered_rms_nm": np.sqrt(
                self._difference_squared / max(self.commands, 1)
            ).tolist(),
        }
