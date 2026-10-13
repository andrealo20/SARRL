"""Timestamped torque transport with scheduled arrivals and stale-packet rejection."""

from __future__ import annotations

import hashlib
from functools import lru_cache

import numpy as np

from sarrl.envs.panda_interface import PandaControlInterface
from sarrl.robots.panda import vector7

PROFILES = {"fixed10": (1, 1), "fixed40": (4, 4), "up": (1, 4), "down": (4, 1)}
SWITCH_STEP = 100


def send_delay(profile, step):
    if profile not in PROFILES:
        raise ValueError(f"unknown packet-delay profile: {profile}")
    if type(step) is not int:
        raise ValueError("packet step must be an integer")
    before, after = PROFILES[profile]
    return before if step < SWITCH_STEP else after


@lru_cache(maxsize=16)
def applied_ages(profile, horizon=300):
    if type(horizon) is not int or horizon < 1:
        raise ValueError("packet horizon must be a positive integer")
    # Independent timestamp enumeration, including the four known prior sends.
    return tuple(
        k - max(i for i in range(-4, k + 1) if i + send_delay(profile, i) <= k)
        for k in range(horizon)
    )


class PacketDelayQueue:
    def __init__(self, profile, history, held=None):
        initial = send_delay(profile, -1)
        history = np.asarray(history, dtype=float)
        if history.shape != (4, 7) or not np.all(np.isfinite(history)):
            raise ValueError("packet startup needs four finite seven-joint sent commands")
        self.profile, self.history = profile, history.copy()
        self.pending = [
            (i + initial, i, command.copy())
            for i, command in zip(range(-4, 0), history, strict=True)
            if i + initial >= 0
        ]
        self.held = np.zeros(7) if held is None else vector7(held, "held command").copy()
        self.last_index = -initial - 1
        self.step = self.discarded = 0

    def deliver(self, command):
        command = vector7(command, "packet command")
        k = self.step
        delay = send_delay(self.profile, k)
        self.pending.append((k + delay, k, command.copy()))
        ready = [item for item in self.pending if item[0] <= k]
        self.pending = [item for item in self.pending if item[0] > k]
        candidates = [item for item in ready if item[1] > self.last_index]
        accepted = max(candidates, key=lambda item: item[1]) if candidates else None
        discarded = len(ready) - int(accepted is not None)
        self.discarded += discarded
        if accepted is not None:
            _, self.last_index, self.held = accepted
        metadata = {
            "sent_packet_delay_steps": delay,
            "delivered_command_index": self.last_index,
            "applied_age_steps": k - self.last_index,
            "discarded_packets": discarded,
            "pending_packets": len(self.pending),
        }
        self.step += 1
        return self.held.copy(), metadata


class PandaPacketInterface(PandaControlInterface):
    def __init__(self, config, seed, profile, history, held=None):
        if config.command_delay_steps != send_delay(profile, -1):
            raise ValueError("packet interface config must record the initial delay")
        super().__init__(config, seed)
        self.transport = PacketDelayQueue(profile, history, held)
        self.delivery_trace = []

    def deliver(self, command, torque_limit):
        limit = vector7(torque_limit, "torque_limit")
        if np.any(limit <= 0.0):
            raise ValueError("torque limits must be positive")
        if self.commands == 0 and (
            np.any(np.abs(self.transport.history) > limit)
            or np.any(np.abs(self.transport.held) > limit)
        ):
            raise ValueError("packet startup commands must already be clipped")
        self.last_sent = np.clip(vector7(command, "command"), -limit, limit)
        self.last_delivered, metadata = self.transport.deliver(self.last_sent)
        self.delivery_trace.append(metadata)
        difference = self.last_sent - self.last_delivered
        self.commands += 1
        self.mismatched_commands += int(np.any(difference != 0.0))
        self._difference_squared += difference**2
        return self.last_delivered.copy()

    def diagnostics(self):
        ages = [item["applied_age_steps"] for item in self.delivery_trace]
        return {
            **super().diagnostics(),
            "transport_kind": "scheduled arrivals, latest timestamp wins, hold between arrivals",
            "applied_age_sha256": hashlib.sha256(np.array(ages, dtype="<i8").tobytes()).hexdigest(),
            "discarded_packets": self.transport.discarded,
        }
