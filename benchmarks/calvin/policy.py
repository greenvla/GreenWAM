# SPDX-License-Identifier: OpenMDW-1.1
"""CALVIN model wrapper that queries a GreenWAM policy server.

The server answers one request with a chunk of ``prediction_horizon`` actions; the wrapper executes the first
``execution_horizon`` of them, one per ``step``, and asks again when the chunk is used up. Each action row is
``(x, y, z, roll, pitch, yaw, gripper, ...)`` in the world frame: the first seven values become an absolute TCP target
for ``calvin_env`` and the gripper value is thresholded at zero to the {-1, 1} command the simulator expects.

Wire format: one msgpack map per request and per reply over a ZMQ REQ socket; numpy arrays travel as
``{"__ndarray__": True, "data": bytes, "dtype": str, "shape": list}``.
"""

from __future__ import annotations

from typing import Any

import msgpack
import numpy as np
import zmq
from calvin_agent.models.calvin_base_model import CalvinBaseModel

_NDARRAY_KEY = "__ndarray__"
_GRIPPER_INDEX = 6
_ACTION_DIM = 7


def _pack_ndarray(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return {_NDARRAY_KEY: True, "data": obj.tobytes(), "dtype": obj.dtype.str, "shape": obj.shape}
    raise TypeError(f"cannot serialize {type(obj)}")


def _unpack_ndarray(obj: Any) -> Any:
    if isinstance(obj, dict) and obj.get(_NDARRAY_KEY):
        return np.frombuffer(obj["data"], dtype=np.dtype(obj["dtype"])).reshape(obj["shape"]).copy()
    return obj


class PolicyClient:
    """Synchronous request/reply client of the policy server."""

    def __init__(self, host: str, port: int, timeout_ms: int = 600_000) -> None:
        self._socket = zmq.Context.instance().socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(f"tcp://{host}:{port}")

    def infer(self, request: dict[str, Any]) -> dict[str, Any]:
        self._socket.send(msgpack.packb(request, default=_pack_ndarray, use_bin_type=True))
        reply = msgpack.unpackb(self._socket.recv(), raw=False, object_hook=_unpack_ndarray)
        if not isinstance(reply, dict):
            raise TypeError(f"expected a map from the server, got {type(reply)}")
        if reply.get("error") is not None:
            raise RuntimeError(str(reply["error"]))
        return reply


class ServerPolicy(CalvinBaseModel):
    def __init__(
        self,
        client: PolicyClient,
        prediction_horizon: int = 20,
        execution_horizon: int = 10,
        guidance: float = 1.5,
        video_text_cfg: float | None = 1.5,
        action_text_cfg: float | None = 1.5,
        num_steps: int = 4,
        shift: float = 3.0,
    ) -> None:
        if execution_horizon > prediction_horizon:
            raise ValueError("execution_horizon must not exceed prediction_horizon")
        self.client = client
        self.prediction_horizon = prediction_horizon
        self.execution_horizon = execution_horizon
        self.guidance = guidance
        self.video_text_cfg = video_text_cfg
        self.action_text_cfg = action_text_cfg
        self.num_steps = num_steps
        self.shift = shift
        self._chunk: list[np.ndarray] = []
        self.num_rollouts = 0
        self.num_steps_taken = 0

    def reset(self) -> None:
        self._chunk = []
        self.num_rollouts += 1

    def _request_chunk(self, obs: dict[str, Any], instruction: str) -> None:
        request: dict[str, Any] = {
            "prompt": instruction,
            "rgb_static": obs["rgb_obs"]["rgb_static"],
            "rgb_gripper": obs["rgb_obs"]["rgb_gripper"],
            "state": obs["robot_obs"],
            "prediction_horizon": self.prediction_horizon,
            "guidance": self.guidance,
            "num_steps": self.num_steps,
            "shift": self.shift,
        }
        if self.video_text_cfg is not None:
            request["video_text_cfg"] = self.video_text_cfg
        if self.action_text_cfg is not None:
            request["action_text_cfg"] = self.action_text_cfg
        reply = self.client.infer(request)
        actions = reply.get("actions", reply.get("action"))
        if actions is None:
            raise KeyError(f"server reply has no actions, keys={list(reply)}")
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[None, :]
        if actions.shape[0] < self.execution_horizon:
            raise ValueError(f"server returned {actions.shape[0]} actions, need {self.execution_horizon}")
        self._chunk = list(actions[: self.execution_horizon])

    def step(self, obs: dict[str, Any], goal: str) -> dict[str, Any]:
        if not self._chunk:
            self._request_chunk(obs, goal)
        row = self._chunk.pop(0)
        row[_GRIPPER_INDEX] = 1 if row[_GRIPPER_INDEX] >= 0 else -1
        self.num_steps_taken += 1
        return {"type": "cartesian_abs", "action": np.asarray(row[:_ACTION_DIM], dtype=np.float64)}
