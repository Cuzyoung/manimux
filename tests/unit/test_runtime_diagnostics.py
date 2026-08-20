from __future__ import annotations

import numpy as np

from manimux.runtime.diagnostics import build_plan_boundary_payload
from manimux.types import ActionChunk, ActionHorizon


def test_plan_boundary_payload_records_trimmed_raw_and_committed_first_steps() -> None:
    chunk = ActionChunk(
        plan_id="plan-2",
        request_seq=2,
        observation_time_ns=100,
        created_time_ns=110,
        action_space="joint_position",
        dt_ns=50,
        groups={"arm": np.asarray([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])},
    )
    committed = ActionHorizon(
        start_time_ns=200,
        dt_ns=50,
        plan_id="plan-2",
        groups={"arm": np.asarray([[1.5, 2.5], [4.0, 5.0]])},
    )

    payload = build_plan_boundary_payload(
        step=12,
        monotonic_ns=200,
        blend_anchor_source="measured_state",
        blend_steps=2,
        trimmed_steps=1,
        previous_reference={"arm": np.asarray([1.0, 2.0])},
        previous_command={"arm": np.asarray([0.8, 1.8])},
        last_command={"arm": np.asarray([0.9, 1.9])},
        measured={"arm": np.asarray([0.7, 1.7])},
        chunk=chunk,
        committed=committed,
    )

    assert payload["raw_first"] == {"arm": [2.0, 3.0]}
    assert payload["committed_first"] == {"arm": [1.5, 2.5]}
    assert payload["last_command"] == {"arm": [0.9, 1.9]}
    assert payload["measured"] == {"arm": [0.7, 1.7]}
