from __future__ import annotations

import numpy as np

from manimux.runtime.timeline import ActionTimeline
from manimux.types import ActionChunk


def _chunk(request_seq: int, observation_time_ns: int = 0) -> ActionChunk:
    values = np.arange(10, dtype=np.float64).reshape(5, 2)
    return ActionChunk(
        plan_id=f"plan-{request_seq}",
        request_seq=request_seq,
        observation_time_ns=observation_time_ns,
        created_time_ns=observation_time_ns,
        action_space="joint_position",
        dt_ns=10,
        groups={"left_arm": values, "right_arm": -values},
    )


def test_commit_is_atomic_and_rejects_stale_sequence() -> None:
    timeline = ActionTimeline({"left_arm": 2, "right_arm": 2})
    current = {"left_arm": np.zeros(2), "right_arm": np.zeros(2)}
    accepted = timeline.commit(
        _chunk(1),
        now_ns=0,
        commit_lead_ns=0,
        max_plan_age_ns=100,
        current_command=current,
        blend_steps=0,
    )
    stale = timeline.commit(
        _chunk(1),
        now_ns=1,
        commit_lead_ns=0,
        max_plan_age_ns=100,
        current_command=current,
        blend_steps=0,
    )
    assert accepted.accepted
    assert not stale.accepted
    assert stale.reason == "stale_request_seq"
    assert timeline.active_plan_id == "plan-1"


def test_commit_trims_obsolete_prefix_and_interpolates() -> None:
    timeline = ActionTimeline({"left_arm": 2, "right_arm": 2})
    current = {"left_arm": np.zeros(2), "right_arm": np.zeros(2)}
    result = timeline.commit(
        _chunk(1),
        now_ns=20,
        commit_lead_ns=0,
        max_plan_age_ns=100,
        current_command=current,
        blend_steps=0,
    )
    assert result.accepted
    assert result.trimmed_steps == 2
    sample = timeline.sample(25)
    assert sample is not None
    np.testing.assert_allclose(sample["left_arm"], [5.0, 6.0])
    np.testing.assert_allclose(sample["right_arm"], [-5.0, -6.0])

    committed = timeline.active_horizon()
    assert committed is not None
    assert committed.start_time_ns == 20
    assert committed.groups["left_arm"].shape == (3, 2)
    np.testing.assert_array_equal(committed.groups["left_arm"][0], [4.0, 5.0])
    assert timeline.cursor(20) == 0
    assert timeline.cursor(31) == 1
    assert timeline.cursor(100) == 3


def test_dimension_mismatch_rejects_entire_chunk() -> None:
    timeline = ActionTimeline({"left_arm": 2, "right_arm": 2})
    current = {"left_arm": np.zeros(2), "right_arm": np.zeros(2)}
    chunk = _chunk(2)
    chunk.groups["right_arm"] = np.zeros((5, 3))
    result = timeline.commit(
        chunk,
        now_ns=0,
        commit_lead_ns=0,
        max_plan_age_ns=100,
        current_command=current,
        blend_steps=0,
    )
    assert not result.accepted
    assert result.reason == "dimension_mismatch:right_arm"
    assert timeline.active_plan_id is None
