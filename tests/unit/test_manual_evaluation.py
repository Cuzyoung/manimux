from __future__ import annotations

import json

import pytest

from manimux.evaluation import write_manual_evaluation


def test_manual_evaluation_is_written_as_an_episode_sidecar(tmp_path) -> None:
    episode = tmp_path / "episode-example"
    episode.mkdir()
    (episode / "meta.json").write_text("{}\n", encoding="utf-8")
    (episode / "result.json").write_text("{}\n", encoding="utf-8")

    target = write_manual_evaluation(
        episode,
        task_result="failure",
        smoothness_score=2,
        failure_tags=["hold_stall", "replay_backtrack", "hold_stall"],
        operator_note=" visible pause ",
        reviewer_id=" Cuzyoung ",
    )

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert target == episode / "evaluation" / "manual-v1.json"
    assert payload["task_result"] == "failure"
    assert payload["smoothness_score"] == 2
    assert payload["failure_tags"] == ["hold_stall", "replay_backtrack"]
    assert payload["operator_note"] == "visible pause"
    assert payload["reviewer_id"] == "Cuzyoung"
    assert payload["review_mode"] == "live"


def test_manual_evaluation_rejects_incomplete_or_invalid_episodes(tmp_path) -> None:
    partial = tmp_path / "episode-example.partial"
    partial.mkdir()
    with pytest.raises(ValueError, match="incomplete"):
        write_manual_evaluation(
            partial,
            task_result="success",
            smoothness_score=5,
            failure_tags=[],
            operator_note="",
            reviewer_id="operator",
        )

    episode = tmp_path / "episode-example"
    episode.mkdir()
    with pytest.raises(ValueError, match="meta.json"):
        write_manual_evaluation(
            episode,
            task_result="success",
            smoothness_score=5,
            failure_tags=[],
            operator_note="",
            reviewer_id="operator",
        )
