#!/usr/bin/env python3
"""Generate and, only after explicit identity confirmation, submit three QZ jobs."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = "/inspire/hdd2/project/liu-ming-huan/public/ziyang/yam_fintune_data"
REMOTE_WORK = f"{ROOT}/operate/manimux-training-clean"
PROJECT_ID = "project-f34ef3ad-b8b5-4c42-bd6e-47b6ed5e6020"
PROJECT_NAME = "intern-ziyang"
USER_ID = "user-3ea382df-a19b-413a-931e-e8b31312ac0e"
WORKSPACE_ID = "ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6"
LOGIC_COMPUTE_GROUP_ID = "lcg-79b2ad0e-a375-43f3-a0b1-b4ce79710fd7"
H100_4GPU_SPEC_ID = "8a53ac21-299a-4dee-85e9-9c04a544cf8d"
IMAGE = "docker.sii.shaipower.online/inspire-studio/pytorch:25.06-py3"
QZ = Path.home() / ".local/bin/qz"
LOCAL_REPO = Path(__file__).resolve().parents[1]
RUN = "assemble-screwdriver-v1-s0-4xh100-3k-20260825"
LEROBOT_REPO = "yam_assemble_screwdriver_20260825_v1"

LEROBOT_INFO_SHA256 = "8ac40d3ec158342d6c0529bf391c872a43e062977751a0d35422ef6e0105c609"
LEROBOT_TREE_SHA256 = "b37eb892db035feb5707637c9efa36c3d5bc7f4c6e8ac54c1cc3cafc5fba1b14"
XR1_MANIFEST_SHA256 = "067c9b6c7468c058fe3cd6ce3da59abed2f41c8ae6c9516e154bca1c418a9a6b"
XR1_STATS_SHA256 = "c4ea4a57e308da268ca21e690acd76cf7d82a251553b6acd2642f4e185e1f967"
PI05_MANIFEST_SHA256 = "78b6898ea1f8897a0225022b0ab799455cff0e15158a9ca349d8efbbe044aa55"
LINGBOT_INDEX_SHA256 = "5a753ec331c51925d064e1e76e921a7eb3dca9770d2a6a91dac5e0e4162d676a"
LINGBOT_STATS_SHA256 = "e6652ad42b23c8ea155fa4c3317a0389c4882a8c59e0688d41c4d328175eeb31"
XR1_BASE_SHA256 = "94d55a79122050a654b379664b644e874ff90d64ccd30a6a633f816555bcecf7"


def git_sha(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def common_preamble(parent_sha: str, xpolicy_sha: str) -> str:
    return f"""set -euo pipefail
ROOT={ROOT}
WORK={REMOTE_WORK}
test "$(git -C "$WORK" rev-parse HEAD)" = {parent_sha}
test "$(git -C "$WORK/XPolicyLab" rev-parse HEAD)" = {xpolicy_sha}
test "$(nvidia-smi -L | wc -l)" -eq 4
unset WANDB_API_KEY WANDB_BASE_URL WANDB_ENTITY WANDB_PROJECT WANDB_MODE
python3 - "$ROOT" <<'PY'
from pathlib import Path
import hashlib, json, sys
root = Path(sys.argv[1])
lerobot = root / "datasets/lerobot/{LEROBOT_REPO}"
xr1 = root / "datasets/xr1/RoboDojo_real-assemble_the_screwdriver-yam_dual-ee"
info = json.loads((lerobot / "meta/info.json").read_text())
assert info["total_episodes"] == 19 and info["total_frames"] == 17789
assert info["features"]["observation.state"]["shape"] == [14]
assert info["features"]["action"]["shape"] == [14]
assert hashlib.sha256((lerobot / "meta/info.json").read_bytes()).hexdigest() == "{LEROBOT_INFO_SHA256}"
h = hashlib.sha256()
for path in sorted(item for item in lerobot.rglob("*") if item.is_file()):
    h.update(str(path.relative_to(lerobot)).encode() + b"\\0")
    item_hash = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            item_hash.update(chunk)
    h.update(item_hash.digest())
assert h.hexdigest() == "{LEROBOT_TREE_SHA256}"
manifest = json.loads((xr1 / "manifest.json").read_text())
assert manifest["schema"] == "manimux.xr1_yam_dataset.v1"
assert manifest["episodes"] == 19 and manifest["frames"] == 17789
assert manifest["instruction"] == "Assemble the screwdriver."
assert hashlib.sha256((xr1 / "manifest.json").read_bytes()).hexdigest() == "{XR1_MANIFEST_SHA256}"
assert hashlib.sha256((xr1 / "norm_stats.json").read_bytes()).hexdigest() == "{XR1_STATS_SHA256}"
print("data contract verified: 19 episodes / 17789 frames / 14D joint")
PY"""


def checked_hash(path: str, expected: str, *, extra: str = "") -> str:
    return f"""test "$(sha256sum "{path}" | awk '{{print $1}}')" = {expected}
{extra}"""


def payload(name: str, description: str, command: str) -> dict[str, Any]:
    return {
        "logic_compute_group_id": LOGIC_COMPUTE_GROUP_ID,
        "project_id": PROJECT_ID,
        "workspace_id": WORKSPACE_ID,
        "framework": "pytorch",
        "auto_fault_tolerance": False,
        "enable_notification": False,
        "enable_troubleshoot": False,
        "envs": [],
        "dataset_info": [],
        "exclude_nodes": [],
        "specified_nodes": [],
        "pre_check_items": [],
        "description": description,
        "is_publicpath_readonly": False,
        "task_priority": 10,
        "reserve_on_fail_ms": "600000",
        "max_running_time_ms": "86400000",
        "name": name,
        "command": command,
        "framework_config": [
            {
                "image": IMAGE,
                "image_type": "SOURCE_OFFICIAL",
                "instance_count": 1,
                "shm_gi": 200,
                "spec_id": H100_4GPU_SPEC_ID,
            }
        ],
    }


def jobs(parent_sha: str, xpolicy_sha: str) -> dict[str, dict[str, Any]]:
    preamble = common_preamble(parent_sha, xpolicy_sha)
    pi_base = f"{ROOT}/weights/base/pi05_base/params/manifest.ocdbt"
    ling_index = f"{ROOT}/weights/base/lingbot-vla-v2-6b/model.safetensors.index.json"
    ling_stats = f"{ROOT}/cache/lingbot-vla2/{LEROBOT_REPO}/norm_stats.json"
    xr1_base = f"{ROOT}/weights/base/xiaomi/model_states.pt"
    pi_command = preamble + "\n" + checked_hash(pi_base, PI05_MANIFEST_SHA256) + f"""
YAM_TRAIN_ROOT="$ROOT" OPENPI_GPU_IDS=0,1,2,3 OPENPI_FSDP_DEVICES=4 \
OPENPI_BATCH_SIZE=384 OPENPI_NUM_WORKERS=0 \
bash "$WORK/scripts/train_pi05_yam_cluster.sh" gate-train {RUN}"""
    ling_command = (
        preamble
        + "\n"
        + checked_hash(ling_index, LINGBOT_INDEX_SHA256)
        + "\n"
        + checked_hash(ling_stats, LINGBOT_STATS_SHA256)
        + f"""
YAM_TRAIN_ROOT="$ROOT" LINGBOT_VLA2_GPU_IDS=0,1,2,3 \
LINGBOT_VLA2_MICRO_BATCH_SIZE=1 LINGBOT_VLA2_GRAD_ACCUM_STEPS=8 \
LINGBOT_VLA2_TRAIN_WORKERS=8 \
bash "$WORK/scripts/train_lingbot_vla2_yam_cluster.sh" gate-train {RUN}"""
    )
    xr1_command = preamble + "\n" + checked_hash(
        xr1_base,
        XR1_BASE_SHA256,
        extra=f'test "$(stat -c %s "{xr1_base}")" -eq 10226684862',
    ) + f"""
YAM_TRAIN_ROOT="$ROOT" XR1_GPU_IDS=0,1,2,3 XR1_LOGGER=tensorboard \
XR1_MICRO_BATCH_SIZE=1 XR1_GRAD_ACCUM_STEPS=8 \
bash "$WORK/scripts/train_xr1_yam_cluster.sh" gate-train {RUN}"""
    suffix = "assemble-screwdriver-4xh100-3k-20260825"
    return {
        "pi05": payload(
            f"pi05-{suffix}",
            "Pi05 YAM assembly: same-allocation smoke then 3000 steps, save 500, local logs only",
            pi_command,
        ),
        "lingbot-vla2": payload(
            f"lingbot-vla2-{suffix}",
            "LingBot-VLA2 YAM assembly: same-allocation smoke then 3000 steps, save 500, local TensorBoard only",
            ling_command,
        ),
        "xr1": payload(
            f"xr1-{suffix}",
            "XR-1 recorded-EE YAM assembly: same-allocation smoke then 3000 steps, save 500, local TensorBoard only",
            xr1_command,
        ),
    }


def qz_json(*args: str) -> Any:
    result = subprocess.run(
        [str(QZ), *args], check=True, text=True, capture_output=True
    )
    return json.loads(result.stdout)


def dictionaries(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from dictionaries(child)
    elif isinstance(value, list):
        for child in value:
            yield from dictionaries(child)


def verify_qz_identity() -> tuple[dict[str, Any], dict[str, Any]]:
    user_response = qz_json("user", "GetUserDetail")
    user = next((item for item in dictionaries(user_response) if item.get("id") == USER_ID), None)
    if user is None:
        raise RuntimeError(f"QZ user is not Ziyang ({USER_ID})")
    project_response = qz_json(
        "project",
        "GetProjectForPage",
        "--data",
        json.dumps({"page": 1, "page_size": 100, "filter": {"name": PROJECT_NAME}}),
    )
    project = next(
        (item for item in dictionaries(project_response) if item.get("id") == PROJECT_ID),
        None,
    )
    if project is None or project.get("name") != PROJECT_NAME:
        raise RuntimeError(f"QZ project is not {PROJECT_NAME} ({PROJECT_ID})")
    status = str(project.get("status", ""))
    if status not in {"FINISHED", "APPROVE_RESOURCE", "PASS_MODIFY_RESOURCE"}:
        raise RuntimeError(f"QZ project is not submit-ready: status={status}")

    schedule_response = qz_json(
        "train", "GetTrainScheduleConfig", "--set", f"workspace_id={WORKSPACE_ID}"
    )
    schedule = next(
        (
            item
            for item in dictionaries(schedule_response)
            if item.get("workspace_id") == WORKSPACE_ID and "predef_train_spec" in item
        ),
        None,
    )
    if schedule is None:
        raise RuntimeError(f"No train schedule config for workspace {WORKSPACE_ID}")
    specs = json.loads(schedule["predef_train_spec"])
    spec = next((item for item in specs if item.get("id") == H100_4GPU_SPEC_ID), None)
    if spec is None or spec.get("gpu_count") != 4:
        raise RuntimeError(f"4-GPU spec is unavailable: {H100_4GPU_SPEC_ID}")
    if LOGIC_COMPUTE_GROUP_ID not in spec.get("logic_compute_group_ids", []):
        raise RuntimeError("Selected 4-GPU spec is not valid in the H100 compute group")

    group_response = qz_json(
        "workspace",
        "GetLogicComputeGroupById",
        "--set",
        f"LogicComputeGroupId={LOGIC_COMPUTE_GROUP_ID}",
    )
    group = next(
        (
            item
            for item in dictionaries(group_response)
            if item.get("logic_compute_group_id") == LOGIC_COMPUTE_GROUP_ID
        ),
        None,
    )
    if (
        group is None
        or group.get("workspace_id") != WORKSPACE_ID
        or "H100" not in str(group.get("name", ""))
        or "distributed_training" not in str(group.get("support_job_type_list", ""))
    ):
        raise RuntimeError("Selected logic compute group is not the expected H100 training pool")
    return user, project


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=("pi05", "lingbot-vla2", "xr1"), action="append")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--confirm-user-id")
    parser.add_argument("--confirm-project-id")
    parser.add_argument("--confirm-parent-sha")
    args = parser.parse_args()

    parent_sha = git_sha(LOCAL_REPO)
    xpolicy_sha = git_sha(LOCAL_REPO / "XPolicyLab")
    selected = jobs(parent_sha, xpolicy_sha)
    if args.only:
        selected = {name: selected[name] for name in args.only}

    output_dir = LOCAL_REPO / "qz/generated"
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, item in selected.items():
        output = output_dir / f"{name}_assemble_screwdriver_3k.json"
        output.write_text(json.dumps(item, indent=2) + "\n")
        print(f"generated {name}: {output}")

    if not args.submit:
        print(f"dry generation only: parent={parent_sha} XPolicyLab={xpolicy_sha}")
        return
    if args.confirm_user_id != USER_ID:
        parser.error(f"--submit requires --confirm-user-id {USER_ID}")
    if args.confirm_project_id != PROJECT_ID:
        parser.error(f"--submit requires --confirm-project-id {PROJECT_ID}")
    if args.confirm_parent_sha != parent_sha:
        parser.error(f"--submit requires --confirm-parent-sha {parent_sha}")
    user, project = verify_qz_identity()
    print(json.dumps({"confirmed_user": user, "confirmed_project": project}, ensure_ascii=False))
    for name, item in selected.items():
        response = qz_json("train", "CreateJob", "--data", json.dumps(item))
        error = next(
            (entry.get("Error") for entry in dictionaries(response) if entry.get("Error")),
            None,
        )
        if error:
            raise RuntimeError(f"QZ rejected {name}: {error}")
        print(f"submitted {name}: {json.dumps(response, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
