#!/usr/bin/env python3
import json
import subprocess
from pathlib import Path

payload = {
    "logic_compute_group_id": "lcg-79b2ad0e-a375-43f3-a0b1-b4ce79710fd7",
    "project_id": "project-f34ef3ad-b8b5-4c42-bd6e-47b6ed5e6020",
    "workspace_id": "ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6",
    "framework": "pytorch",
    "auto_fault_tolerance": False,
    "enable_notification": False,
    "enable_troubleshoot": False,
    "envs": [],
    "dataset_info": [],
    "exclude_nodes": [],
    "specified_nodes": [],
    "pre_check_items": [],
    "description": "LingBot-VLA2 screwdriver full-model training: 4 H100, AdamW FSDP2, 3000 steps, save every 500, W&B disabled",
    "is_publicpath_readonly": False,
    "task_priority": 10,
    "reserve_on_fail_ms": "600000",
    "max_running_time_ms": "86400000",
    "name": "lingbot-vla2-screwdriver-4xh100-3k-20260829-v1",
    "command": "bash /inspire/hdd2/project/liu-ming-huan/public/ziyang/manimux/experiments/assemble_screwdriver/train/lingbot_4gpu_train.sh",
    "framework_config": [
        {
            "image": "docker.sii.shaipower.online/inspire-studio/pytorch:25.06-py3",
            "image_type": "SOURCE_OFFICIAL",
            "instance_count": 1,
            "shm_gi": 200,
            "spec_id": "8a53ac21-299a-4dee-85e9-9c04a544cf8d",
        }
    ],
}

result = subprocess.run(
    [str(Path.home() / ".local/bin/qz"), "train", "CreateJob", "--data", json.dumps(payload)],
    check=True,
    text=True,
    capture_output=True,
)
print(result.stdout)
