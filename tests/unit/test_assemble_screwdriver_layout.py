import ast
from pathlib import Path
import shlex


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = REPO_ROOT / "experiments" / "assemble_screwdriver"
REMOTE_CODE_ROOT = "/inspire/hdd2/project/liu-ming-huan/public/ziyang/manimux/"


def _payload(path: Path) -> dict:
    module = ast.parse(path.read_text())
    assignment = next(
        node
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "payload" for target in node.targets)
    )
    return ast.literal_eval(assignment.value)


def test_current_model_layout_and_submit_targets() -> None:
    expected = {"pi05", "lingbot-vla2", "xiaomi-xr1", "gr00t-n17"}
    model_root = EXPERIMENT_ROOT / "models"
    assert {path.name for path in model_root.iterdir() if path.is_dir()} == expected

    job_names = set()
    for model_name in expected:
        model_dir = model_root / model_name
        for filename in ("train.sh", "smoke.sh", "submit.py", "submit_smoke.py"):
            assert (model_dir / filename).is_file()
        for filename in ("submit.py", "submit_smoke.py"):
            payload = _payload(model_dir / filename)
            assert payload["name"] not in job_names
            job_names.add(payload["name"])
            script = next(token for token in shlex.split(payload["command"]) if token.endswith(".sh"))
            assert script.startswith(REMOTE_CODE_ROOT)
            local_script = REPO_ROOT / script.removeprefix(REMOTE_CODE_ROOT)
            assert local_script.is_file()
            assert model_dir in local_script.parents


def test_formal_settings_and_directory_contract() -> None:
    pi05 = (EXPERIMENT_ROOT / "models" / "pi05" / "train.sh").read_text()
    lingbot = (EXPERIMENT_ROOT / "models" / "lingbot-vla2" / "train.sh").read_text()
    xr1 = (EXPERIMENT_ROOT / "models" / "xiaomi-xr1" / "train.sh").read_text()

    assert "OPENPI_BATCH_SIZE=64" in pi05
    assert "OPENPI_NUM_TRAIN_STEPS=15000" in pi05
    assert "OPENPI_SAVE_INTERVAL=1000" in pi05

    assert "--train.micro_batch_size 1" in lingbot
    assert "--train.gradient_accumulation_steps 8" in lingbot
    assert "--train.max_steps 15000" in lingbot
    assert "--train.save_steps 1000" in lingbot

    assert "--nproc_per_node=8" in xr1
    assert "trainer.accumulate_grad_batches=8" in xr1
    assert "trainer.max_steps=15000" in xr1
    assert "trainer.save_interval=1000" in xr1

    for script in (pi05, lingbot, xr1):
        assert "yam_fintune_data/operate" not in script
        assert "/home/ubuntu/manimux" not in script


def test_active_pi05_chain_has_no_custom_preflight_or_finished_guard() -> None:
    generic = (REPO_ROOT / "scripts" / "train_pi05_yam_cluster.sh").read_text()
    for marker in ("require_file", "preflight", "check_environment", "gate-train", "Refusing to overwrite"):
        assert marker not in generic
