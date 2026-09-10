"""Explicit manual download of official task-specific TabFM weights."""
import argparse
from workflow_paths import CONFIG, configured_path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("classification", "regression", "both"), default="both")
    parser.add_argument("--revision", default=CONFIG["checkpoint_revision"], help="Prefer a fixed Hub commit for a reproducible release")
    args = parser.parse_args()
    from huggingface_hub import HfApi, snapshot_download
    from direct_shap import atomic_json, digest_file
    repo = CONFIG["checkpoint_repository"]
    revision = HfApi().model_info(repo_id=repo, revision=args.revision).sha
    root = configured_path("checkpoint_root")
    tasks = ("classification", "regression") if args.task == "both" else (args.task,)
    for task in tasks:
        print(f"Downloading {repo} at {revision}, {task}. Several GB per task.", flush=True)
        snapshot_download(repo_id=repo, revision=revision, local_dir=root,
                          allow_patterns=[f"{task}/config.json", f"{task}/model.safetensors", "LICENSE", "README.md"])
        atomic_json(root / f"{task}_download.json", dict(repository=repo, resolved_revision=revision,
            files={name: digest_file(root / task / name) for name in ("config.json", "model.safetensors")}))

if __name__ == "__main__":
    main()
