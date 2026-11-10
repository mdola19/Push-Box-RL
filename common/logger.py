import json
from pathlib import Path
from typing import Any, Dict, List

class JsonlLogger:
    def __init__(self, path: str, config: Dict[str, Any]):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        with (self.path / "config.json").open("w", encoding="utf-8") as file:
            json.dump(config, file, indent=2, sort_keys=True)
            file.write("\n")
        self._file = (self.path / "metrics.jsonl").open("w", encoding="utf-8")

    def log(self, **metrics) -> None:
        self._file.write(json.dumps(metrics) + "\n")
        self._file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()


def load_runs(folder: str) -> List[Dict[str, Any]]:
    runs = []
    for seed_path in sorted(Path(folder).glob("seed_*")):
        if not seed_path.is_dir():
            continue
        config_path = seed_path / "config.json"
        metrics_path = seed_path / "metrics.jsonl"
        with config_path.open(encoding="utf-8") as file:
            config = json.load(file)
        metrics = []
        with metrics_path.open(encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    metrics.append(json.loads(line))
        runs.append({"config": config, "metrics": metrics})
    return runs
