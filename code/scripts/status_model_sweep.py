"""Print compact progress and failures for the five-model GPU comparison."""

import json
from pathlib import Path

RUN = Path(__file__).resolve().parents[1] / "generated/five_model_sweep"


def main():
    for model in json.loads(
        (RUN.parents[1] / "results/five_model_sweep/models.json").read_text()
    ):
        name = model["name"]
        counts = {}
        for protocol in ("main", "shared_split"):
            results = [
                json.loads(p.read_text())
                for p in (RUN / "results" / name / protocol).glob("*.json")
            ]
            counts[protocol] = {
                "done": len(results),
                "scored": sum("faithfulness" in r for r in results),
            }
        print(name, counts)
        for log in sorted(RUN.glob(f"{name}_*.log")):
            lines = log.read_text().replace("\r", "\n").splitlines()
            useful = [
                line
                for line in lines
                if line.startswith(
                    ("LOAD", "VERIFIED", "START", "LAYER", "DONE", "COMPLETE")
                )
            ]
            errors = [line for line in lines if "Error:" in line or "Traceback" in line]
            print(
                " ",
                log.stem,
                (errors[-1] if errors else useful[-1] if useful else "starting")[:220],
            )


if __name__ == "__main__":
    main()
