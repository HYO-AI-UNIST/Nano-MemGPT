from __future__ import annotations

import json
from pathlib import Path

import jsonlines
import typer

from nanomemgpt.training.formatting import iter_sft_records
from nanomemgpt.trajectory.schema import TrajectoryStep


app = typer.Typer()


@app.command()
def main(
    trajectories: Path = Path("data/trajectories/gpt4_memgpt_train.jsonl"),
    output: Path = Path("data/processed/sft/function_calls.jsonl"),
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    steps: list[TrajectoryStep] = []
    with jsonlines.open(trajectories) as reader:
        for row in reader:
            steps.append(TrajectoryStep.model_validate(row))
    with jsonlines.open(output, mode="w") as writer:
        for record in iter_sft_records(steps):
            writer.write(record)
    print(json.dumps({"input": str(trajectories), "output": str(output), "records": len(steps)}))


if __name__ == "__main__":
    app()
