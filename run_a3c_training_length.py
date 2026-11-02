"""Ejecuta la comparación A3C de 1.000 frente a 2.000 episodios."""

import json
import os
from pathlib import Path
import sys

import run_a3c_search as search_runner


NOTEBOOK_PATH = Path(__file__).with_name("taller_trader2_higinio.ipynb")
OUTPUT_DIR = Path(__file__).with_name("artifacts") / "a3c_training_length"


def execute_notebook_cell(runtime, marker):
    """Ejecuta la primera celda de código que contiene el marcador."""
    notebook = json.loads(NOTEBOOK_PATH.read_text())
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if cell.get("cell_type") == "code" and marker in source:
            exec(compile(source, NOTEBOOK_PATH.name, "exec"), runtime)
            return
    raise RuntimeError(f"No se encontró la celda: {marker}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "training_length.pid").write_text(str(os.getpid()))
    log_path = OUTPUT_DIR / "training_length.log"

    with log_path.open("a", buffering=1) as log_file:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = search_runner.Tee(original_stdout, log_file)
        sys.stderr = search_runner.Tee(original_stderr, log_file)
        try:
            print("Iniciando o reanudando la comparación 1000/2000 episodios")
            runtime = search_runner.load_notebook_runtime()
            execute_notebook_cell(
                runtime, "def estimate_a3c_convergence_episode"
            )
            execute_notebook_cell(
                runtime, "class A3CTrainingLengthExperiment"
            )
            comparisons = runtime["run_a3c_training_length_experiments"](
                verbose=True
            )
            print("Comparación completada")
            for generator_name, results_by_duration in comparisons.items():
                for n_episodes, result in sorted(results_by_duration.items()):
                    summary = result["summary"]
                    convergence = runtime[
                        "summarize_a3c_runs_convergence"
                    ](result["runs"], n_episodes=n_episodes)
                    mean_episode = convergence["mean_convergence_episode"]
                    convergence_text = (
                        "no detectada"
                        if mean_episode is None
                        else f"episodio {mean_episode:.0f}"
                    )
                    print(
                        f"{generator_name} · {n_episodes}: "
                        f"retorno={summary['mean_return']:.4%}, "
                        f"p05={summary['return_percentile_05']:.4%}, "
                        f"convergencia={convergence_text}, "
                        f"tasa={convergence['convergence_rate']:.0%}"
                    )
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    main()
