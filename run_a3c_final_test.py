"""Ejecuta una vez la evaluación final A3C sobre el banco común de test."""

import json
import os
from pathlib import Path
import sys

import run_a3c_search as search_runner


NOTEBOOK_PATH = Path(__file__).with_name("taller_trader2_higinio.ipynb")
OUTPUT_DIR = Path(__file__).with_name("artifacts") / "a3c_final_test"


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
    (OUTPUT_DIR / "final_test.pid").write_text(str(os.getpid()))
    log_path = OUTPUT_DIR / "final_test.log"

    with log_path.open("a", buffering=1) as log_file:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = search_runner.Tee(original_stdout, log_file)
        sys.stderr = search_runner.Tee(original_stderr, log_file)
        try:
            print("Ejecutando o cargando la evaluación final sobre test")
            runtime = search_runner.load_notebook_runtime()
            execute_notebook_cell(
                runtime, "def estimate_a3c_convergence_episode"
            )
            execute_notebook_cell(runtime, "class A3CFinalTestEvaluator")
            results = runtime["A3CFinalTestEvaluator"]().run(verbose=True)
            rows = runtime["build_a3c_final_test_rows"](results)
            runtime["print_a3c_final_test_rows"](rows)
            print("Evaluación final completada y guardada")
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    main()
