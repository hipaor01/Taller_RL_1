"""Ejecuta la búsqueda A3C definida en taller_trader2_higinio.ipynb."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import hashlib
from itertools import product
import json
import os
from pathlib import Path
import sys
from threading import Lock
from time import perf_counter

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


NOTEBOOK_PATH = Path(__file__).with_name("taller_trader2_higinio.ipynb")
OUTPUT_DIR = Path(__file__).with_name("artifacts") / "a3c_search"
CELL_MARKERS = (
    "class SimpleTradingEnv",
    "def _resolver_rng",
    "def generar_precios_browniano",
    "def generar_precios_reversion_media",
    "def generar_precios_regimenes",
    "class EvaluationProtocol",
    "class ActorCriticNetwork",
    "def compute_n_step_returns",
    "class A3CWorker",
    "class A3CTrainingConfig",
    "def evaluate_a3c_deterministically",
    "A3C_HYPERPARAMETER_GRID",
    "def aggregate_a3c_validation_runs",
    "def _serialize_a3c_search_results",
)


class Tee:
    """Replica la salida en terminal y en un archivo de progreso."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def load_notebook_runtime():
    """Ejecuta solo las celdas reutilizables, omitiendo demos y gráficas."""
    notebook = json.loads(NOTEBOOK_PATH.read_text())
    runtime = {
        "asdict": asdict,
        "Categorical": Categorical,
        "dataclass": dataclass,
        "deepcopy": deepcopy,
        "gym": gym,
        "hashlib": hashlib,
        "Lock": Lock,
        "nn": nn,
        "np": np,
        "Path": Path,
        "perf_counter": perf_counter,
        "product": product,
        "replace": replace,
        "spaces": spaces,
        "ThreadPoolExecutor": ThreadPoolExecutor,
        "torch": torch,
    }

    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if cell.get("cell_type") == "code" and any(
            marker in source for marker in CELL_MARKERS
        ):
            exec(compile(source, NOTEBOOK_PATH.name, "exec"), runtime)

    return runtime


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "search.pid").write_text(str(os.getpid()))
    log_path = OUTPUT_DIR / "search.log"
    with log_path.open("a", buffering=1) as log_file:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = Tee(original_stdout, log_file)
        sys.stderr = Tee(original_stderr, log_file)
        try:
            print("Iniciando o reanudando la búsqueda A3C para cuatro generadores")
            runtime = load_notebook_runtime()
            all_results = runtime["run_all_a3c_searches"](
                output_dir=OUTPUT_DIR,
                verbose=True,
            )
            print("Búsqueda completada")
            for generator_name, results in all_results.items():
                best = results[0]
                print(
                    f"{generator_name}: configuración "
                    f"{best['configuration_index']}, "
                    f"retorno={best['summary']['mean_return']:.4%}"
                )
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    main()
