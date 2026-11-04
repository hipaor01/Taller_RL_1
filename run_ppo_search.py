"""Búsqueda de hiperparámetros PPO en validación (protocolo común) + test final.
Uso: python3 run_ppo_search.py <mercado> [state]   →  guarda artifacts/ppo_<state>/<mercado>.pt
"""
import sys, os, json, time, pickle
from dataclasses import replace
from itertools import product
from pathlib import Path
import numpy as np, torch
import ppo_core as C

market = sys.argv[1]
state = sys.argv[2] if len(sys.argv) > 2 else "original"
only_cfg = sys.argv[3] if len(sys.argv) > 3 else None   # "best_of:<ruta.pt>" → reutiliza la config elegida en otro run
OUT = Path("artifacts") / f"ppo_{state}"; OUT.mkdir(parents=True, exist_ok=True)
VAL = C.generate_evaluation_datasets(C.VALIDATION_PROTOCOL)[market]
TEST = C.generate_evaluation_datasets(C.TEST_PROTOCOL)[market]
FP_OK = C.dataset_fingerprint(TEST) == C.EXPECTED_TEST_FINGERPRINTS[market]
print(f"[{market}] huella del test coincide con Higinio/Javi: {FP_OK} (si no, difiere solo en redondeo de plataforma ~1e-15)", flush=True)

GRID = {"learning_rate": (3e-4, 1e-3), "gamma": (0.95, 0.99),
        "clip_ratio": (0.1, 0.2), "entropy_coefficient": (0.01, 0.05)}
if only_cfg:
    base = torch.load(only_cfg, weights_only=False)["selected"]["config"]
    configs = [C.PPOConfig(**{**base, "state": state})]
else:
    configs = [C.PPOConfig(learning_rate=lr, gamma=g, clip_ratio=c, entropy_coefficient=e, state=state)
               for lr, g, c, e in product(*GRID.values())]

ckpt = OUT / f"{market}_search.pkl"
done = pickle.load(open(ckpt, "rb")) if ckpt.exists() else {}
t0 = time.time()
for ci, cfg in enumerate(configs):
    key = json.dumps(C.asdict(cfg), sort_keys=True)
    if key in done:
        continue
    runs = [C.train_ppo(market, s, cfg) for s in C.TRAINING_SEEDS]
    evals = [C.evaluate_ppo(r, market, VAL) for r in runs]
    summ = C.summarize_evaluations(evals)
    done[key] = {"config": C.asdict(cfg), "runs": runs, "validation": evals, "summary": summ}
    pickle.dump(done, open(ckpt, "wb"))
    print(f"[{market}/{state}] {ci+1}/{len(configs)} lr={cfg.learning_rate} g={cfg.gamma} clip={cfg.clip_ratio} ent={cfg.entropy_coefficient} "
          f"val={summ['mean_return']:.2%} P05={summ['return_percentile_05']:.2%} sd={summ['std_between_seeds']:.2%} "
          f"train={np.mean([r['training_seconds'] for r in runs]):.0f}s  [{(time.time()-t0)/60:.0f} min]", flush=True)

candidates = list(done.values())
selected = max(candidates, key=lambda c: C.candidate_rank(c["summary"]))
# ---- test final: una única evaluación, cinco modelos, 200 series comunes
test_evals = [C.evaluate_ppo(r, market, TEST) for r in selected["runs"]]
test_summary = C.summarize_evaluations(test_evals)
baselines = C.evaluate_baselines(market, TEST, C.TEST_PROTOCOL)
torch.save({"market": market, "state": state, "candidates": candidates, "selected": selected,
            "test": test_evals, "test_summary": test_summary, "baselines": baselines,
            "test_fingerprint": C.dataset_fingerprint(TEST), "fingerprint_matches": FP_OK, "obs_scales": C.OBS_SCALES[market]}, OUT / f"{market}.pt")
print(f"[{market}/{state}] SELECCIONADA {selected['config']} | TEST: ret={test_summary['mean_return']:.2%} "
      f"P05={test_summary['return_percentile_05']:.2%} sd={test_summary['std_between_seeds']:.2%} ops={test_summary['mean_trade_count']:.2f} "
      f"| B&H={baselines['buy_and_hold']['mean_return']:.2%} rnd={baselines['random']['mean_return']:.2%}", flush=True)
