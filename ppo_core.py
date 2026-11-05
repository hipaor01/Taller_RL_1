"""PPO para el taller de trading — núcleo reutilizable.

Comparte con los notebooks de Higinio (A3C) y Javi (DQN):
  * entorno SimpleTradingEnv y los cuatro generadores con RNG explícito,
  * protocolo de semillas (TRAINING_SEEDS, VALIDATION_PROTOCOL, TEST_PROTOCOL),
  * series de entrenamiento SeedSequence([semilla, mercado, episodio, 0]),
  * red 2-32-32 tanh con cabezas actor (3 logits) y crítico V(s), misma inicialización que A3C,
  * evaluación determinista (argmax) con evaluate_rule_based_policy y baselines.
"""
from dataclasses import dataclass, asdict
from time import perf_counter
from copy import deepcopy
import hashlib
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
from torch import nn
from torch.distributions import Categorical

torch.set_num_threads(1)

# ----------------------------------------------------------------------------- entorno (copiado del enunciado)
class SimpleTradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, prices):
        super(SimpleTradingEnv, self).__init__()
        self.prices = prices
        self.max_steps = len(prices) - 1
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=np.array([-1.0, 0], dtype=np.float32),
            high=np.array([1.0, 1], dtype=np.float32), dtype=np.float32)

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 5
        self.position = 0
        self.initial_balance = 1000.0
        self.balance = self.initial_balance
        self.cash = self.initial_balance
        self.shares = 0.0
        return self._get_obs(), {}

    def _get_obs(self):
        window = self.prices[self.current_step - 5: self.current_step + 1]
        trend = (window[-1] - np.mean(window)) / np.mean(window)
        return np.array([np.clip(trend, -1.0, 1.0), self.position], dtype=np.float32)

    def step(self, action):
        prev_price = self.prices[self.current_step]
        self.current_step += 1
        curr_price = self.prices[self.current_step]
        reward = 0.0
        cost_rate = 0.001
        if action == 2 and self.position == 0:
            commission = self.cash * cost_rate
            effective_cash = self.cash - commission
            self.shares = effective_cash / prev_price
            self.cash = 0.0
            self.position = 1
            reward -= cost_rate
        elif action == 0 and self.position == 1:
            sale_value = self.shares * prev_price
            commission = sale_value * cost_rate
            self.cash = sale_value - commission
            self.shares = 0.0
            self.position = 0
            reward -= cost_rate
        if self.position == 1:
            price_change = (curr_price - prev_price) / prev_price
            reward += price_change
        if self.position == 1:
            self.balance = self.cash + (self.shares * curr_price)
        else:
            self.balance = self.cash
        terminated = self.current_step >= self.max_steps
        return self._get_obs(), reward, terminated, False, {"balance": self.balance}


# ----------------------------------------------------------------------------- generadores (versión con rng explícito de Javi/Higinio)
def _resolver_rng(rng):
    if rng is None:
        return np.random.default_rng()
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng debe ser una instancia de np.random.Generator")
    return rng


def generar_precios(n_steps=1000, rng=None):
    rng = _resolver_rng(rng)
    t = np.linspace(0, 50, n_steps)
    drift = 0.0005 * t
    ciclo = np.sin(t) * 0.05
    ruido = rng.normal(0, 0.1, n_steps)
    retornos = drift + ciclo + ruido
    return 100 * np.cumprod(1 + retornos)


def generar_precios_browniano(n_steps=1000, start_price=100.0, mu=0.0001, sigma=0.02, rng=None):
    rng = _resolver_rng(rng)
    dt = 1.0
    retornos = rng.normal((mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), n_steps)
    return start_price * np.cumprod(np.exp(retornos))


def generar_precios_reversion_media(n_steps=1000, start_price=100.0, mean_price=100.0, kappa=0.05, volatility=0.02, rng=None):
    rng = _resolver_rng(rng)
    precios = np.zeros(n_steps)
    precios[0] = start_price
    for t in range(1, n_steps):
        delta = kappa * (mean_price - precios[t-1]) + rng.normal(0, volatility * mean_price)
        precios[t] = precios[t-1] + delta
        precios[t] = max(precios[t], 0.1)
    return precios


def generar_precios_regimenes(n_steps=1000, start_price=100.0, rng=None):
    rng = _resolver_rng(rng)
    precios = np.zeros(n_steps)
    precios[0] = start_price
    estado = 0
    retornos = []
    for _ in range(n_steps):
        if rng.random() < 0.02:
            estado = 1 - estado
        if estado == 0:
            ret = rng.normal(0.001, 0.01)
        else:
            ret = rng.normal(-0.0015, 0.025)
        retornos.append(ret)
    retornos = np.array(retornos)
    precios = start_price * np.cumprod(1 + retornos)
    return np.maximum(precios, 0.1)


# ----------------------------------------------------------------------------- protocolo común (celdas de Higinio, literales)
@dataclass(frozen=True)
class EvaluationProtocol:
    """Define un banco de evaluación reproducible e independiente."""

    name: str
    n_series: int
    n_steps: int
    price_seed: int
    action_seed: int

    def __post_init__(self):
        if not self.name:
            raise ValueError("name no puede estar vacío")
        if self.n_series <= 0:
            raise ValueError("n_series debe ser positivo")
        if self.n_steps < 7:
            raise ValueError("n_steps debe ser al menos 7")
        if self.price_seed < 0 or self.action_seed < 0:
            raise ValueError("las semillas no pueden ser negativas")

    @staticmethod
    def _rng(base_seed, generator_id, series_index):
        if generator_id < 0 or series_index < 0:
            raise ValueError("generator_id y series_index no pueden ser negativos")
        seed_sequence = np.random.SeedSequence([base_seed, generator_id, series_index])
        return np.random.default_rng(seed_sequence)

    def price_rng(self, generator_id, series_index):
        return self._rng(self.price_seed, generator_id, series_index)

    def action_rng(self, generator_id, series_index):
        return self._rng(self.action_seed, generator_id, series_index)

    def generate_dataset(self, price_generator, generator_id):
        series = [np.asarray(price_generator(n_steps=self.n_steps, rng=self.price_rng(generator_id, series_index)),
                             dtype=np.float64) for series_index in range(self.n_series)]
        dataset = np.stack(series)
        if dataset.shape != (self.n_series, self.n_steps):
            raise ValueError("el generador devolvió una forma no válida")
        if not np.all(np.isfinite(dataset)) or np.any(dataset <= 0.0):
            raise ValueError("los precios deben ser finitos y positivos")
        return dataset


TRAINING_SEEDS = (101, 202, 303, 404, 505)
VALIDATION_PROTOCOL = EvaluationProtocol(name="validation", n_series=50, n_steps=500, price_seed=5_000, action_seed=6_000)
TEST_PROTOCOL = EvaluationProtocol(name="test", n_series=200, n_steps=500, price_seed=10_000, action_seed=20_000)

PRICE_GENERATORS = {
    "tendencia_ciclo": (0, generar_precios),
    "browniano": (1, generar_precios_browniano),
    "reversion_media": (2, generar_precios_reversion_media),
    "regimenes": (3, generar_precios_regimenes),
}
MARKET_LABELS = {"tendencia_ciclo": "Tendencia + ciclo", "browniano": "Browniano",
                 "reversion_media": "Reversión a la media", "regimenes": "Cambios de régimen"}


def generate_evaluation_datasets(protocol):
    return {name: protocol.generate_dataset(generator, generator_id)
            for name, (generator_id, generator) in PRICE_GENERATORS.items()}


def dataset_fingerprint(dataset):
    normalized = np.ascontiguousarray(dataset, dtype="<f8")
    return hashlib.sha256(normalized.tobytes()).hexdigest()


EXPECTED_TEST_FINGERPRINTS = {
    'tendencia_ciclo': 'f7ba229a24755ad123d0b01e608ed72e16f347bb16be7c87fd309e1afb38763d',
    'browniano': 'd05b4102b634985e678b6a0511fcd0497d524f3b4b72d396043ebe25e362688d',
    'reversion_media': '16ac74754351e609c738490d97f7c69383a10a31c4920d4b25423df215d86339',
    'regimenes': '061f1f1ef1f21f034eea600b65d5d87702d4430699cf38250fb552fb58b013ce'}


def training_prices(generator_name, training_seed, episode_index, n_steps=500):
    """Misma serie de entrenamiento que usan A3C y DQN para (semilla, mercado, episodio)."""
    generator_id, generator = PRICE_GENERATORS[generator_name]
    rng = np.random.default_rng(np.random.SeedSequence([training_seed, generator_id, episode_index, 0]))
    return np.asarray(generator(n_steps=n_steps, rng=rng), dtype=np.float64)


# ----------------------------------------------------------------------------- estados
def obs_raw(prices, t, position):
    """Observación del enunciado, vectorizada sobre series: (tendencia 6 precios, posición)."""
    window = prices[:, t - 5: t + 1]
    mean = window.mean(axis=1)
    trend = np.clip((window[:, -1] - mean) / mean, -1.0, 1.0)
    return np.stack([trend, position.astype(np.float64)], axis=1).astype(np.float32)


MA_WINDOW = 20


def dist_raw(prices, t):
    """Distancia del precio a su media móvil de 20 pasos (solo precios ya observados)."""
    w = prices[:, max(0, t - MA_WINDOW + 1): t + 1]
    return np.clip(prices[:, t] / w.mean(axis=1) - 1.0, -1.0, 1.0).astype(np.float32)


def observation_scales(generator_name, n_series=10, seed=101):
    """Estandarización de la observación: desviación típica de tendencia y distancia en 10 series de
    ENTRENAMIENTO (semilla 101, episodios 0-9). Nunca usa validación ni test. La posición (0/1) no se escala."""
    P = np.stack([training_prices(generator_name, seed, e) for e in range(n_series)])
    trend = np.concatenate([obs_raw(P, t, np.zeros(n_series))[:, 0] for t in range(5, P.shape[1] - 1)])
    dist = np.concatenate([dist_raw(P, t) for t in range(5, P.shape[1] - 1)])
    return {"trend": float(trend.std()), "dist": float(dist.std())}


OBS_SCALES = {name: observation_scales(name) for name in PRICE_GENERATORS}


def make_obs_fn(state, generator_name):
    """Devuelve la función de observación (vectorizada) del estado indicado, ya estandarizada para el mercado."""
    sc = OBS_SCALES[generator_name]
    if state == "original":
        return 2, lambda prices, t, position: obs_raw(prices, t, position) * np.array([1 / sc["trend"], 1.0], dtype=np.float32)
    if state == "enriquecido":
        def fn(prices, t, position):
            base = obs_raw(prices, t, position)
            return np.stack([base[:, 0] / sc["trend"], dist_raw(prices, t) / sc["dist"], base[:, 1]], axis=1).astype(np.float32)
        return 3, fn
    raise ValueError(state)


STATE_DIMS = {"original": 2, "enriquecido": 3}


class VectorTradingEnv:
    """N copias de SimpleTradingEnv en paralelo (misma dinámica, mismas comisiones, mismo reward)."""

    def __init__(self, prices, obs_fn=obs_raw):
        self.prices = np.asarray(prices, dtype=np.float64)
        self.n, self.T = self.prices.shape
        self.max_steps = self.T - 1
        self.obs_fn = obs_fn

    def reset(self):
        self.t = 5
        self.position = np.zeros(self.n, dtype=np.int64)
        self.cash = np.full(self.n, 1000.0)
        self.shares = np.zeros(self.n)
        self.balance = np.full(self.n, 1000.0)
        return self.obs_fn(self.prices, self.t, self.position)

    def step(self, action):
        prev = self.prices[:, self.t]
        self.t += 1
        curr = self.prices[:, self.t]
        reward = np.zeros(self.n)
        buy = (action == 2) & (self.position == 0)
        sell = (action == 0) & (self.position == 1)
        self.shares[buy] = (self.cash[buy] - self.cash[buy] * 0.001) / prev[buy]
        self.cash[buy] = 0.0
        self.position[buy] = 1
        reward[buy] -= 0.001
        sale = self.shares[sell] * prev[sell]
        self.cash[sell] = sale - sale * 0.001
        self.shares[sell] = 0.0
        self.position[sell] = 0
        reward[sell] -= 0.001
        held = self.position == 1
        reward[held] += (curr[held] - prev[held]) / prev[held]
        self.balance = np.where(held, self.cash + self.shares * curr, self.cash)
        done = self.t >= self.max_steps
        return self.obs_fn(self.prices, self.t, self.position), reward, done


# ----------------------------------------------------------------------------- red (idéntica a ActorCriticNetwork de A3C)
class ActorCriticNetwork(nn.Module):
    def __init__(self, observation_dim, action_dim=3, hidden_dim=32):
        super().__init__()
        self.observation_dim = observation_dim
        self.feature_extractor = nn.Sequential(nn.Linear(observation_dim, hidden_dim), nn.Tanh(),
                                               nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        self.critic_head = nn.Linear(hidden_dim, 1)
        hidden_gain = nn.init.calculate_gain("tanh")
        for layer in self.feature_extractor:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=hidden_gain)
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.zeros_(self.actor_head.bias)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.zeros_(self.critic_head.bias)

    def forward(self, observations):
        features = self.feature_extractor(observations)
        return self.actor_head(features), self.critic_head(features).squeeze(-1)


# ----------------------------------------------------------------------------- PPO
@dataclass(frozen=True)
class PPOConfig:
    n_episodes: int = 1_000          # presupuesto común: 1.000 episodios × 500 precios
    n_steps: int = 500
    n_envs: int = 16                 # actores paralelos síncronos (A3C usa 2 workers asíncronos)
    rollout_steps: int = 32          # pasos por actor y por actualización → lote de 512 transiciones
    n_epochs: int = 4
    minibatch_size: int = 128
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coefficient: float = 0.01   # rejilla: 0,01 / 0,05
    value_loss_coefficient: float = 0.5
    max_gradient_norm: float = 0.5
    hidden_dim: int = 32
    state: str = "original"

    def __post_init__(self):
        if (self.n_envs * self.rollout_steps) % self.minibatch_size:
            raise ValueError("el lote debe ser múltiplo del minibatch")


def train_ppo(generator_name, training_seed, config):
    """Entrena PPO consumiendo exactamente los 1.000 episodios del protocolo (series SeedSequence)."""
    torch.manual_seed(training_seed)
    obs_dim, obs_fn = make_obs_fn(config.state, generator_name)
    model = ActorCriticNetwork(obs_dim, 3, config.hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, eps=1e-5)
    generator_id, _ = PRICE_GENERATORS[generator_name]
    sample_rng = torch.Generator().manual_seed(int(np.random.SeedSequence([training_seed, generator_id, 31]).generate_state(1)[0]))
    shuffle_rng = np.random.default_rng([training_seed, generator_id, 32])
    n_env, T = config.n_envs, config.rollout_steps
    n_rounds = -(-config.n_episodes // n_env)   # cada ronda: n_env episodios en paralelo (la última puede ser parcial)
    history, start, updates = [], perf_counter(), 0
    for round_index in range(n_rounds):
        episodes = list(range(round_index * n_env, min((round_index + 1) * n_env, config.n_episodes)))
        n_env = len(episodes)
        prices = np.stack([training_prices(generator_name, training_seed, e, config.n_steps) for e in episodes])
        env = VectorTradingEnv(prices, obs_fn)
        obs = env.reset()
        reward_sum = np.zeros(n_env)
        done_all = False
        while not done_all:
            # ---- rollout de T pasos (o hasta el final del episodio)
            obs_buf, act_buf, logp_buf, val_buf, rew_buf, done_buf = [], [], [], [], [], []
            for _ in range(T):
                obs_t = torch.from_numpy(obs)
                with torch.no_grad():
                    logits, value = model(obs_t)
                    dist = Categorical(logits=logits)
                    action = torch.multinomial(dist.probs, 1, generator=sample_rng).squeeze(1)
                    logp = dist.log_prob(action)
                next_obs, reward, done = env.step(action.numpy())
                obs_buf.append(obs_t); act_buf.append(action); logp_buf.append(logp); val_buf.append(value)
                rew_buf.append(torch.from_numpy(reward.astype(np.float32))); done_buf.append(done)
                reward_sum += reward
                obs = next_obs
                if done:
                    done_all = True
                    break
            # ---- GAE
            with torch.no_grad():
                last_value = torch.zeros(n_env) if done_all else model(torch.from_numpy(obs))[1]
            L = len(rew_buf)
            adv = torch.zeros(L, n_env)
            gae = torch.zeros(n_env)
            for k in reversed(range(L)):
                nonterminal = 0.0 if (k == L - 1 and done_all) else 1.0
                next_v = last_value if k == L - 1 else val_buf[k + 1]
                delta = rew_buf[k] + config.gamma * next_v * nonterminal - val_buf[k]
                gae = delta + config.gamma * config.gae_lambda * nonterminal * gae
                adv[k] = gae
            returns = adv + torch.stack(val_buf)
            b_obs = torch.cat(obs_buf); b_act = torch.cat(act_buf); b_logp = torch.cat(logp_buf)
            b_adv = adv.reshape(-1); b_ret = returns.reshape(-1)
            b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
            # ---- actualización PPO (varias épocas sobre el mismo lote)
            n = b_obs.shape[0]
            mb = min(config.minibatch_size, n)
            for _ in range(config.n_epochs):
                perm = shuffle_rng.permutation(n)
                for s in range(0, n, mb):
                    idx = torch.from_numpy(perm[s:s + mb])
                    logits, value = model(b_obs[idx])
                    dist = Categorical(logits=logits)
                    ratio = torch.exp(dist.log_prob(b_act[idx]) - b_logp[idx])
                    a = b_adv[idx]
                    policy_loss = -torch.min(ratio * a, torch.clamp(ratio, 1 - config.clip_ratio, 1 + config.clip_ratio) * a).mean()
                    value_loss = nn.functional.mse_loss(value, b_ret[idx])
                    entropy = dist.entropy().mean()
                    loss = policy_loss + config.value_loss_coefficient * value_loss - config.entropy_coefficient * entropy
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), config.max_gradient_norm)
                    optimizer.step()
                    updates += 1
        for k in range(n_env):
            history.append({"episode": episodes[k] + 1, "return": float(env.balance[k] / 1000 - 1), "reward": float(reward_sum[k]),
                            "elapsed_seconds": perf_counter() - start})
    history.sort(key=lambda h: h["episode"])
    return {"training_seed": training_seed, "config": asdict(config), "model_state_dict": deepcopy(model.state_dict()),
            "history": history, "training_seconds": perf_counter() - start, "transitions": config.n_episodes * (config.n_steps - 6),
            "updates": updates}


# ----------------------------------------------------------------------------- evaluación (función literal de Higinio + baselines de Javi)
def evaluate_rule_based_policy(price_dataset, policy_factory):
    dataset = np.asarray(price_dataset, dtype=np.float64)
    episode_rewards, final_balances, episode_returns, action_counts, trade_counts = [], [], [], [], []
    evaluation_start = perf_counter()
    for series_index, prices in enumerate(dataset):
        env = SimpleTradingEnv(prices)
        observation, _ = env.reset()
        select_action = policy_factory(series_index)
        terminated = truncated = False
        total_reward = 0.0
        episode_action_counts = np.zeros(env.action_space.n, dtype=np.int64)
        episode_trade_count = 0
        info = {"balance": env.balance}
        while not (terminated or truncated):
            action = int(select_action(observation, env))
            previous_position = env.position
            episode_action_counts[action] += 1
            observation, reward, terminated, truncated, info = env.step(action)
            episode_trade_count += int(env.position != previous_position)
            total_reward += float(reward)
        final_balance = float(info["balance"])
        episode_rewards.append(total_reward); final_balances.append(final_balance)
        episode_returns.append(final_balance / env.initial_balance - 1.0)
        action_counts.append(episode_action_counts); trade_counts.append(episode_trade_count)
    episode_returns = np.asarray(episode_returns); trade_counts = np.asarray(trade_counts)
    return {"n_series": int(dataset.shape[0]), "episode_rewards": np.asarray(episode_rewards),
            "final_balances": np.asarray(final_balances), "returns": episode_returns,
            "action_counts": np.stack(action_counts), "trade_counts": trade_counts,
            "mean_return": float(episode_returns.mean()), "std_return": float(episode_returns.std()),
            "median_return": float(np.median(episode_returns)),
            "return_percentile_05": float(np.percentile(episode_returns, 5)),
            "return_percentile_95": float(np.percentile(episode_returns, 95)),
            "mean_trade_count": float(trade_counts.mean()), "evaluation_seconds": perf_counter() - evaluation_start}


def ppo_policy_factory(run, generator_name):
    """Política determinista (argmax de los logits), sin muestreo ni aprendizaje."""
    cfg = run["config"]
    obs_dim, obs_fn = make_obs_fn(cfg["state"], generator_name)
    model = ActorCriticNetwork(obs_dim, 3, cfg["hidden_dim"])
    model.load_state_dict(run["model_state_dict"])
    model.eval()

    def factory(series_index):
        def choose(observation, env):
            x = torch.from_numpy(obs_fn(env.prices[None, :], env.current_step, np.array([env.position]))[0])
            with torch.no_grad():
                return int(model(x)[0].argmax())
        return choose
    return factory


def evaluate_ppo(run, generator_name, dataset):
    return evaluate_rule_based_policy(dataset, ppo_policy_factory(run, generator_name))


def evaluate_baselines(generator_name, dataset, protocol):
    generator_id, _ = PRICE_GENERATORS[generator_name]

    def cash_factory(series_index):
        return lambda observation, env: 1

    def buy_and_hold_factory(series_index):
        first = [True]

        def select_action(observation, env):
            if first[0]:
                first[0] = False
                return 2
            return 1
        return select_action

    def random_factory(series_index):
        rng = protocol.action_rng(generator_id, series_index)
        return lambda observation, env: int(rng.integers(env.action_space.n))

    return {"cash": evaluate_rule_based_policy(dataset, cash_factory),
            "buy_and_hold": evaluate_rule_based_policy(dataset, buy_and_hold_factory),
            "random": evaluate_rule_based_policy(dataset, random_factory)}


def summarize_evaluations(evaluations):
    """Métricas acordadas en el grupo: retorno medio, desv. entre semillas, P05, operaciones."""
    returns = np.stack([item["returns"] for item in evaluations])
    seed_means = returns.mean(axis=1)
    return {"mean_return": float(returns.mean()), "std_return": float(returns.std()),
            "std_between_seeds": float(seed_means.std(ddof=1)) if len(seed_means) > 1 else None,
            "return_percentile_05": float(np.percentile(returns, 5)), "median_return": float(np.median(returns)),
            "mean_trade_count": float(np.mean([e["mean_trade_count"] for e in evaluations]))}


def candidate_rank(summary):
    """Mismo criterio de selección que DQN: retorno medio, P05 y menor dispersión entre semillas."""
    return summary["mean_return"], summary["return_percentile_05"], -summary["std_between_seeds"]


def estimate_convergence_episode(episode_returns, block_size=50, change_threshold=0.05, stable_blocks=3, scale_floor=0.01):
    """Criterio heurístico de Higinio/Javi: bloques de 50 episodios, cambios relativos ≤ 5 %."""
    r = np.asarray(episode_returns, dtype=np.float64)
    nb = r.size // block_size
    if nb < stable_blocks + 1:
        return None
    m = r[:nb * block_size].reshape(nb, block_size).mean(axis=1)
    changes = np.abs(np.diff(m)) / np.maximum(np.abs(m[:-1]), scale_floor)
    for i in range(len(changes) - stable_blocks + 1):
        if np.all(changes[i:] <= change_threshold):
            return int((i + 2) * block_size)
    return None
