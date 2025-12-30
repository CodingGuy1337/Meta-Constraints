#!/usr/bin/env python3
"""
latent_state_meta_constraints.py
================================

Meta-Constraint: **Latent State Continuity** (Representation constraint)

This experiment isolates one structural claim from the essay:

    Maintaining a continuous latent state z_t (a sufficient statistic of history)
    stabilizes sequential learning dynamics relative to memoryless, instantaneous
    likelihood matching.

Formalization
-------------
Observation embedding:
    z_obs,t  ≈  s_t + noise, then normalized to ||z_obs,t|| = 1

Latent state continuity:
    z_{t+1} = Normalize((1-α) z_t + α z_obs,t)

Action selection (both agents):
    a_t = argmax_a ⟨representation_t, u_a⟩

where representation_t is either z_obs,t (memoryless) or z_t (continuous).

Coherence metrics:
- switches / policy churn (oscillation proxy)
- dissonance = 1 - ⟨z_t, z_obs,t⟩
- drift      = 1 - ⟨z_t, z_{t+1}⟩

Backwards-compatible aliases are kept:
InstantObsAgent, ZStateAgent.
"""



from __future__ import annotations
import argparse
import json
import os
import time
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# -----------------------------
# Utility: logging + checkpoints
# -----------------------------

def now_utc_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

class TeeLogger:
    def __init__(self, log_path: str):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"\n=== LOG START {now_utc_str()} ===\n")

    def log(self, msg: str):
        line = f"[{now_utc_str()}] {msg}"
        print(line, flush=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

def atomic_write(path: str, data: str):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(data)
    os.replace(tmp, path)

def save_json(path: str, obj):
    atomic_write(path, json.dumps(obj, indent=2, sort_keys=True))

def append_jsonl(path: str, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")

def load_done_set(jsonl_path: str) -> set:
    done = set()
    if not os.path.exists(jsonl_path):
        return done
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                done.add((obj["agent"], obj["seed"], obj["episode"]))
            except Exception:
                continue
    return done

# -----------------------------
# Math helpers (cosine geometry)
# -----------------------------

def norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v) + 1e-12
    return (v / n).astype(np.float32)

def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

# -----------------------------
# Environment: drifting latent topic on a sphere
# -----------------------------

@dataclass
class StepResult:
    obs: np.ndarray        # z_obs (normalized)
    reward: float          # reward after costs/penalties
    info: Dict

class DriftingSphereBandit:
    def __init__(
        self,
        d: int = 16,
        k_actions: int = 12,
        horizon: int = 250,
        drift_sigma: float = 0.03,
        obs_noise: float = 0.25,
        reward_noise: float = 0.05,
        switch_cost: float = 0.02,
        churn_window: int = 30,
        churn_thresh: int = 6,
        delay: int = 8,
        delayed_penalty: float = 0.25,
        shock_prob: float = 0.02,
        shock_scale: float = 0.35,
        seed: int = 0,
    ):
        self.d = d
        self.k = k_actions
        self.horizon = horizon
        self.drift_sigma = drift_sigma
        self.obs_noise = obs_noise
        self.reward_noise = reward_noise
        self.switch_cost = switch_cost
        self.churn_window = churn_window
        self.churn_thresh = churn_thresh
        self.delay = delay
        self.delayed_penalty = delayed_penalty
        self.shock_prob = shock_prob
        self.shock_scale = shock_scale
        self.rng = np.random.default_rng(seed)

        # fixed action anchors u_a on the sphere
        U = self.rng.normal(0, 1.0, size=(self.k, self.d)).astype(np.float32)
        self.U = (U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-12)).astype(np.float32)

        self.reset()

    def reset(self):
        self.t = 0
        self.s = norm(self.rng.normal(0, 1.0, size=(self.d,)).astype(np.float32))
        self.prev_action = int(self.rng.integers(0, self.k))
        self.switch_times: List[int] = []
        self.scheduled_penalties: List[Tuple[int, float]] = []
        return self._observe()

    def _observe(self) -> np.ndarray:
        z = self.s + self.rng.normal(0, self.obs_noise, size=(self.d,)).astype(np.float32)
        return norm(z)

    def step(self, action: int) -> StepResult:
        assert 0 <= action < self.k

        # Drift (smooth)
        self.s = norm(self.s + self.rng.normal(0, self.drift_sigma, size=(self.d,)).astype(np.float32))

        # Occasional shock (regime jump)
        if self.rng.random() < self.shock_prob:
            self.s = norm(self.s + self.rng.normal(0, self.shock_scale, size=(self.d,)).astype(np.float32))

        # Base reward is alignment between hidden state and chosen anchor
        base = float(np.dot(self.s, self.U[action])) + float(self.rng.normal(0, self.reward_noise))

        # Switching cost
        sw = 1 if action != self.prev_action else 0
        switch_pen = self.switch_cost if sw else 0.0
        if sw:
            self.switch_times.append(self.t)

        # windowed churn
        while self.switch_times and self.switch_times[0] < self.t - self.churn_window:
            self.switch_times.pop(0)
        churn = len(self.switch_times)

        # schedule delayed penalty if churn is too high
        if sw and churn >= self.churn_thresh:
            self.scheduled_penalties.append((self.t + self.delay, self.delayed_penalty))

        # apply due penalties
        due_pen = 0.0
        if self.scheduled_penalties:
            still = []
            for tt, pen in self.scheduled_penalties:
                if tt == self.t:
                    due_pen += pen
                else:
                    still.append((tt, pen))
            self.scheduled_penalties = still

        reward = base - switch_pen - due_pen
        obs = self._observe()

        info = {
            "t": self.t,
            "base": base,
            "switch": sw,
            "switch_pen": switch_pen,
            "churn": churn,
            "due_pen": due_pen,
            "done": (self.t + 1 >= self.horizon),
            # evaluation-only (not given to agent)
            "s_dot_best": float(np.max(self.U @ self.s)),
            "s_dot_chosen": float(np.dot(self.s, self.U[action])),
        }

        self.prev_action = action
        self.t += 1
        return StepResult(obs=obs, reward=reward, info=info)

# -----------------------------
# Agents: only differ by z usage
# -----------------------------

class BaseAgent:
    def __init__(self, U: np.ndarray, seed: int):
        self.U = U
        self.k, self.d = U.shape
        self.rng = np.random.default_rng(seed)

    def reset(self):
        pass

    def act(self, z_obs: np.ndarray) -> int:
        raise NotImplementedError

    def update(self, z_obs: np.ndarray, reward: float):
        pass

class MemorylessObservationAgent(BaseAgent):
    """
    Acts directly on instantaneous observation embedding z_obs:
      a_t = argmax_a <z_obs, u_a>
    """
    def reset(self):
        pass

    def act(self, z_obs: np.ndarray) -> int:
        scores = (self.U @ z_obs).astype(np.float64)
        return int(np.argmax(scores))

class LatentStateContinuityAgent(BaseAgent):
    """
    Maintains streaming latent state z_t on unit sphere:
      z_{t+1} = normalize((1-α) z_t + α z_obs)
    Acts on z_t:
      a_t = argmax_a <z_t, u_a>

    Also logs:
      dissonance_t = 1 - <z_t, z_obs>
      drift_t      = 1 - <z_t, z_{t+1}>
    """
    def __init__(self, U: np.ndarray, seed: int, alpha: float = 0.08):
        super().__init__(U, seed)
        self.alpha = alpha

    def reset(self):
        self.z = norm(self.rng.normal(0, 1.0, size=(self.d,)).astype(np.float32))
        self.last_dissonance = 0.0
        self.last_drift = 0.0

    def act(self, z_obs: np.ndarray) -> int:
        scores = (self.U @ self.z).astype(np.float64)
        return int(np.argmax(scores))

    def update(self, z_obs: np.ndarray, reward: float):
        # dissonance between state and current observation
        self.last_dissonance = float(1.0 - np.dot(self.z, z_obs))
        z_next = norm((1 - self.alpha) * self.z + self.alpha * z_obs)
        self.last_drift = float(1.0 - np.dot(self.z, z_next))
        self.z = z_next


# Backwards-compatible aliases (old names)
InstantObsAgent = MemorylessObservationAgent
ZStateAgent = LatentStateContinuityAgent

# -----------------------------
# Metrics
# -----------------------------

def episode_metrics(actions: List[int], rewards: List[float], dis: List[float], dr: List[float]) -> Dict[str, float]:
    a = np.asarray(actions, dtype=np.int64)
    r = np.asarray(rewards, dtype=np.float64)
    switches = int(np.sum(a[1:] != a[:-1])) if len(a) > 1 else 0
    churn_rate = switches / max(1, len(a)-1)
    return {
        "total_reward": float(np.sum(r)),
        "avg_reward": float(np.mean(r)) if len(r) else 0.0,
        "switches": switches,
        "churn_rate": float(churn_rate),
        "mean_dissonance": float(np.mean(dis)) if len(dis) else 0.0,
        "mean_drift": float(np.mean(dr)) if len(dr) else 0.0,
    }

def summarize(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby(["agent"]).agg(
        episodes=("episode", "count"),
        total_reward_mean=("total_reward", "mean"),
        total_reward_std=("total_reward", "std"),
        churn_mean=("churn_rate", "mean"),
        churn_std=("churn_rate", "std"),
        switches_mean=("switches", "mean"),
        dissonance_mean=("mean_dissonance", "mean"),
        drift_mean=("mean_drift", "mean"),
    ).reset_index()
    return agg.sort_values("total_reward_mean", ascending=False)

# -----------------------------
# Runner
# -----------------------------

def run_episode(env: DriftingSphereBandit, agent_name: str, agent: BaseAgent, seed: int, episode: int) -> Dict:
    z = env.reset()
    agent.reset()

    actions, rewards = [], []
    dissonances, drifts = [], []

    for _ in range(env.horizon):
        a = agent.act(z)
        sr = env.step(a)
        # update agent after seeing obs + reward
        if isinstance(agent, LatentStateContinuityAgent):
            agent.update(sr.obs, sr.reward)
            dissonances.append(agent.last_dissonance)
            drifts.append(agent.last_drift)
        else:
            # keep vectors same length for metrics
            dissonances.append(0.0)
            drifts.append(0.0)

        actions.append(a)
        rewards.append(sr.reward)
        z = sr.obs
        if sr.info.get("done"):
            break

    m = episode_metrics(actions, rewards, dissonances, drifts)
    return {"agent": agent_name, "seed": seed, "episode": episode, **m}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="runs/z_state_geometry")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--horizon", type=int, default=250)
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--k_actions", type=int, default=12)

    # Environment knobs
    ap.add_argument("--drift_sigma", type=float, default=0.03)
    ap.add_argument("--shock_prob", type=float, default=0.02)
    ap.add_argument("--shock_scale", type=float, default=0.35)
    ap.add_argument("--obs_noise", type=float, default=0.25)
    ap.add_argument("--reward_noise", type=float, default=0.05)
    ap.add_argument("--switch_cost", type=float, default=0.02)
    ap.add_argument("--churn_window", type=int, default=30)
    ap.add_argument("--churn_thresh", type=int, default=6)
    ap.add_argument("--delay", type=int, default=8)
    ap.add_argument("--delayed_penalty", type=float, default=0.25)

    # Z-state knob
    ap.add_argument("--alpha", type=float, default=0.08)

    # Ops
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--log_every", type=int, default=50)

    args = ap.parse_args()

    out = args.out
    os.makedirs(out, exist_ok=True)
    logger = TeeLogger(os.path.join(out, "run.log"))

    jsonl_path = os.path.join(out, "results.jsonl")
    csv_path = os.path.join(out, "results.csv")
    summary_path = os.path.join(out, "summary.csv")
    meta_path = os.path.join(out, "meta.json")

    meta = {
        "started_utc": now_utc_str(),
        "episodes": args.episodes,
        "seeds": args.seeds,
        "horizon": args.horizon,
        "d": args.d,
        "k_actions": args.k_actions,
        "env": {
            "drift_sigma": args.drift_sigma,
            "shock_prob": args.shock_prob,
            "shock_scale": args.shock_scale,
            "obs_noise": args.obs_noise,
            "reward_noise": args.reward_noise,
            "switch_cost": args.switch_cost,
            "churn_window": args.churn_window,
            "churn_thresh": args.churn_thresh,
            "delay": args.delay,
            "delayed_penalty": args.delayed_penalty,
        },
        "z_state": {"alpha": args.alpha},
        "agents": ["InstantObs", "ZState(alpha)"],
    }
    save_json(meta_path, meta)

    done = load_done_set(jsonl_path) if args.resume else set()
    if args.resume:
        logger.log(f"Resume enabled. Loaded {len(done)} completed trials from {jsonl_path}")

    all_rows: List[Dict] = []
    if args.resume and os.path.exists(csv_path):
        try:
            all_rows = pd.read_csv(csv_path).to_dict("records")
            logger.log(f"Loaded existing CSV rows: {len(all_rows)}")
        except Exception:
            logger.log("Could not load existing CSV. Will continue from JSONL only.")

    t0 = time.time()
    logger.log("Starting z-state geometry experiment (ONLY z-state vs instantaneous obs).")
    logger.log(f"Out: {out}")
    logger.log(f"Episodes={args.episodes} Seeds={args.seeds} Horizon={args.horizon} d={args.d} K={args.k_actions}")
    logger.log(f"Env: drift_sigma={args.drift_sigma} shock_prob={args.shock_prob} obs_noise={args.obs_noise} switch_cost={args.switch_cost}")
    logger.log(f"Delayed penalty: window={args.churn_window} thresh={args.churn_thresh} delay={args.delay} λ={args.delayed_penalty}")
    logger.log(f"Z-state alpha={args.alpha}")
    logger.log("Checkpoints: results.jsonl (append), results.csv + summary.csv (rolling)")

    trial = 0
    for seed in range(args.seeds):
        # Make a fresh env per seed; anchors are fixed inside env, so ensure comparability by using same env seed for both agents
        env_seed = 1337 + seed * 1000

        env = DriftingSphereBandit(
            d=args.d,
            k_actions=args.k_actions,
            horizon=args.horizon,
            drift_sigma=args.drift_sigma,
            obs_noise=args.obs_noise,
            reward_noise=args.reward_noise,
            switch_cost=args.switch_cost,
            churn_window=args.churn_window,
            churn_thresh=args.churn_thresh,
            delay=args.delay,
            delayed_penalty=args.delayed_penalty,
            shock_prob=args.shock_prob,
            shock_scale=args.shock_scale,
            seed=env_seed,
        )

        # Build agents using the SAME anchor matrix U from the environment
        agents = {
            "InstantObs": InstantObsAgent(env.U, seed=env_seed + 10),
            f"ZState(alpha={args.alpha})": ZStateAgent(env.U, seed=env_seed + 20, alpha=args.alpha),
        }

        for agent_name, agent in agents.items():
            # To be fair: reset the env RNG stream identically per agent by reconstructing env with same seed
            # This makes the two agents see the same underlying s_t / obs / noise sequence per episode index.
            for ep in range(args.episodes):
                key = (agent_name, seed, ep)
                if key in done:
                    continue

                # Re-create env each episode with deterministic seed tied to (seed, ep)
                # so different agents see same episode realizations.
                ep_seed = env_seed + 100000 + ep * 17
                env_ep = DriftingSphereBandit(
                    d=args.d,
                    k_actions=args.k_actions,
                    horizon=args.horizon,
                    drift_sigma=args.drift_sigma,
                    obs_noise=args.obs_noise,
                    reward_noise=args.reward_noise,
                    switch_cost=args.switch_cost,
                    churn_window=args.churn_window,
                    churn_thresh=args.churn_thresh,
                    delay=args.delay,
                    delayed_penalty=args.delayed_penalty,
                    shock_prob=args.shock_prob,
                    shock_scale=args.shock_scale,
                    seed=ep_seed,
                )
                # Use same anchors across agents by copying U from the original env
                env_ep.U = env.U.copy()

                row = run_episode(env_ep, agent_name, agent, seed, ep)
                append_jsonl(jsonl_path, row)
                all_rows.append(row)
                done.add(key)

                trial += 1
                if trial % args.log_every == 0:
                    elapsed = time.time() - t0
                    rps = trial / max(1e-9, elapsed)
                    logger.log(f"Progress: {trial} trials | {rps:.2f} trials/s | last={agent_name} seed{seed} ep{ep}")
                    df = pd.DataFrame(all_rows)
                    df.to_csv(csv_path, index=False)
                    summarize(df).to_csv(summary_path, index=False)
                    logger.log(f"Wrote checkpoints: {csv_path} and {summary_path}")

    df = pd.DataFrame(all_rows)
    df.to_csv(csv_path, index=False)
    summ = summarize(df)
    summ.to_csv(summary_path, index=False)

    elapsed = time.time() - t0
    logger.log(f"DONE. Rows={len(df)} Elapsed={elapsed:.1f}s")
    logger.log("Top-line summary:")
    for _, r in summ.iterrows():
        std = r["total_reward_std"]
        std = 0.0 if (std is None or (isinstance(std, float) and math.isnan(std))) else float(std)
        logger.log(
            f"  {r['agent']:<18} | R={r['total_reward_mean']:.2f}±{std:.2f} | churn={r['churn_mean']:.3f} "
            f"| switches={r['switches_mean']:.1f} | drift={r['drift_mean']:.4f} | disson={r['dissonance_mean']:.4f}"
        )

if __name__ == "__main__":
    main()
