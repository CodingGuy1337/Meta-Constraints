"""
full_test_suite_checkpointed_meta_constraints.py
===============================================

Checkpointed test harness for oscillation / churn failure modes.

This suite focuses on **Commitment** as a meta-constraint (switch hysteresis /
minimum dwell) in two toy environments with delayed switching penalties.

It logs coherence metrics (reward, churn, oscillations) and writes incremental
checkpoints so long runs survive interruptions.
"""

from __future__ import annotations

import csv
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

# ============================================================
# FULL TEST SUITE (checkpoint logging per experiment/block)
# ============================================================
# Writes per-agent results immediately (append) + periodic checkpoints:
#   - results_suite_rows.csv     (append-only, one row per agent per block)
#   - results_suite_partial.csv  (full checkpoint table, rewritten often)
#   - results_suite.csv          (final)
#   - run_log.txt                (timestamped progress log)
#
# Ctrl+C safe: everything finished so far is on disk.
# ============================================================

# ---- logging knobs ----
VERBOSE = True
PROGRESS_EVERY = 25
PRINT_BLOCK_START = True

# ---- run size knobs ----
EPISODES_PER_SEED = 200
SEEDS = [1, 3, 5, 7, 11]
COMPUTE_SWEEP = [2, 4, 8, 16, 32, 64, 128]

# ---- output paths ----
OUT_DIR = "."  # set to "." or an absolute path on your desktop
ROWS_CSV = os.path.join(OUT_DIR, "results_suite_rows.csv")
PARTIAL_CSV = os.path.join(OUT_DIR, "results_suite_partial.csv")
FINAL_CSV = os.path.join(OUT_DIR, "results_suite.csv")
RUN_LOG = os.path.join(OUT_DIR, "run_log.txt")


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


class ResultLogger:
    """
    Append rows as soon as they are produced, plus periodic full-table checkpoints.
    Handles mixed schemas (EnvA vs EnvB) by maintaining a superset of fieldnames.
    """

    def __init__(self, rows_csv: str, partial_csv: str, run_log: str):
        self.rows_csv = rows_csv
        self.partial_csv = partial_csv
        self.run_log = run_log

        self.fieldnames = []  # superset schema, grows as new keys appear

        # If rows_csv already exists, load its header as initial schema
        if os.path.exists(rows_csv) and os.path.getsize(rows_csv) > 0:
            with open(rows_csv, "r", newline="", encoding="utf-8") as f:
                r = csv.reader(f)
                try:
                    self.fieldnames = next(r)
                except StopIteration:
                    self.fieldnames = []

    def log(self, msg: str) -> None:
        line = f"[{_now()}] {msg}"
        print(line)
        try:
            with open(self.run_log, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass

    def _ensure_schema(self, row: dict) -> bool:
        """Ensure self.fieldnames covers row keys. Returns True if schema expanded."""
        new_keys = [k for k in row.keys() if k not in self.fieldnames]
        if not new_keys:
            return False
        self.fieldnames.extend(new_keys)
        return True

    def _rewrite_rows_csv_with_new_schema(self) -> None:
        """Rewrite append log CSV to include new columns in header."""
        if not (os.path.exists(self.rows_csv) and os.path.getsize(self.rows_csv) > 0):
            return

        tmp_path = self.rows_csv + ".tmp"
        with open(self.rows_csv, "r", newline="", encoding="utf-8") as src, \
             open(tmp_path, "w", newline="", encoding="utf-8") as dst:
            reader = csv.DictReader(src)
            writer = csv.DictWriter(dst, fieldnames=self.fieldnames)
            writer.writeheader()
            for r in reader:
                writer.writerow(r)

        os.replace(tmp_path, self.rows_csv)

    def append_row(self, row: Dict[str, Any]) -> None:
        # Expand schema if needed; if expanded, rewrite existing rows file first
        expanded = self._ensure_schema(row)
        if expanded:
            self.log(f"Schema expanded (+{len([k for k in row.keys() if k in self.fieldnames])} keys). Rewriting rows CSV header.")
            self._rewrite_rows_csv_with_new_schema()

        write_header = not (os.path.exists(self.rows_csv) and os.path.getsize(self.rows_csv) > 0)

        with open(self.rows_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.fieldnames, extrasaction="ignore")
            if write_header:
                w.writeheader()
            w.writerow(row)
            f.flush()
            os.fsync(f.fileno())

    def checkpoint_full(self, all_rows: List[Dict[str, Any]]) -> None:
        if not all_rows:
            return

        # Expand schema to cover everything we've seen so far
        expanded_any = False
        for r in all_rows:
            if self._ensure_schema(r):
                expanded_any = True

        with open(self.partial_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.fieldnames, extrasaction="ignore")
            w.writeheader()
            for r in all_rows:
                w.writerow(r)
            f.flush()
            os.fsync(f.fileno())
@dataclass
class EpisodeMetrics:
    reward: float
    switches: int
    churn: float
    oscillations: int


def summarize(metrics: List[EpisodeMetrics]) -> Dict[str, float]:
    r = np.array([m.reward for m in metrics], dtype=float)
    sw = np.array([m.switches for m in metrics], dtype=float)
    ch = np.array([m.churn for m in metrics], dtype=float)
    os_ = np.array([m.oscillations for m in metrics], dtype=float)
    return {
        "reward_mean": float(r.mean()),
        "reward_std": float(r.std()),
        "switch_mean": float(sw.mean()),
        "switch_std": float(sw.std()),
        "churn_mean": float(ch.mean()),
        "churn_std": float(ch.std()),
        "osc_mean": float(os_.mean()),
        "osc_std": float(os_.std()),
    }


# ============================================================
# ENV A: 2-mode flip world with delayed penalty
# ============================================================
@dataclass
class EnvAConfig:
    T: int = 60
    mode_adv: float = 0.06
    flip_prob: float = 0.03
    noise_std: float = 0.25
    switch_limit: int = 6
    delayed_penalty: float = 25.0
    discount: float = 0.98


StateA = Tuple[int, str, int, str]  # (t, mode, switches, good_mode)
ActionA = int  # 0=stay, 1=switch


class EnvA:
    def __init__(self, cfg: EnvAConfig):
        self.cfg = cfg

    def reset(self, rng: random.Random) -> StateA:
        good_mode = rng.choice(["A", "B"])
        mode = rng.choice(["A", "B"])
        return (0, mode, 0, good_mode)

    def step(self, s: StateA, a: ActionA, rng: random.Random) -> Tuple[StateA, float, bool]:
        t, mode, switches, good_mode = s
        done = (t >= self.cfg.T - 1)

        if rng.random() < self.cfg.flip_prob:
            good_mode = "A" if good_mode == "B" else "B"

        if a == 1:
            mode2 = "A" if mode == "B" else "B"
            switches2 = switches + 1
        else:
            mode2 = mode
            switches2 = switches

        base = 1.0 + (self.cfg.mode_adv if mode2 == good_mode else -self.cfg.mode_adv)
        r = base + rng.gauss(0.0, self.cfg.noise_std)

        if done and switches2 > self.cfg.switch_limit:
            r -= self.cfg.delayed_penalty

        return (t + 1, mode2, switches2, good_mode), r, done


# ============================================================
# ENV B: nonstationary bandit with switching + delayed penalty
# ============================================================
@dataclass
class EnvBConfig:
    K: int = 8
    T: int = 80

    reward_noise_std: float = 0.25
    drift_std: float = 0.02
    shock_prob: float = 0.03
    shock_std: float = 0.25

    switch_cost: float = 0.03
    switch_limit: int = 8
    delayed_penalty: float = 18.0

    discount: float = 0.98


StateB = Tuple[int, int, int, Tuple[float, ...]]  # (t, last_arm, switches, means)
ActionB = int  # arm index


class EnvB:
    def __init__(self, cfg: EnvBConfig):
        self.cfg = cfg

    def reset(self, rng: random.Random) -> StateB:
        means = tuple(rng.gauss(0.0, 0.4) for _ in range(self.cfg.K))
        last_arm = rng.randrange(self.cfg.K)
        return (0, last_arm, 0, means)

    def step(self, s: StateB, a: ActionB, rng: random.Random) -> Tuple[StateB, float, bool]:
        t, last_arm, switches, means_t = s
        done = (t >= self.cfg.T - 1)

        means = np.array(means_t, dtype=float)
        means += np.array([rng.gauss(0.0, self.cfg.drift_std) for _ in range(self.cfg.K)], dtype=float)
        if rng.random() < self.cfg.shock_prob:
            means += np.array([rng.gauss(0.0, self.cfg.shock_std) for _ in range(self.cfg.K)], dtype=float)

        switches2 = switches + (1 if a != last_arm else 0)

        r = means[a] + rng.gauss(0.0, self.cfg.reward_noise_std)
        if a != last_arm:
            r -= self.cfg.switch_cost

        if done and switches2 > self.cfg.switch_limit:
            r -= self.cfg.delayed_penalty

        return (t + 1, a, switches2, tuple(float(x) for x in means)), float(r), done


# ============================================================
# Rollout planning
# ============================================================
def rollout_value_envA(env: EnvA, s: StateA, first_action: ActionA, rng: random.Random,
                       H: int, default_stay_prob: float) -> float:
    total = 0.0
    disc = 1.0
    ss = s
    ss, r, done = env.step(ss, first_action, rng)
    total += disc * r
    disc *= env.cfg.discount
    if done:
        return total
    for _ in range(H - 1):
        a = 0 if rng.random() < default_stay_prob else 1
        ss, r, done = env.step(ss, a, rng)
        total += disc * r
        disc *= env.cfg.discount
        if done:
            break
    return total


def rollout_value_envB(env: EnvB, s: StateB, first_action: ActionB, rng: random.Random,
                       H: int, default_stay_prob: float) -> float:
    total = 0.0
    disc = 1.0
    ss = s
    ss, r, done = env.step(ss, first_action, rng)
    total += disc * r
    disc *= env.cfg.discount
    if done:
        return total
    for _ in range(H - 1):
        _, last_arm, _, _ = ss
        if rng.random() < default_stay_prob:
            a = last_arm
        else:
            a = rng.randrange(env.cfg.K)
        ss, r, done = env.step(ss, a, rng)
        total += disc * r
        disc *= env.cfg.discount
        if done:
            break
    return total


def estimate_action_stats_envA(env: EnvA, s: StateA, rng: random.Random,
                               N: int, H: int, p: float) -> Tuple[float, float, float, float]:
    vals0 = [rollout_value_envA(env, s, 0, rng, H, p) for _ in range(N)]
    vals1 = [rollout_value_envA(env, s, 1, rng, H, p) for _ in range(N)]
    m0, v0 = float(np.mean(vals0)), float(np.var(vals0) + 1e-9)
    m1, v1 = float(np.mean(vals1)), float(np.var(vals1) + 1e-9)
    return m0, v0, m1, v1


def estimate_action_stats_envB(env: EnvB, s: StateB, rng: random.Random,
                               N: int, H: int, p: float) -> Tuple[np.ndarray, np.ndarray]:
    means = []
    vars_ = []
    for arm in range(env.cfg.K):
        vals = [rollout_value_envB(env, s, arm, rng, H, p) for _ in range(N)]
        means.append(float(np.mean(vals)))
        vars_.append(float(np.var(vals) + 1e-9))
    return np.array(means, dtype=float), np.array(vars_, dtype=float)


# ============================================================
# Agents
# ============================================================
class AgentA_Argmax:
    def __init__(self, N: int, H: int, p: float): self.N, self.H, self.p = N, H, p
    def reset(self): pass
    def act(self, env: EnvA, s: StateA, rng: random.Random) -> int:
        m0, _, m1, _ = estimate_action_stats_envA(env, s, rng, self.N, self.H, self.p)
        return 0 if m0 >= m1 else 1


class AgentA_Softmax:
    def __init__(self, N: int, H: int, p: float, tau: float = 0.5): self.N, self.H, self.p, self.tau = N, H, p, tau
    def reset(self): pass
    def act(self, env: EnvA, s: StateA, rng: random.Random) -> int:
        m0, _, m1, _ = estimate_action_stats_envA(env, s, rng, self.N, self.H, self.p)
        x0 = m0 / max(self.tau, 1e-9); x1 = m1 / max(self.tau, 1e-9)
        mx = max(x0, x1)
        p1 = math.exp(x1 - mx) / (math.exp(x0 - mx) + math.exp(x1 - mx))
        return 1 if rng.random() < p1 else 0


class AgentA_LCB:
    def __init__(self, N: int, H: int, p: float, k: float = 1.0): self.N, self.H, self.p, self.k = N, H, p, k
    def reset(self): pass
    def act(self, env: EnvA, s: StateA, rng: random.Random) -> int:
        m0, v0, m1, v1 = estimate_action_stats_envA(env, s, rng, self.N, self.H, self.p)
        l0 = m0 - self.k * math.sqrt(v0); l1 = m1 - self.k * math.sqrt(v1)
        return 0 if l0 >= l1 else 1


class AgentA_EMA_DeltaQ:
    def __init__(self, N: int, H: int, p: float, alpha: float = 0.2):
        self.N, self.H, self.p, self.alpha = N, H, p, alpha
        self.ema = 0.0
    def reset(self): self.ema = 0.0
    def act(self, env: EnvA, s: StateA, rng: random.Random) -> int:
        m0, _, m1, _ = estimate_action_stats_envA(env, s, rng, self.N, self.H, self.p)
        self.ema = (1 - self.alpha) * self.ema + self.alpha * (m1 - m0)
        return 1 if self.ema > 0 else 0


class AgentA_DwellK:
    """Commitment: enforce a minimum dwell time before another switch is allowed."""
    def __init__(self, base, K: int):
        self.base = base; self.K = K; self.left = 0; self.last: Optional[int] = None
    def reset(self): self.base.reset(); self.left = 0; self.last = None
    def act(self, env: EnvA, s: StateA, rng: random.Random) -> int:
        if self.left > 0 and self.last is not None:
            self.left -= 1
            return self.last
        a = self.base.act(env, s, rng)
        self.last = a
        self.left = self.K - 1
        return a


class AgentA_CommitmentHysteresis:
    """Commitment: switch only when integrated evidence exceeds a threshold (hysteresis)."""
    def __init__(self, N: int, H: int, p: float, evidence_decay: float = 0.95, z_thresh: float = 1.5):
        self.N, self.H, self.p = N, H, p
        self.decay = evidence_decay; self.z_thresh = z_thresh
        self.committed_mode: Optional[str] = None
        self.evidence = 0.0
    def reset(self): self.committed_mode = None; self.evidence = 0.0
    def act(self, env: EnvA, s: StateA, rng: random.Random) -> int:
        _, mode, _, _ = s
        m0, v0, m1, v1 = estimate_action_stats_envA(env, s, rng, self.N, self.H, self.p)
        stay_mode = mode
        switch_mode = "A" if mode == "B" else "B"
        if self.committed_mode is None:
            self.committed_mode = stay_mode if m0 >= m1 else switch_mode
            self.evidence = 0.0
        other = "A" if self.committed_mode == "B" else "B"
        if self.committed_mode == stay_mode:
            m_c, v_c = m0, v0; m_o, v_o = m1, v1
        else:
            m_c, v_c = m1, v1; m_o, v_o = m0, v0
        se = math.sqrt(v_c / self.N + v_o / self.N)
        z = (m_o - m_c) / (se + 1e-9)
        self.evidence = self.evidence * self.decay + z
        if self.evidence > self.z_thresh:
            self.committed_mode = other
            self.evidence = 0.0
        return 0 if mode == self.committed_mode else 1


# EnvB agents
class AgentB_Argmax:
    def __init__(self, N: int, H: int, p: float): self.N, self.H, self.p = N, H, p
    def reset(self): pass
    def act(self, env: EnvB, s: StateB, rng: random.Random) -> int:
        m, _ = estimate_action_stats_envB(env, s, rng, self.N, self.H, self.p)
        return int(np.argmax(m))


class AgentB_Softmax:
    def __init__(self, N: int, H: int, p: float, tau: float = 0.6): self.N, self.H, self.p, self.tau = N, H, p, tau
    def reset(self): pass
    def act(self, env: EnvB, s: StateB, rng: random.Random) -> int:
        m, _ = estimate_action_stats_envB(env, s, rng, self.N, self.H, self.p)
        x = m / max(self.tau, 1e-9)
        x = x - np.max(x)
        probs = np.exp(x); probs = probs / np.sum(probs)
        r = rng.random(); c = 0.0
        for i, pi in enumerate(probs):
            c += float(pi)
            if r <= c:
                return i
        return int(np.argmax(m))


class AgentB_LCB:
    def __init__(self, N: int, H: int, p: float, k: float = 1.0): self.N, self.H, self.p, self.k = N, H, p, k
    def reset(self): pass
    def act(self, env: EnvB, s: StateB, rng: random.Random) -> int:
        m, v = estimate_action_stats_envB(env, s, rng, self.N, self.H, self.p)
        return int(np.argmax(m - self.k * np.sqrt(v)))


class AgentB_EMA_Values:
    def __init__(self, N: int, H: int, p: float, alpha: float = 0.2, K: int = 8):
        self.N, self.H, self.p, self.alpha = N, H, p, alpha
        self.ema = np.zeros(K, dtype=float)
    def reset(self): self.ema[:] = 0.0
    def act(self, env: EnvB, s: StateB, rng: random.Random) -> int:
        m, _ = estimate_action_stats_envB(env, s, rng, self.N, self.H, self.p)
        self.ema = (1 - self.alpha) * self.ema + self.alpha * m
        return int(np.argmax(self.ema))


class AgentB_DwellK:
    """Commitment: enforce a minimum dwell time before another switch is allowed."""
    def __init__(self, base, K: int):
        self.base = base; self.K = K; self.left = 0; self.last: Optional[int] = None
    def reset(self): self.base.reset(); self.left = 0; self.last = None
    def act(self, env: EnvB, s: StateB, rng: random.Random) -> int:
        if self.left > 0 and self.last is not None:
            self.left -= 1
            return self.last
        a = self.base.act(env, s, rng)
        self.last = a
        self.left = self.K - 1
        return a


class AgentB_CommitmentHysteresis:
    """Commitment: suppress noise-driven arm switching via evidential hysteresis."""
    def __init__(self, N: int, H: int, p: float, evidence_decay: float = 0.95, z_thresh: float = 1.5, K: int = 8):
        self.N, self.H, self.p = N, H, p
        self.decay = evidence_decay; self.z_thresh = z_thresh
        self.K = K
        self.committed: Optional[int] = None
        self.evidence = 0.0
    def reset(self): self.committed = None; self.evidence = 0.0
    def act(self, env: EnvB, s: StateB, rng: random.Random) -> int:
        m, v = estimate_action_stats_envB(env, s, rng, self.N, self.H, self.p)
        best = int(np.argmax(m))
        if self.committed is None:
            self.committed = best; self.evidence = 0.0
            return best
        c = int(self.committed)
        if best == c:
            self.evidence = self.evidence * self.decay
            return c
        se = math.sqrt(v[c] / self.N + v[best] / self.N)
        z = (m[best] - m[c]) / (se + 1e-9)
        self.evidence = self.evidence * self.decay + float(z)
        if self.evidence > self.z_thresh:
            self.committed = best
            self.evidence = 0.0
        return int(self.committed)


class AgentB_CommitmentThenArgmax:
    def __init__(self, N: int, H: int, p: float, M: int, evidence_decay: float = 0.95, z_thresh: float = 1.5, K: int = 8):
        self.N, self.H, self.p = N, H, p
        self.M = max(2, M)
        self.decay = evidence_decay; self.z_thresh = z_thresh
        self.K = K
        self.committed: Optional[int] = None
        self.evidence = 0.0
    def reset(self): self.committed = None; self.evidence = 0.0
    def act(self, env: EnvB, s: StateB, rng: random.Random) -> int:
        m, v = estimate_action_stats_envB(env, s, rng, self.N, self.H, self.p)
        order = list(np.argsort(-m))
        top = order[: self.M - 1]
        if self.committed is None:
            self.committed = int(order[0]); self.evidence = 0.0
        c = int(self.committed)
        cand = list(dict.fromkeys([c] + [int(x) for x in top]))
        best_cand = max(cand, key=lambda i: m[i])
        if best_cand == c:
            self.evidence = self.evidence * self.decay
            return c
        se = math.sqrt(v[c] / self.N + v[best_cand] / self.N)
        z = (m[best_cand] - m[c]) / (se + 1e-9)
        self.evidence = self.evidence * self.decay + float(z)
        if self.evidence > self.z_thresh:
            self.committed = int(best_cand)
            self.evidence = 0.0
        return int(self.committed)


# ============================================================
# Episode runners
# ============================================================
def run_episode_envA(env: EnvA, agent, seed: int) -> EpisodeMetrics:
    rng = random.Random(seed)
    s = env.reset(rng)
    agent.reset()
    total = 0.0
    switches = 0
    modes = [s[1]]
    for _ in range(env.cfg.T):
        a = agent.act(env, s, rng)
        s2, r, done = env.step(s, a, rng)
        total += r
        if a == 1:
            switches += 1
        modes.append(s2[1])
        s = s2
        if done:
            break
    osc = 0
    for i in range(2, len(modes)):
        if modes[i] == modes[i - 2] and modes[i] != modes[i - 1]:
            osc += 1
    return EpisodeMetrics(reward=total, switches=switches, churn=switches / env.cfg.T, oscillations=osc)


def run_episode_envB(env: EnvB, agent, seed: int) -> EpisodeMetrics:
    rng = random.Random(seed)
    s = env.reset(rng)
    agent.reset()
    total = 0.0
    switches = 0
    actions = [s[1]]
    for _ in range(env.cfg.T):
        a = agent.act(env, s, rng)
        s2, r, done = env.step(s, a, rng)
        total += r
        if a != s[1]:
            switches += 1
        actions.append(a)
        s = s2
        if done:
            break
    osc = 0
    for i in range(2, len(actions)):
        if actions[i] == actions[i - 2] and actions[i] != actions[i - 1]:
            osc += 1
    return EpisodeMetrics(reward=total, switches=switches, churn=switches / env.cfg.T, oscillations=osc)


# ============================================================
# Experiment definitions
# ============================================================
@dataclass
class PlanConfig:
    rollouts_per_action: int
    rollout_horizon: int
    default_stay_prob: float = 0.85


def make_agents_envA(pc: PlanConfig) -> Dict[str, Any]:
    base = AgentA_Argmax(pc.rollouts_per_action, pc.rollout_horizon, pc.default_stay_prob)
    return {
        "Argmax": base,
        "Commitment": AgentA_CommitmentHysteresis(pc.rollouts_per_action, pc.rollout_horizon, pc.default_stay_prob),
        "Softmax(tau=0.5)": AgentA_Softmax(pc.rollouts_per_action, pc.rollout_horizon, pc.default_stay_prob, tau=0.5),
        "LCB(k=1.0)": AgentA_LCB(pc.rollouts_per_action, pc.rollout_horizon, pc.default_stay_prob, k=1.0),
        "EMA(ΔQ,a=0.2)": AgentA_EMA_DeltaQ(pc.rollouts_per_action, pc.rollout_horizon, pc.default_stay_prob, alpha=0.2),
        "Dwell(k=4)": AgentA_DwellK(base, K=4),
    }


def make_agents_envB(pc: PlanConfig, K: int) -> Dict[str, Any]:
    base = AgentB_Argmax(pc.rollouts_per_action, pc.rollout_horizon, pc.default_stay_prob)
    return {
        "Argmax": base,
        "Commitment": AgentB_CommitmentHysteresis(pc.rollouts_per_action, pc.rollout_horizon, pc.default_stay_prob, K=K),
        "Softmax(tau=0.6)": AgentB_Softmax(pc.rollouts_per_action, pc.rollout_horizon, pc.default_stay_prob, tau=0.6),
        "LCB(k=1.0)": AgentB_LCB(pc.rollouts_per_action, pc.rollout_horizon, pc.default_stay_prob, k=1.0),
        "EMA(values,a=0.2)": AgentB_EMA_Values(pc.rollouts_per_action, pc.rollout_horizon, pc.default_stay_prob, alpha=0.2, K=K),
        "Dwell(k=4)": AgentB_DwellK(base, K=4),
        "Commit->Argmax(M=3)": AgentB_CommitmentThenArgmax(pc.rollouts_per_action, pc.rollout_horizon, pc.default_stay_prob, M=3, K=K),
    }


def run_block_envA(logger: ResultLogger, all_rows: List[Dict[str, Any]],
                   block_name: str, env_cfg: EnvAConfig, pc: PlanConfig,
                   episodes: int, seeds: List[int]) -> None:
    env = EnvA(env_cfg)
    agents = make_agents_envA(pc)
    if PRINT_BLOCK_START:
        logger.log(f"=== Block {block_name} (EnvA) | rollouts={pc.rollouts_per_action} horizon={pc.rollout_horizon} ===")
    for agent_name, agent in agents.items():
        ms: List[EpisodeMetrics] = []
        for sd in seeds:
            base_seed = sd * 100_000
            for i in range(episodes):
                if VERBOSE and (i % PROGRESS_EVERY == 0) and sd == seeds[0] and i > 0:
                    logger.log(f"{block_name} | {agent_name} | episode {i}/{episodes}")
                ms.append(run_episode_envA(env, agent, base_seed + i))
        stats = summarize(ms)
        logger.log(f"{block_name} EnvA {agent_name} | R {stats['reward_mean']:.2f}±{stats['reward_std']:.2f} | "
                   f"ch {stats['churn_mean']:.3f} | osc {stats['osc_mean']:.2f}")
        row = {
            "block": block_name, "env": "EnvA", "agent": agent_name,
            "rollouts": pc.rollouts_per_action, "horizon": pc.rollout_horizon,
            "noise_std": env_cfg.noise_std, "flip_prob": env_cfg.flip_prob, "mode_adv": env_cfg.mode_adv,
            "delayed_penalty": env_cfg.delayed_penalty, "switch_limit": env_cfg.switch_limit,
            **stats
        }
        all_rows.append(row)
        logger.append_row(row)
        logger.checkpoint_full(all_rows)


def run_block_envB(logger: ResultLogger, all_rows: List[Dict[str, Any]],
                   block_name: str, env_cfg: EnvBConfig, pc: PlanConfig,
                   episodes: int, seeds: List[int]) -> None:
    env = EnvB(env_cfg)
    agents = make_agents_envB(pc, K=env_cfg.K)
    if PRINT_BLOCK_START:
        logger.log(f"=== Block {block_name} (EnvB) | rollouts={pc.rollouts_per_action} horizon={pc.rollout_horizon} ===")
    for agent_name, agent in agents.items():
        ms: List[EpisodeMetrics] = []
        for sd in seeds:
            base_seed = sd * 100_000
            for i in range(episodes):
                if VERBOSE and (i % PROGRESS_EVERY == 0) and sd == seeds[0] and i > 0:
                    logger.log(f"{block_name} | {agent_name} | episode {i}/{episodes}")
                ms.append(run_episode_envB(env, agent, base_seed + i))
        stats = summarize(ms)
        logger.log(f"{block_name} EnvB {agent_name} | R {stats['reward_mean']:.2f}±{stats['reward_std']:.2f} | "
                   f"ch {stats['churn_mean']:.3f} | osc {stats['osc_mean']:.2f}")
        row = {
            "block": block_name, "env": "EnvB", "agent": agent_name,
            "rollouts": pc.rollouts_per_action, "horizon": pc.rollout_horizon, "K": env_cfg.K,
            "reward_noise_std": env_cfg.reward_noise_std, "drift_std": env_cfg.drift_std, "shock_prob": env_cfg.shock_prob,
            "switch_cost": env_cfg.switch_cost, "delayed_penalty": env_cfg.delayed_penalty, "switch_limit": env_cfg.switch_limit,
            **stats
        }
        all_rows.append(row)
        logger.append_row(row)
        logger.checkpoint_full(all_rows)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    logger = ResultLogger(ROWS_CSV, PARTIAL_CSV, RUN_LOG)
    logger.log("Run started.")
    logger.log(f"Output dir: {os.path.abspath(OUT_DIR)}")

    seeds = SEEDS
    episodes = EPISODES_PER_SEED
    base_pc = PlanConfig(rollouts_per_action=8, rollout_horizon=12, default_stay_prob=0.85)

    all_rows: List[Dict[str, Any]] = []

    # 1) Mechanism isolation (EnvA)
    base_envA = EnvAConfig()
    mech_blocks = [
        ("Mech_Base", base_envA),
        ("Mech_Noise0", EnvAConfig(**{**base_envA.__dict__, "noise_std": 0.0})),
        ("Mech_Penalty0", EnvAConfig(**{**base_envA.__dict__, "delayed_penalty": 0.0})),
        ("Mech_Flip0", EnvAConfig(**{**base_envA.__dict__, "flip_prob": 0.0})),
        ("Mech_MarginBig", EnvAConfig(**{**base_envA.__dict__, "mode_adv": 0.20})),
    ]
    for name, cfg in mech_blocks:
        run_block_envA(logger, all_rows, name, cfg, base_pc, episodes, seeds)

    # 2) Compute scaling (EnvA)
    for N in COMPUTE_SWEEP:
        pc = PlanConfig(rollouts_per_action=N, rollout_horizon=12, default_stay_prob=0.85)
        run_block_envA(logger, all_rows, f"ComputeSweep_N={N}", base_envA, pc, episodes, seeds)

    # 3) Robustness (EnvB)
    base_envB = EnvBConfig()
    run_block_envB(logger, all_rows, "Robust_Base", base_envB, base_pc, episodes, seeds)

    for N in [2, 8, 32]:
        pc = PlanConfig(rollouts_per_action=N, rollout_horizon=12, default_stay_prob=0.85)
        run_block_envB(logger, all_rows, f"Robust_Compute_N={N}", base_envB, pc, episodes, seeds)

    # final rewrite
    with open(FINAL_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    logger.log(f"Run finished. Final CSV written: {FINAL_CSV}")


if __name__ == "__main__":
    main()
