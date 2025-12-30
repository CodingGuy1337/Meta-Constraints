#!/usr/bin/env python3
"""
vigilance_meta_constraints.py
=============================

Meta-Constraint: **Vigilance** (Change-sensitivity constraint)
and its controlled tension with **Commitment** (action-switching constraint).

Dataset-driven nonstationary bandit:
- long stable segments,
- rare regime changes,
- noise bursts that imitate change,
- switching costs + delayed churn penalties.

Event-based metrics:
- detection lag after true change points
- false alarms (switches not near changes)
- post-change regret AUC over a fixed window

Vigilance uses a hazard proxy h_t derived from recent prediction-error statistics
to selectively *increase* adaptation when the world changes, while staying inert
during stable periods.

Backwards-compatible aliases are kept:
CommitmentAgent, VigilantAgent.
"""



from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np


# -----------------------------
# Dataset generator
# -----------------------------

@dataclass
class DatasetConfig:
    T: int = 600
    K: int = 8
    M: int = 12

    # Regime change process (explicit change points)
    stable_mean: int = 120          # average stable segment length (steps)
    min_stable: int = 40            # minimum stable length
    max_stable: int = 220           # cap for stable length
    storm_len: int = 0              # if >0, inject rapid successive regime swaps for this many steps

    # Rewards
    base_mean: float = 0.0
    gap: float = 2.0                # how much better the best arm is than others in each regime
    subgap: float = 0.35            # small structure on non-best arms
    sigma: float = 0.35
    sigma_burst: float = 1.0
    burst_prob: float = 0.012       # chance per step to start a noise burst
    burst_len: int = 10

    # Costs
    switch_cost: float = 0.12
    delay: int = 25
    churn_penalty: float = 0.50

    seed: int = 0


@dataclass
class Dataset:
    regimes: np.ndarray            # (T,) int
    change_points: np.ndarray      # indices where regime changes (inclusive change time)
    means: np.ndarray              # (K,M) float
    sigmas: np.ndarray             # (T,) float


def _clip_int(x: int, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, x)))


def generate_dataset(cfg: DatasetConfig) -> Dataset:
    rng = np.random.default_rng(cfg.seed)

    # Means: each regime has a distinct best arm (cycled), with clear gap.
    means = np.full((cfg.K, cfg.M), cfg.base_mean, dtype=np.float64)

    best_arms = np.arange(cfg.K) % cfg.M
    # ensure best arms are diverse if K>M
    rng.shuffle(best_arms)

    for k in range(cfg.K):
        b = int(best_arms[k])
        means[k, :] += rng.normal(0.0, cfg.subgap, size=cfg.M)
        means[k, b] = cfg.base_mean + cfg.gap + rng.normal(0.0, 0.05)

    # Build regime trajectory with explicit change points (stable segments)
    regimes = np.zeros(cfg.T, dtype=np.int64)
    cps: List[int] = [0]
    cur = int(rng.integers(0, cfg.K))
    t = 0
    while t < cfg.T:
        # sample stable segment length
        # geometric-ish, but clipped
        L = int(rng.geometric(1.0 / max(1, cfg.stable_mean)))
        L = _clip_int(L, cfg.min_stable, cfg.max_stable)
        end = min(cfg.T, t + L)
        regimes[t:end] = cur
        t = end
        if t >= cfg.T:
            break
        # pick a different regime at change
        nxt = int(rng.integers(0, cfg.K - 1))
        if nxt >= cur:
            nxt += 1
        cur = nxt
        cps.append(t)

        # optional "storm": rapid successive changes to punish sluggishness
        if cfg.storm_len > 0:
            storm_end = min(cfg.T, t + cfg.storm_len)
            while t < storm_end:
                # swap every 3-5 steps
                jump = int(rng.integers(3, 6))
                end2 = min(storm_end, t + jump)
                regimes[t:end2] = cur
                t = end2
                if t >= storm_end:
                    break
                nxt = int(rng.integers(0, cfg.K - 1))
                if nxt >= cur:
                    nxt += 1
                cur = nxt
                cps.append(t)

    change_points = np.array(sorted(set(cps[1:])), dtype=np.int64)  # exclude 0

    # Noise schedule with bursts that mimic regime change (false alarms)
    sigmas = np.full(cfg.T, cfg.sigma, dtype=np.float64)
    in_burst = 0
    for i in range(cfg.T):
        if in_burst > 0:
            sigmas[i] = cfg.sigma_burst
            in_burst -= 1
        else:
            if rng.random() < cfg.burst_prob:
                in_burst = cfg.burst_len
                sigmas[i] = cfg.sigma_burst

    return Dataset(regimes=regimes, change_points=change_points, means=means, sigmas=sigmas)


# -----------------------------
# Dataset-driven Environment
# -----------------------------

@dataclass
class EnvConfig:
    switch_cost: float = 0.12
    delay: int = 25
    churn_penalty: float = 0.50


class DatasetBanditEnv:
    """
    Uses a pre-generated dataset (regime[t], sigmas[t]) and regime-specific means.
    Adds immediate switch cost + delayed churn penalty.
    """
    def __init__(self, ds: Dataset, cfg: EnvConfig, seed: int = 0):
        self.ds = ds
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.M = ds.means.shape[1]
        self.reset()

    def reset(self):
        self.t = 0
        self.prev_a: Optional[int] = None
        self.switch_queue = [0] * int(self.cfg.delay)
        return {"t": 0}

    def step(self, a: int) -> Tuple[Dict, float, bool, Dict]:
        t = self.t
        reg = int(self.ds.regimes[t])
        mu = float(self.ds.means[reg, a])
        r = float(mu + self.rng.normal(0.0, float(self.ds.sigmas[t])))

        switched = 0
        if self.prev_a is not None and a != self.prev_a:
            r -= float(self.cfg.switch_cost)
            switched = 1

        past_switch = self.switch_queue.pop(0)
        self.switch_queue.append(switched)
        if past_switch == 1:
            r -= float(self.cfg.churn_penalty)

        self.prev_a = int(a)
        self.t += 1
        done = self.t >= len(self.ds.regimes)

        info = {
            "regime": reg,
            "mu": mu,
            "switched": int(switched),
            "sigma": float(self.ds.sigmas[t]),
            "is_change_point": int(t in set(self.ds.change_points.tolist())),
        }
        return {"t": self.t}, r, done, info



# ---------------------------------------------------------------------
# Terminology note (essay alignment)
# ---------------------------------------------------------------------
# Commitment: constraints on action switching (dwell-time, evidential hysteresis)
# Vigilance : change-sensitive modulation of commitment + exploration via hazard h_t
# ---------------------------------------------------------------------

# -----------------------------
# Agents (same family as before)
# -----------------------------

class Agent:
    name: str = "Agent"
    def reset(self, M: int):
        raise NotImplementedError
    def act(self, t: int) -> int:
        raise NotImplementedError
    def observe(self, t: int, a: int, r: float):
        pass


class GreedyArgmax(Agent):
    name = "GreedyArgmax"
    def __init__(self, eps: float = 0.0, seed: int = 0):
        self.eps = eps
        self.seed = seed
    def reset(self, M: int):
        self.M = M
        self.rng = np.random.default_rng(self.seed)
        self.n = np.zeros(M, dtype=np.int64)
        self.q = np.zeros(M, dtype=np.float64)
    def act(self, t: int) -> int:
        if self.rng.random() < self.eps:
            return int(self.rng.integers(0, self.M))
        return int(np.argmax(self.q))
    def observe(self, t: int, a: int, r: float):
        self.n[a] += 1
        self.q[a] += (r - self.q[a]) / max(1, self.n[a])


class SoftmaxAgent(Agent):
    name = "Softmax"
    def __init__(self, tau: float = 0.6, seed: int = 1):
        self.tau = tau
        self.seed = seed
    def reset(self, M: int):
        self.M = M
        self.rng = np.random.default_rng(self.seed)
        self.n = np.zeros(M, dtype=np.int64)
        self.q = np.zeros(M, dtype=np.float64)
    def act(self, t: int) -> int:
        z = self.q / max(1e-9, self.tau)
        z -= np.max(z)
        p = np.exp(z)
        p /= np.sum(p)
        return int(self.rng.choice(self.M, p=p))
    def observe(self, t: int, a: int, r: float):
        self.n[a] += 1
        self.q[a] += (r - self.q[a]) / max(1, self.n[a])


class EMAValueAgent(Agent):
    name = "EMA"
    def __init__(self, beta: float = 0.06, seed: int = 2):
        self.beta = beta
        self.seed = seed
    def reset(self, M: int):
        self.M = M
        self.rng = np.random.default_rng(self.seed)
        self.q = np.zeros(M, dtype=np.float64)
        self.seen = np.zeros(M, dtype=np.int64)
    def act(self, t: int) -> int:
        return int(np.argmax(self.q))
    def observe(self, t: int, a: int, r: float):
        self.seen[a] += 1
        b = self.beta if self.seen[a] > 1 else 1.0
        self.q[a] = (1 - b) * self.q[a] + b * r


class FixedDwellAgent(Agent):
    name = "FixedDwell"
    def __init__(self, dwell: int = 8, base: Optional[Agent] = None):
        self.dwell = dwell
        self.base = base if base is not None else GreedyArgmax()
    def reset(self, M: int):
        self.M = M
        self.base.reset(M)
        self.a_star = 0
        self.lock = 0
    def act(self, t: int) -> int:
        if self.lock <= 0:
            self.a_star = self.base.act(t)
            self.lock = self.dwell
        self.lock -= 1
        return int(self.a_star)
    def observe(self, t: int, a: int, r: float):
        self.base.observe(t, a, r)


class EvidentialCommitmentAgent(Agent):
    name = "EvidentialCommitment"
    def __init__(self, thr: float = 1.2, seed: int = 3):
        self.thr = thr
        self.seed = seed
    def reset(self, M: int):
        self.M = M
        self.rng = np.random.default_rng(self.seed)
        self.n = np.zeros(M, dtype=np.int64)
        self.q = np.zeros(M, dtype=np.float64)
        self.a_star = int(self.rng.integers(0, M))
        self.acc = 0.0
    def act(self, t: int) -> int:
        challenger = int(np.argmax(self.q))
        if challenger != self.a_star:
            adv = float(self.q[challenger] - self.q[self.a_star])
            self.acc = max(0.0, self.acc + max(0.0, adv))
            if self.acc > self.thr:
                self.a_star = challenger
                self.acc = 0.0
        else:
            self.acc = max(0.0, self.acc - 0.05)
        return int(self.a_star)
    def observe(self, t: int, a: int, r: float):
        self.n[a] += 1
        self.q[a] += (r - self.q[a]) / max(1, self.n[a])


class VigilanceAgent(Agent):
    name = "Vigilance(HazardGate+Precision)"
    def __init__(
        self,
        thr_base: float = 1.0,
        thr_scale: float = 2.0,
        tau_low: float = 0.12,
        tau_high: float = 0.95,
        ema_err: float = 0.10,
        k_slope: float = 10.0,
        k_level: float = 3.0,
        seed: int = 4,
    ):
        self.thr_base = thr_base
        self.thr_scale = thr_scale
        self.tau_low = tau_low
        self.tau_high = tau_high
        self.ema_err = ema_err
        self.k_slope = k_slope
        self.k_level = k_level
        self.seed = seed

    def reset(self, M: int):
        self.M = M
        self.rng = np.random.default_rng(self.seed)
        self.n = np.zeros(M, dtype=np.int64)
        self.q = np.zeros(M, dtype=np.float64)
        self.a_star = int(self.rng.integers(0, M))
        self.acc = 0.0
        self.err_ema = 0.0
        self.err2_ema = 0.0
        self.prev_err_ema = 0.0
        self.tau = self.tau_low

    def _sigmoid(self, x: float) -> float:
        return 1.0 / (1.0 + np.exp(-x))

    def _hazard(self) -> float:
        slope = self.err_ema - self.prev_err_ema
        z = self.k_slope * slope + self.k_level * (self.err_ema - 0.5 * np.sqrt(max(1e-9, self.err2_ema)))
        return float(np.clip(self._sigmoid(z), 0.0, 1.0))

    def act(self, t: int) -> int:
        h = self._hazard()
        self.tau = float(self.tau_low + (self.tau_high - self.tau_low) * h)

        # challenger via softmax to allow quick recovery when hazard is high
        z = self.q / max(1e-9, self.tau)
        z -= np.max(z)
        p = np.exp(z)
        p /= np.sum(p)
        challenger = int(self.rng.choice(self.M, p=p))

        thr_t = float(self.thr_base * (1.0 + self.thr_scale * (1.0 - h)))

        if challenger != self.a_star:
            adv = float(self.q[challenger] - self.q[self.a_star])
            self.acc = max(0.0, self.acc + (1.0 + h) * max(0.0, adv))
            if self.acc > thr_t:
                self.a_star = challenger
                self.acc = 0.0
        else:
            self.acc = max(0.0, self.acc - 0.05)
        return int(self.a_star)

    def observe(self, t: int, a: int, r: float):
        self.n[a] += 1
        self.q[a] += (r - self.q[a]) / max(1, self.n[a])
        pred = float(self.q[a])
        err = abs(r - pred)

        self.prev_err_ema = self.err_ema
        b = self.ema_err
        self.err_ema = (1 - b) * self.err_ema + b * err
        self.err2_ema = (1 - b) * self.err2_ema + b * (err * err)


# -----------------------------
# Evaluation: make vigilance visible
# -----------------------------

@dataclass
class RunStats:
    total_reward: float
    churn: float
    switches: int
    regret_like: float
    detection_lag_mean: float
    false_alarm_rate: float
    post_change_regret_auc: float


def _best_arm(ds: Dataset, t: int) -> int:
    reg = int(ds.regimes[t])
    return int(np.argmax(ds.means[reg]))


def run_episode(env: DatasetBanditEnv, agent: Agent, ds: Dataset, post_window: int = 30, near_cp: int = 6) -> RunStats:
    env.reset()
    agent.reset(env.M)

    T = len(ds.regimes)
    total = 0.0
    switches = 0
    regret_like = 0.0

    actions = np.zeros(T, dtype=np.int64)
    regimes = ds.regimes

    prev_a = None
    for t in range(T):
        a = agent.act(t)
        _, r, done, info = env.step(a)
        agent.observe(t, a, r)

        actions[t] = a
        total += r

        reg = int(info["regime"])
        best_mu = float(np.max(ds.means[reg]))
        regret_like += (best_mu - float(info["mu"]))

        if prev_a is not None and a != prev_a:
            switches += 1
        prev_a = a

        if done:
            break

    churn = switches / max(1, T - 1)

    # Detection lag: for each change point, how long until the agent plays the new best arm
    lags = []
    aucs = []
    cps = ds.change_points
    for cp in cps:
        cp = int(cp)
        if cp >= T:
            continue
        best_after = _best_arm(ds, cp)
        lag = None
        # find first time >= cp where action is best_after
        for t in range(cp, min(T, cp + post_window)):
            if int(actions[t]) == best_after:
                lag = t - cp
                break
        if lag is None:
            lag = post_window  # maxed out
        lags.append(lag)

        # post-change regret AUC over window
        auc = 0.0
        for t in range(cp, min(T, cp + post_window)):
            reg = int(regimes[t])
            mu_best = float(np.max(ds.means[reg]))
            mu_a = float(ds.means[reg, int(actions[t])])
            auc += (mu_best - mu_a)
        aucs.append(auc)

    detection_lag_mean = float(np.mean(lags)) if lags else 0.0
    post_change_regret_auc = float(np.mean(aucs)) if aucs else 0.0

    # False alarms: switches that occur not near a true change point (within near_cp window)
    cp_mask = np.zeros(T, dtype=bool)
    for cp in cps:
        lo = max(0, int(cp) - near_cp)
        hi = min(T, int(cp) + near_cp + 1)
        cp_mask[lo:hi] = True

    switch_events = np.zeros(T, dtype=bool)
    switch_events[1:] = actions[1:] != actions[:-1]
    false_alarms = int(np.sum(switch_events & (~cp_mask)))
    total_switches = int(np.sum(switch_events))
    false_alarm_rate = float(false_alarms / max(1, total_switches))

    return RunStats(
        total_reward=float(total),
        churn=float(churn),
        switches=int(switches),
        regret_like=float(regret_like),
        detection_lag_mean=detection_lag_mean,
        false_alarm_rate=false_alarm_rate,
        post_change_regret_auc=post_change_regret_auc,
    )


def summarize(results: Dict[str, List[RunStats]]) -> str:
    lines = []
    header = (
        f"{'Agent':28s} | {'Reward':>14s} | {'Churn':>10s} | {'Lag':>9s} | "
        f"{'FalseAlm':>9s} | {'PostAUC':>10s}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for name, stats in results.items():
        R = np.array([s.total_reward for s in stats], dtype=float)
        C = np.array([s.churn for s in stats], dtype=float)
        L = np.array([s.detection_lag_mean for s in stats], dtype=float)
        F = np.array([s.false_alarm_rate for s in stats], dtype=float)
        A = np.array([s.post_change_regret_auc for s in stats], dtype=float)

        lines.append(
            f"{name:28s} | {R.mean():8.2f}±{R.std():5.2f} | {C.mean():6.3f}±{C.std():4.3f} | "
            f"{L.mean():6.2f}±{L.std():4.2f} | {F.mean():6.3f}±{F.std():4.3f} | {A.mean():7.2f}±{A.std():5.2f}"
        )
    return "\n".join(lines)


def parse_args():
    p = argparse.ArgumentParser(description="Dataset-driven vigilance experiment with bursty regime shifts + noise bursts.")
    # dataset
    p.add_argument("--seeds", type=int, default=128)
    p.add_argument("--T", type=int, default=600)
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--M", type=int, default=12)
    p.add_argument("--stable_mean", type=int, default=120)
    p.add_argument("--min_stable", type=int, default=40)
    p.add_argument("--max_stable", type=int, default=220)
    p.add_argument("--storm_len", type=int, default=0, help="inject rapid successive changes after each change point (0 disables)")
    p.add_argument("--gap", type=float, default=2.0)
    p.add_argument("--subgap", type=float, default=0.35)
    p.add_argument("--sigma", type=float, default=0.35)
    p.add_argument("--sigma_burst", type=float, default=1.0)
    p.add_argument("--burst_prob", type=float, default=0.012)
    p.add_argument("--burst_len", type=int, default=10)

    # costs
    p.add_argument("--switch_cost", type=float, default=0.12)
    p.add_argument("--delay", type=int, default=25)
    p.add_argument("--churn_penalty", type=float, default=0.50)

    # evaluation
    p.add_argument("--post_window", type=int, default=30, help="window after each change for lag/AUC")
    p.add_argument("--near_cp", type=int, default=6, help="how many steps around change count as 'near' for false alarms")

    # agents
    p.add_argument("--eps", type=float, default=0.0)
    p.add_argument("--tau", type=float, default=0.6)
    p.add_argument("--ema_beta", type=float, default=0.06)
    p.add_argument("--dwell", type=int, default=8)
    p.add_argument("--commit_thr", type=float, default=1.2)

    # vigilance params (defaults tuned for this dataset)
    p.add_argument("--v_thr_base", type=float, default=1.0)
    p.add_argument("--v_thr_scale", type=float, default=2.0)
    p.add_argument("--v_tau_low", type=float, default=0.12)
    p.add_argument("--v_tau_high", type=float, default=0.95)
    p.add_argument("--v_ema_err", type=float, default=0.10)
    p.add_argument("--v_k_slope", type=float, default=10.0)
    p.add_argument("--v_k_level", type=float, default=3.0)

    # optional save
    p.add_argument("--save_npz", type=str, default="", help="if set, saves one generated dataset to this .npz path")
    return p.parse_args()


def make_agents(args) -> List[Agent]:
    return [
        GreedyArgmax(eps=args.eps, seed=0),
        SoftmaxAgent(tau=args.tau, seed=1),
        EMAValueAgent(beta=args.ema_beta, seed=2),
        FixedDwellAgent(dwell=args.dwell, base=GreedyArgmax(eps=args.eps, seed=10)),
        EvidentialCommitmentAgent(thr=args.commit_thr, seed=3),
        VigilanceAgent(
            thr_base=args.v_thr_base,
            thr_scale=args.v_thr_scale,
            tau_low=args.v_tau_low,
            tau_high=args.v_tau_high,
            ema_err=args.v_ema_err,
            k_slope=args.v_k_slope,
            k_level=args.v_k_level,
            seed=4,
        ),
    ]



# Backwards-compatible / descriptive aliases
CommitmentAgent = EvidentialCommitmentAgent
VigilantAgent = VigilanceAgent

def main():
    args = parse_args()

    results: Dict[str, List[RunStats]] = {}
    agents = make_agents(args)
    for ag in agents:
        results[ag.name] = []

    # Optionally save one dataset for inspection / plotting
    if args.save_npz:
        ds0 = generate_dataset(DatasetConfig(
            T=args.T, K=args.K, M=args.M,
            stable_mean=args.stable_mean, min_stable=args.min_stable, max_stable=args.max_stable,
            storm_len=args.storm_len,
            gap=args.gap, subgap=args.subgap,
            sigma=args.sigma, sigma_burst=args.sigma_burst, burst_prob=args.burst_prob, burst_len=args.burst_len,
            switch_cost=args.switch_cost, delay=args.delay, churn_penalty=args.churn_penalty,
            seed=0,
        ))
        np.savez_compressed(
            args.save_npz,
            regimes=ds0.regimes,
            change_points=ds0.change_points,
            means=ds0.means,
            sigmas=ds0.sigmas,
        )
        print(f"Saved dataset to {args.save_npz}")

    for s in range(args.seeds):
        ds = generate_dataset(DatasetConfig(
            T=args.T, K=args.K, M=args.M,
            stable_mean=args.stable_mean, min_stable=args.min_stable, max_stable=args.max_stable,
            storm_len=args.storm_len,
            gap=args.gap, subgap=args.subgap,
            sigma=args.sigma, sigma_burst=args.sigma_burst, burst_prob=args.burst_prob, burst_len=args.burst_len,
            switch_cost=args.switch_cost, delay=args.delay, churn_penalty=args.churn_penalty,
            seed=s,
        ))
        env = DatasetBanditEnv(ds, EnvConfig(
            switch_cost=args.switch_cost,
            delay=args.delay,
            churn_penalty=args.churn_penalty,
        ), seed=s)

        for ag in agents:
            st = run_episode(env, ag, ds, post_window=args.post_window, near_cp=args.near_cp)
            results[ag.name].append(st)

    print("\n=== Bursty Dataset Vigilance Experiment ===")
    print(f"T={args.T} K={args.K} M={args.M} stable_mean={args.stable_mean} (min={args.min_stable}, max={args.max_stable}) storm_len={args.storm_len}")
    print(f"gap={args.gap} sigma={args.sigma}  noise_bursts: prob={args.burst_prob} len={args.burst_len} sigma_burst={args.sigma_burst}")
    print(f"switch_cost={args.switch_cost} delay={args.delay} churn_penalty={args.churn_penalty}")
    print(f"metrics: post_window={args.post_window} near_cp={args.near_cp}")
    print()
    print(summarize(results))


if __name__ == "__main__":
    main()
