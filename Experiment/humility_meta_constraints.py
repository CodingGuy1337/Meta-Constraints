#!/usr/bin/env python3
"""
humility_meta_constraints.py
============================

Meta-Constraint: **Humility** (Belief update magnitude constraint)

Sequential inference over a binary hypothesis H uses a log-odds state s_t:

    s_{t+1} = s_t + g_t * ℓ_t
    p_t = σ(s_t)

where ℓ_t is the log-likelihood ratio (LLR) increment from observation x_t.

Humility constrains the gain g_t by normalizing evidence magnitude by its
recent variability (scale invariance):

    a_t = |ℓ_t|
    μ_t, σ_t  = EMA moments of a_t
    z_t = (a_t - μ_t - τ) / (σ_t + ε)
    g_t = g_min + (g_max - g_min) * sigmoid(-β z_t)

Ablation removes the σ_t normalization to demonstrate scale sensitivity.

The environment creates a spurious-correlation distribution shift:
pre-shift evidence is biased toward ¬H; post-shift evidence becomes reliable.
"""



from __future__ import annotations
import argparse
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


# -------------------------
# Utilities
# -------------------------

def sigmoid(x: float) -> float:
    if x >= 60:
        return 1.0
    if x <= -60:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def logistic(s: float) -> float:
    if s >= 60:
        return 1.0
    if s <= -60:
        return 0.0
    return 1.0 / (1.0 + math.exp(-s))


def ema(prev: float, x: float, rho: float) -> float:
    return (1.0 - rho) * prev + rho * x


# -------------------------
# Environment (spurious correlation shift)
# -------------------------

@dataclass
class EnvCfg:
    T: int = 450
    t_reveal: int = 250

    # pre-shift deception probability (optional); post-shift alpha=0
    alpha: float = 0.35

    # separability for normal evidence
    sep: float = 1.2

    # pre-shift spurious bias strength (points toward ¬H); post-shift bias=0
    bias: float = 1.1

    # single reveal observation strength (acts like a correction signal)
    sep_gold: float = 5.0


class SpuriousShiftEnv:
    """
    x_t | H_eff ~ N(mu, 1), mu determined by regime.

    Pre-shift (t < t_reveal):
      - With prob alpha, evidence is deceptive (as if ¬H).
      - Additionally, a spurious bias pushes x toward ¬H even when not deceptive.
        This is the distribution shift driver (spurious feature / representation mismatch).

    Shift point (t == t_reveal):
      - One 'gold' sample with large sep_gold, no deception/bias. (Optional but useful to force correction.)

    Post-shift (t > t_reveal):
      - Reliable evidence: alpha=0, bias=0, sep=sep
    """
    def __init__(self, cfg: EnvCfg, seed: int):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.H = int(self.rng.integers(0, 2))

    def sample(self, t: int) -> (float, float):
        if t == self.cfg.t_reveal:
            sep = self.cfg.sep_gold
            deceptive = False
            bias = 0.0
        elif t < self.cfg.t_reveal:
            sep = self.cfg.sep
            deceptive = (self.rng.random() < self.cfg.alpha)
            bias = self.cfg.bias
        else:
            sep = self.cfg.sep
            deceptive = False
            bias = 0.0

        H_eff = (1 - self.H) if deceptive else self.H
        mu = (+sep if H_eff == 1 else -sep)

        # bias pushes toward ¬H regardless of H
        mu += (-bias if self.H == 1 else +bias)

        x = float(self.rng.normal(mu, 1.0))
        return x, sep


def llr_increment(x: float, sep: float) -> float:
    # For N(±sep,1): log p(x|1)/p(x|0) = 2*sep*x
    return 2.0 * sep * x


# -------------------------
# Agents
# -------------------------

class AgentBase:
    name: str = "Agent"

    def reset(self):
        raise NotImplementedError

    def step(self, x: float, sep: float) -> Dict[str, float]:
        raise NotImplementedError


class NaiveBayes(AgentBase):
    name = "Naive-Bayes"

    def reset(self):
        self.s = 0.0

    def step(self, x: float, sep: float) -> Dict[str, float]:
        inc = llr_increment(x, sep)
        self.s += inc
        p = logistic(self.s)
        return {"p_internal": p, "p_report": p, "g": 1.0, "inc": inc}


class FixedGain(AgentBase):
    def __init__(self, g: float, name: str):
        self.g = g
        self.name = name
        self.reset()

    def reset(self):
        self.s = 0.0

    def step(self, x: float, sep: float) -> Dict[str, float]:
        inc = llr_increment(x, sep)
        self.s += self.g * inc
        p = logistic(self.s)
        return {"p_internal": p, "p_report": p, "g": self.g, "inc": inc}


class HumilityGateNorm:
    """
    Normalized gate (recommended):
      z = (a - mu - tau)/(std+eps)
      g = g_min + (g_max-g_min)*sigmoid(-beta*z)
    """
    def __init__(self, g_min: float, g_max: float, beta: float, tau: float, rho: float, var_floor: float):
        self.g_min = g_min
        self.g_max = g_max
        self.beta = beta
        self.tau = tau
        self.rho = rho
        self.var_floor = var_floor
        self.reset()

    def reset(self):
        self.mu = 0.0
        self.m2 = 0.0

    def compute_g(self, a_t: float) -> float:
        raw_var = self.m2 - self.mu * self.mu
        var_prev = max(self.var_floor, raw_var)
        std_prev = math.sqrt(var_prev)
        z = (a_t - self.mu - self.tau) / (std_prev + 1e-12)
        unit = sigmoid(-self.beta * z)
        return float(self.g_min + (self.g_max - self.g_min) * unit)

    def update(self, a_t: float):
        self.mu = ema(self.mu, a_t, self.rho)
        self.m2 = ema(self.m2, a_t * a_t, self.rho)


class HumilityGateNoNorm:
    """
    Ablation: NO volatility normalization.
      z = (a - mu - tau)
      g = g_min + (g_max-g_min)*sigmoid(-beta_nonnorm*z)

    This is intentionally scale-sensitive; it should degrade under SNR/scale changes.
    """
    def __init__(self, g_min: float, g_max: float, beta_nonnorm: float, tau: float, rho: float):
        self.g_min = g_min
        self.g_max = g_max
        self.beta = beta_nonnorm
        self.tau = tau
        self.rho = rho
        self.reset()

    def reset(self):
        self.mu = 0.0

    def compute_g(self, a_t: float) -> float:
        z = (a_t - self.mu - self.tau)
        unit = sigmoid(-self.beta * z)
        return float(self.g_min + (self.g_max - self.g_min) * unit)

    def update(self, a_t: float):
        self.mu = ema(self.mu, a_t, self.rho)



# ---------------------------------------------------------------------
# Terminology note (essay alignment)
# ---------------------------------------------------------------------
# s   : log-odds / logit belief state (s_t)
# inc : LLR increment (ℓ_t)
# g   : humility gain / step-size (g_t)
# mu,m2 : EMA moments over |ℓ_t|
# tau : margin before down-weighting
# ---------------------------------------------------------------------

class HumilityUpdate(AgentBase):
    name = "Humility-update"

    def __init__(self, gate):
        self.gate = gate
        self.reset()

    def reset(self):
        self.s = 0.0
        self.gate.reset()

    def step(self, x: float, sep: float) -> Dict[str, float]:
        inc = llr_increment(x, sep)
        a = abs(inc)
        g = self.gate.compute_g(a)
        self.s += g * inc
        self.gate.update(a)
        p = logistic(self.s)
        return {"p_internal": p, "p_report": p, "g": g, "inc": inc}


class HumilityProject(HumilityUpdate):
    name = "Humility-project"

    def step(self, x: float, sep: float) -> Dict[str, float]:
        d = super().step(x, sep)
        p = d["p_internal"]
        g = d["g"]
        q = g * p + (1.0 - g) * 0.5
        d["p_report"] = q
        return d


# -------------------------
# Metrics
# -------------------------

def compute_metrics(traj: List[Dict[str, float]], H: int, t_reveal: int,
                    conf_thr: float, pre_conf_thr: float,
                    rec_target: float, rec_window: int) -> Dict[str, float]:
    T = len(traj)

    # reported q metrics
    brier = 0.0
    oce = 0.0
    oce_n = 0
    cwr = 0.0
    abstain = 0

    # internal p (pre-shift)
    pre_conf = 0.0
    wrong_pre = 0.0
    pre_n = 0

    # recovery post-shift (EMA)
    rho = 2.0 / (rec_window + 1.0)
    ema_int: Optional[float] = None
    ema_rep: Optional[float] = None
    rec_int = math.inf
    rec_rep = math.inf

    for t, d in enumerate(traj):
        p = d["p_internal"]
        q = d["p_report"]

        # ----- reported -----
        brier += (q - H) ** 2
        conf = abs(q - 0.5)
        wrong = ((q >= 0.5) != (H == 1))
        if conf >= conf_thr and wrong:
            oce += conf * conf
            oce_n += 1
            cwr += 1.0
        if conf < conf_thr:
            abstain += 1

        # ----- internal pre-shift -----
        if t < t_reveal:
            pre_n += 1
            is_conf = abs(p - 0.5) >= pre_conf_thr
            if is_conf:
                pre_conf += 1.0
                if ((p >= 0.5) != (H == 1)):
                    wrong_pre += 1.0

        # ----- recovery after shift -----
        if t > t_reveal:
            p_true_int = p if H == 1 else (1.0 - p)
            p_true_rep = q if H == 1 else (1.0 - q)
            ema_int = p_true_int if ema_int is None else ema(ema_int, p_true_int, rho)
            ema_rep = p_true_rep if ema_rep is None else ema(ema_rep, p_true_rep, rho)
            if rec_int is math.inf and ema_int >= rec_target:
                rec_int = t - t_reveal
            if rec_rep is math.inf and ema_rep >= rec_target:
                rec_rep = t - t_reveal

    return {
        "brier": float(brier / T),
        "oce": float(oce / max(1, oce_n)),
        "cwr": float(cwr / T),
        "abstain": float(abstain / T),
        "pre_conf": float(pre_conf / max(1, pre_n)),
        "wrong_pre": float(wrong_pre / max(1, pre_n)),
        "rec_int": float(rec_int),
        "rec_rep": float(rec_rep),
    }


# -------------------------
# Runner / aggregation
# -------------------------

def run_seed(seed: int, cfg: EnvCfg, agents: List[AgentBase],
             conf_thr: float, pre_conf_thr: float, rec_target: float, rec_window: int) -> List[Dict[str, float]]:
    env = SpuriousShiftEnv(cfg, seed=seed)
    out = []
    for agent in agents:
        agent.reset()
        traj = []
        for t in range(cfg.T):
            x, sep_t = env.sample(t)
            traj.append(agent.step(x, sep_t))
        out.append({"agent": agent.name, **compute_metrics(traj, env.H, cfg.t_reveal, conf_thr, pre_conf_thr, rec_target, rec_window)})
    return out


def aggregate(all_runs: List[List[Dict[str, float]]]) -> List[Dict[str, float]]:
    by: Dict[str, List[Dict[str, float]]] = {}
    for run in all_runs:
        for row in run:
            by.setdefault(row["agent"], []).append(row)

    summary = []
    for agent, rows in by.items():
        def mstd(key: str):
            arr = np.array([r[key] for r in rows], dtype=float)
            return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0

        b_m, b_s = mstd("brier")
        o_m, o_s = mstd("oce")
        cw_m, cw_s = mstd("cwr")
        ab_m, ab_s = mstd("abstain")
        pc_m, pc_s = mstd("pre_conf")
        wp_m, wp_s = mstd("wrong_pre")

        def rec_stats(key: str):
            arr = np.array([r[key] for r in rows], dtype=float)
            fin = arr[np.isfinite(arr)]
            rate = float(len(fin) / len(arr))
            if len(fin) > 0:
                return float(fin.mean()), float(fin.std(ddof=1)) if len(fin) > 1 else 0.0, rate
            return float("inf"), 0.0, rate

        ri_m, ri_s, _ = rec_stats("rec_int")
        rr_m, rr_s, _ = rec_stats("rec_rep")

        summary.append({
            "agent": agent,
            "brier_mean": b_m, "brier_std": b_s,
            "oce_mean": o_m, "oce_std": o_s,
            "cwr_mean": cw_m, "cwr_std": cw_s,
            "abstain_mean": ab_m, "abstain_std": ab_s,
            "pre_conf_mean": pc_m, "pre_conf_std": pc_s,
            "wrong_pre_mean": wp_m, "wrong_pre_std": wp_s,
            "rec_int_mean": ri_m, "rec_int_std": ri_s,
            "rec_rep_mean": rr_m, "rec_rep_std": rr_s,
        })

    # stable ordering
    order = {
        "Naive-Bayes": 0,
        "FixedGain-low": 1,
        "FixedGain-high": 2,
        "Humility-update": 3,
        "Humility-project": 4,
        "Humility-update(no-norm)": 5,
        "Humility-project(no-norm)": 6,
    }
    summary.sort(key=lambda d: order.get(d["agent"], 99))
    return summary


def print_table(summary: List[Dict[str, float]], title: str):
    if title:
        print(title)
    header = (
        f"{'Agent':24} | {'Brier(q)':>10} | {'OCE(q)':>10} | {'CWR(q)':>8} | {'Abst(q)':>8} | "
        f"{'PreConf(p)':>10} | {'WrongPre(p)':>11} | {'RecInt':>8} | {'RecRep':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in summary:
        b = f"{r['brier_mean']:.4f}±{r['brier_std']:.4f}"
        o = f"{r['oce_mean']:.4f}±{r['oce_std']:.4f}"
        cw = f"{r['cwr_mean']:.3f}±{r['cwr_std']:.3f}"
        ab = f"{r['abstain_mean']:.3f}±{r['abstain_std']:.3f}"
        pc = f"{r['pre_conf_mean']:.3f}±{r['pre_conf_std']:.3f}"
        wp = f"{r['wrong_pre_mean']:.3f}±{r['wrong_pre_std']:.3f}"
        ri = f"{r['rec_int_mean']:.1f}±{r['rec_int_std']:.1f}" if math.isfinite(r["rec_int_mean"]) else "inf"
        rr = f"{r['rec_rep_mean']:.1f}±{r['rec_rep_std']:.1f}" if math.isfinite(r["rec_rep_mean"]) else "inf"
        print(f"{r['agent']:24} | {b:>10} | {o:>10} | {cw:>8} | {ab:>8} | {pc:>10} | {wp:>11} | {ri:>8} | {rr:>8}")
    print()


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=256)
    ap.add_argument("--T", type=int, default=450)
    ap.add_argument("--t_reveal", type=int, default=250)

    ap.add_argument("--alpha", type=float, default=0.35)
    ap.add_argument("--sep", type=float, default=1.2)
    ap.add_argument("--bias", type=float, default=1.1)
    ap.add_argument("--sep_gold", type=float, default=5.0)

    ap.add_argument("--conf_thr", type=float, default=0.25)
    ap.add_argument("--pre_conf_thr", type=float, default=0.45)
    ap.add_argument("--rec_target", type=float, default=0.9)
    ap.add_argument("--rec_window", type=int, default=30)

    # fixed gains
    ap.add_argument("--g_low", type=float, default=0.25)
    ap.add_argument("--g_high", type=float, default=0.85)

    # humility (normalized)
    ap.add_argument("--g_min", type=float, default=0.10)
    ap.add_argument("--g_max", type=float, default=1.00)
    ap.add_argument("--beta", type=float, default=6.0)
    ap.add_argument("--tau", type=float, default=0.15)
    ap.add_argument("--rho", type=float, default=0.06)
    ap.add_argument("--var_floor", type=float, default=1e-6)

    # humility (no-norm ablation) needs a different beta scale
    ap.add_argument("--beta_nonnorm", type=float, default=0.15)

    # sweeps
    ap.add_argument("--sweep_bias", nargs="*", type=float, default=None)
    ap.add_argument("--sweep_sep", nargs="*", type=float, default=None)

    args = ap.parse_args()

    def build_agents():
        gate_norm = HumilityGateNorm(args.g_min, args.g_max, args.beta, args.tau, args.rho, args.var_floor)
        gate_nonnorm = HumilityGateNoNorm(args.g_min, args.g_max, args.beta_nonnorm, args.tau, args.rho)

        return [
            NaiveBayes(),
            FixedGain(args.g_low, "FixedGain-low"),
            FixedGain(args.g_high, "FixedGain-high"),
            HumilityUpdate(gate_norm),
            HumilityProject(gate_norm),
            HumilityUpdate(gate_nonnorm).__class__(gate_nonnorm),  # placeholder fixed below
        ]

    # minor hack to name the no-norm agents without duplicating classes
    def make_humility_no_norm(project: bool):
        gate = HumilityGateNoNorm(args.g_min, args.g_max, args.beta_nonnorm, args.tau, args.rho)
        if project:
            a = HumilityProject(gate)
            a.name = "Humility-project(no-norm)"
            return a
        a = HumilityUpdate(gate)
        a.name = "Humility-update(no-norm)"
        return a

    def run_cfg(cfg: EnvCfg):
        agents: List[AgentBase] = [
            NaiveBayes(),
            FixedGain(args.g_low, "FixedGain-low"),
            FixedGain(args.g_high, "FixedGain-high"),
            HumilityUpdate(HumilityGateNorm(args.g_min, args.g_max, args.beta, args.tau, args.rho, args.var_floor)),
            HumilityProject(HumilityGateNorm(args.g_min, args.g_max, args.beta, args.tau, args.rho, args.var_floor)),
            make_humility_no_norm(project=False),
            make_humility_no_norm(project=True),
        ]
        runs = [run_seed(i, cfg, agents, args.conf_thr, args.pre_conf_thr, args.rec_target, args.rec_window) for i in range(args.seeds)]
        return aggregate(runs)

    base_cfg = EnvCfg(T=args.T, t_reveal=args.t_reveal, alpha=args.alpha, sep=args.sep, bias=args.bias, sep_gold=args.sep_gold)

    if args.sweep_bias:
        print("=== Epistemic Self-Restraint Experiment v5 (bias sweep; distribution shift) ===")
        print(f"seeds={args.seeds} T={args.T} t_reveal={args.t_reveal} alpha={args.alpha} sep={args.sep} sep_gold={args.sep_gold}")
        print(f"Gate norm beta={args.beta}; Gate no-norm beta_nonnorm={args.beta_nonnorm}\n")
        for b in args.sweep_bias:
            cfg = EnvCfg(**{**base_cfg.__dict__, "bias": float(b)})
            summary = run_cfg(cfg)
            print_table(summary, f"--- bias = {b:.2f} ---")
        return

    if args.sweep_sep:
        print("=== Epistemic Self-Restraint Experiment v5 (sep sweep; scale sensitivity) ===")
        print(f"seeds={args.seeds} T={args.T} t_reveal={args.t_reveal} alpha={args.alpha} bias={args.bias} sep_gold={args.sep_gold}")
        print(f"Gate norm beta={args.beta}; Gate no-norm beta_nonnorm={args.beta_nonnorm}\n")
        for s in args.sweep_sep:
            cfg = EnvCfg(**{**base_cfg.__dict__, "sep": float(s)})
            summary = run_cfg(cfg)
            print_table(summary, f"--- sep = {s:.2f} ---")
        return

    summary = run_cfg(base_cfg)

    print("=== Epistemic Self-Restraint Experiment v5 (distribution shift + ablation) ===")
    print(f"T={base_cfg.T} t_reveal={base_cfg.t_reveal} alpha={base_cfg.alpha} sep={base_cfg.sep} bias={base_cfg.bias} sep_gold={base_cfg.sep_gold}")
    print(f"Gate (normalized):   z=(|ℓ|-μ-τ)/(σ+ε), g = g_min+(g_max-g_min)*sigmoid(-β z)  β={args.beta}")
    print(f"Gate (NO-NORM ablation): z=(|ℓ|-μ-τ),      g = g_min+(g_max-g_min)*sigmoid(-β' z) β'={args.beta_nonnorm}")
    print(f"g in [{args.g_min},{args.g_max}] tau={args.tau} rho={args.rho}  conf_thr={args.conf_thr} pre_conf_thr={args.pre_conf_thr}")
    print(f"Recovery: EMA target={args.rec_target} window={args.rec_window}\n")
    print_table(summary, "")

    print("Interpretation for Option A audience:")
    print("- PreConf/WrongPre quantify premature commitment under spurious correlation (pre-shift).")
    print("- CWR/OCE quantify confident wrongness (miscalibration under shift).")
    print("- RecInt/RecRep quantify how quickly the system recovers after the shift breaks the spurious feature.")
    print("- The NO-NORM ablation should degrade especially under sep sweeps (scale changes), showing normalization is essential.")


if __name__ == "__main__":
    main()
