# Meta-Constraints for Coherent Learning Agents

This repository accompanies the paper **“Meta-Constraints for Coherent Learning Agents: A Structural Critique of Contemporary Learning Formulations.”**

The central claim of this project is that many persistent failure modes in learning systems do not arise primarily from poor objectives, insufficient data, or suboptimal optimization, but from a structural omission: the absence of explicit constraints on how learning dynamics are allowed to evolve over time.

Rather than proposing new objectives or architectures, this work identifies a small set of **meta-constraints**—constraints on belief updates, policy switching, representation continuity, and adaptation—that are necessary for coherent learning under noise, partial observability, and nonstationarity.

---

## Core Idea

Standard learning formulations often assume that coherence will emerge automatically as models scale and optimization improves. In practice, learning agents frequently exhibit:

- oscillatory policies under noise  
- premature or brittle certainty  
- delayed or missed adaptation to regime shifts  
- unstable behavior under partial observability  

This repository explores the hypothesis that such failures are structural and predictable when learning dynamics are unconstrained.

We focus on four meta-constraints:

1. **Latent State Continuity** — constrains how internal representations evolve over time  
2. **Commitment** — constrains the rate of policy switching under noisy evidence  
3. **Humility** — constrains belief update magnitude relative to evidence reliability  
4. **Vigilance** — constrains sensitivity to nonstationarity and regime change  

Each constraint is treated as a restriction on learning dynamics, not as a task-specific heuristic or performance optimization.

---

## Repository Structure

```
Meta-Constraints/
├── commitment_meta_constraints.py
├── humility_meta_constraints.py
├── latent_state_meta_constraints.py
├── vigilance_meta_constraints.py
├── paper.pdf
└── README.md
```

### File Descriptions

- **latent_state_meta_constraints.py**  
  Reference implementation illustrating latent state continuity versus memoryless representations.

- **commitment_meta_constraints.py**  
  Demonstrates policy inertia and evidential hysteresis to prevent oscillatory action switching.

- **humility_meta_constraints.py**  
  Implements scale- and variance-aware belief updates that enforce calibrated learning dynamics.

- **vigilance_meta_constraints.py**  
  Illustrates hazard-based adaptation to nonstationarity without conflating exploration and change detection.

- **paper.pdf**  
  The accompanying theoretical paper describing the motivation, formalization, and scope of the meta-constraint framework.

These implementations are intended as **executable clarifications**, not benchmark-optimized agents.

---

## Scope and Intent

This project is **not** intended to:

- claim state-of-the-art performance  
- propose a unified learning algorithm  
- replace task-specific objectives or architectures  

Instead, it aims to:

- make implicit assumptions in learning theory explicit  
- demonstrate predictable failure modes when constraints are absent  
- provide minimal, inspectable implementations of structural constraints  

The code is deliberately simple and modular to emphasize causal structure rather than performance.

---

## How to Use

Each Python file can be run independently and inspected directly. No special setup is required beyond standard scientific Python packages.

The code is best read alongside the paper, which provides the conceptual motivation and formal framing for each constraint.

---

## Citation

If you use or reference this work, please cite:

```
Anthony Johnson.
Meta-Constraints for Coherent Learning Agents: A Structural Critique of Contemporary Learning Formulations.
```

---

## License

This repository is provided for research and educational purposes. See the LICENSE file for details.

---

## Notes

This project is intentionally theory-forward. The primary contribution is the framing of learning failures as violations of structural constraints on learning dynamics, rather than as isolated algorithmic shortcomings.

Critical feedback and discussion are welcome.
