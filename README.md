# Revisiting the Relationship Between the Scale Factor _a(t)_ and Cosmic Time _t_

We test the hypothesis that the expansion history of the universe can be described not by the standard ΛCDM model, but by a **power-law scaling** of the form _a(t) ∝ t^α_ or even a **dynamically evolving exponent α(t)** modeled within a scalar–tensor framework.

## Datasets Used

- **Type Ia Supernovae (Pantheon+SH0ES)**  
  [https://pantheonplussh0es.github.io/](https://pantheonplussh0es.github.io/)

- **Gamma-Ray Bursts (GRB)**  
  Calibrated from Amati relation, see [arXiv:1610.00854](https://arxiv.org/abs/1610.00854)

- **Cosmic Chronometers (CC)**  
  Hubble parameter from differential galaxy ages, see [arXiv:2009.10701](https://arxiv.org/abs/2009.10701)

- **Planck 2018 CMB TT Power Spectrum**  
  Transformed into log-distance-like domain for joint likelihood fitting, see [esdcdoi.esac.esa.int](https://esdcdoi.esac.esa.int/doi/html/data/astronomy/planck/Cosmology.html)

---

## Key Results

| Dataset      | Model        | H₀    | α or Ωₘ | χ²       | AIC      | BIC      |
| ------------ | ------------ | ----- | ------- | -------- | -------- | -------- |
| **CC**       | ΛCDM         | 68.16 | 0.319   | 14.55    | 18.55    | 21.49    |
|              | Time-Scaling | 70.00 | 1.252   | 25.17    | 29.17    | 32.11    |
| **SN**       | ΛCDM         | 71.42 | 0.351   | 758.58   | 762.58   | 773.46   |
|              | Time-Scaling | 70.89 | 1.388   | 763.42   | 767.42   | 778.30   |
| **GRB**      | ΛCDM         | 75.00 | 0.500   | 167.36   | 171.36   | 177.53   |
|              | Time-Scaling | 75.00 | 1.000   | 174.97   | 178.97   | 185.15   |
| **CMB**      | ΛCDM         | 67.19 | —       | 84136.8  | 84138.8  | 84144.6  |
|              | Time-Scaling | 70.00 | 1.966   | 83939.5  | 83943.5  | 83955.1  |
| **Combined** | ΛCDM         | 65.00 | 0.357   | 626542.7 | 626550.7 | 626576.3 |
|              | Time-Scaling | 70.00 | 1.060   | 225288.4 | 225296.4 | 225321.9 |

> **ΔAIC = 401254.37** → Strong evidence in favor of Time-Scaling model in the global fit.

## Scalar–Tensor Dynamics

We model α(t) as a scalar field coupled to gravity via a Brans–Dicke–like action, and test three potentials:

- Quadratic: _V(α) ∝ α²_
- Cosine: _V(α) ∝ cos(α)_
- Asymmetric: _V(α) ∝ α³·sin(α)_

Key finding: **time-directionality emerges** under asymmetric potentials — Lyapunov analysis shows:

- Forward evolution: λₗ ~ 10⁻³
- Backward evolution: λₗ < 0

No thermodynamic or quantum mechanisms are required to induce the arrow of time.

## Repository Contents

```
.
├── model.py              # Full numerical analysis pipeline
├── results.txt           # Full fitting + MCMC + dynamics output
├── fig_obs.png           # Fits to SN, GRB, CC, and CMB
├── fig_obs_combined.png  # Combined observational fit
├── fig_dynamics.png      # Scalar field evolution (α(t))
├── fig_lyap.png          # Lyapunov exponent analysis
├── fig_mcmc.png          # Posterior samples for ΛCDM and Time-Scaling
└── README.md             # This file
```

## How to Run

### Requirements

Having Python ≥ 3.10, install dependencies:

```bash
pip install numpy scipy matplotlib pandas sympy emcee
```

### Run the Pipeline

```bash
python model.py
```

Outputs include:

- Full observational fitting (SN, GRB, CC, CMB)
- Global AIC/BIC comparison
- Posterior sampling (optional MCMC)
- Scalar dynamics and Lyapunov evolution
- All plots saved as PNG

## Citation

If using this code or results, please cite:

> Chudzik, A. (2025). Revisiting the relationship between the scale factor _a(t)_ and cosmic time _t_ using numerical analysis. _Mathematics_ (MDPI).
