# Path-Enhanced FK-PINNs

**Stochastic physics-informed neural networks: Feynman–Kac path signatures for high-dimensional PDEs**

Federico Ganzer · IIB project, Department of Engineering, University of Cambridge

> [!NOTE]
> **Active work in progress.** This repository tracks ongoing research. Code, notation and results will change, and new work will be added here as the research goes on. Nothing in it should be read as a final result yet.

## Overview

Physics-informed neural networks (PINNs) solve PDEs without a mesh by penalising the PDE residual at sampled points (Raissi et al., 2019). Training on the residual inherits the conditioning of $`\mathcal{L}^*\mathcal{L}`$, which is set by the PDE operator, and often stalls (De Ryck et al., 2024). Tepakbong et al. (2026) add Monte Carlo point labels at a few collocation points, computed from the Feynman–Kac (FK) representation of the solution. They call the result an *FK-PINN*. The point labels add an identity term to the loss metric, so convergence on that term is governed by the spectrum of the network's neural tangent kernel (NTK) (Jacot et al., 2018). That spectrum decays quickly, so high-frequency error still converges slowly.

The simulated paths behind each FK label also carry noisy but unbiased information about $`\nabla u, \nabla^2 u, \dots`$. Ordinary FK-PINNs discard it, because every path is reduced to a single number before averaging (Figure 2). This project uses that information to supervise the network's derivatives as well as its values. The derivative labels are coefficients of the FK payoff in an orthogonal basis of **path signatures** (Chevyrev et al., 2026), computed on the same paths as the FK labels. A path signature is the sequence of iterated integrals of a path (Chevyrev & Kormilitzin, 2016; Lyons & McLeod, 2025). Each level of the hierarchy is weighted by its Parseval norm, which makes the extra loss term a Sobolev-type preconditioner.

<p align="center">
  <img src="figures/presentations/nb00_architecture.png" alt="Training loss: PINN residual, FK supervision and a signature-based term" width="95%">
</p>

**Figure 1.** The training loss. Top: the standard PINN residual, computed by automatic differentiation. Middle: FK supervision, which regresses $`u_\theta`$ onto a Monte Carlo average over SDE paths. Bottom: the question this project works on, which is how to combine the signatures $`S(\hat X)`$ of the same paths with $`u_\theta`$ to form a further loss term.

## Problem setting

Two classes of PDE are considered. Both have an FK representation driven by the SDE $`dX = b(X)\thinspace dt + \sigma(X)\thinspace dW`$, $`X_0 = x`$, whose generator is $`\mathcal{L} = b\cdot\nabla + \tfrac12\mathrm{tr}(\sigma\sigma^\top\nabla^2)`$.

**Parabolic.** $`\partial_t u = \mathcal{L}u`$ with initial condition $`u(\cdot,0) = u_0`$. The FK representation is $`u(x,t) = \mathbb{E}[Y_{x,t}]`$ with payoff $`Y_{x,t} = u_0(X^x_t)`$.

**Elliptic.** $`\mathcal{L}u - Vu = -f`$ in $`\Omega`$, with $`u = g`$ on $`\partial\Omega`$. The payoff is a functional of the path stopped at its exit time $`\tau`$ from $`\Omega`$:

```math
u(x) = \mathbb{E}_x\Big[\int_0^\tau e^{-\int_0^s V(X_r)\thinspace dr} f(X_s)\thinspace ds + e^{-\int_0^\tau V(X_r)\thinspace dr}\thinspace g(X_\tau)\Big].
```

An FK label at $`x`$ is the Monte Carlo average of the payoff over $`M`$ Euler–Maruyama paths.

<p align="center">
  <img src="figures/presentations/nb00_fk_funnel.png" alt="64 paths reduced to 64 path scores and then to one FK label" width="90%">
</p>

**Figure 2.** How an FK label is formed. 64 paths of 33 steps in 2D (4,224 numbers) are each scored by the FK functional, giving 64 numbers. These are averaged into one label $`\bar u`$. Ordinary FK supervision keeps only this last number.

Related methods also train on stochastic representations of the PDE. Deep BSDE (Han, Jentzen & E, 2018) and forward–backward stochastic neural networks (Raissi, 2018) recast parabolic PDEs as backward SDEs. The derivative-free loss method uses FK walkers inside the loss, and Han & Lee (2026) analyse its bias and trainability.

## Why supervise derivatives: a conditioning argument

Write $`e = u_\theta - u`$ for the network's error. Around the current parameters, gradient-flow training is approximately

```math
\dot e = -K\mathcal{M}e + \epsilon, \qquad (Kg)(x) = \int \sum_j \partial_{\theta_j}u_\theta(x)\thinspace\partial_{\theta_j}u_\theta(x')\thinspace g(x')\thinspace dx',
```

where the dot is a derivative in training time, $`K`$ is the NTK integral operator and $`\mathcal{M}`$ is the metric operator of the loss. Suppose, heuristically, that $`K`$ and $`\mathcal{M}`$ share Fourier modes, with eigenvalues $`\mu_k`$ and $`m_k`$. Then error mode $`k`$ decays at rate $`\mu_k m_k`$, and the condition number over the resolved band is $`\kappa = \max_k \mu_k m_k / \min_k \mu_k m_k`$.

| Loss term | Metric $`\mathcal{M}`$ | Multiplier $`m_k`$ | Effect |
|---|---|---|---|
| FK value labels | $`I`$ | $`1`$ | $`\kappa = \kappa(K)`$, so spectral bias remains |
| Second-order PDE residual | $`\mathcal{L}^*\mathcal{L}`$ | $`\sim\lvert k\rvert^4`$ | Overcompensates and moves the stiffness to high frequencies |
| Sobolev (derivative labels) | $`\sum_n \lambda_n(\nabla^n)^*\nabla^n`$ | $`\sum_n \lambda_n\lvert k\rvert^{2n}`$ | Well-chosen $`\lambda_n`$ flatten $`\mu_k m_k`$ |

The catch is noise. With noisy derivative labels $`g_n = \nabla^n u + \xi_n`$, the dynamics gain a forcing term $`K\sum_n \lambda_n(\nabla^n)^*\xi_n`$. This weights the noise in mode $`k`$ by $`\lambda_n\lvert k\rvert^n`$, so the weights that improve conditioning also amplify high-frequency label noise. The research question is whether Monte Carlo derivative labels from the FK paths are cheap and quiet enough to give the Sobolev conditioning gain without amplifying noise too much, and which weights $`\lambda_n`$ to use.

## Current approach: a signature hierarchy of FK labels

Path signatures give a natural coordinate system on the simulated paths. Gram–Schmidt orthogonalisation of the time-augmented Itô signature, under the Wiener measure, gives an orthogonal polynomial basis $`\lbrace\Phi_w\rbrace`$ on path space, indexed by words $`w`$ (Chevyrev et al., 2026). Any square-integrable FK payoff then has a polynomial chaos expansion

```math
Y = \sum_w \beta_w \Phi_w, \qquad \beta_w = \frac{\mathbb{E}[Y\Phi_w]}{\langle \hat p_w, \hat p_w\rangle}.
```

- Each coefficient $`\beta_w`$ is a scalar expectation. Monte Carlo on the paths that already produce the point label estimates it, with no further path simulation.
- Orthogonality holds under the law of the driving Brownian motion, which is the measure the FK labels are computed under.
- On Markov problems the coefficients recover the derivatives of $`u`$ up to known constants, so the hierarchy supplies the derivative labels that Sobolev supervision needs.
- Parseval's identity fixes a weight for each level, so the per-level loss weights need no hand-tuning.

### The loss term

The new loss term is the model's residual in chaos space, over all words $`W_N`$ up to length $`N`$:

```math
R_\mathrm{sig} = \mathbb{E}\Big[\Big(\sum_{w\in W_N}\big(\beta^\theta_w - \beta_w\big)\Phi_w\Big)^2\Big] = \sum_{w\in W_N}\langle \hat p_w, \hat p_w\rangle\thinspace\delta\beta_w^2, \qquad \delta\beta_w = \frac{\mathbb{E}[(Y^\theta - Y)\Phi_w]}{\langle \hat p_w, \hat p_w\rangle}.
```

Here $`Y^\theta`$ is the network's counterpart of the payoff on the same path. The error enters through

```math
Y^\theta - Y = e(x,t) + \int_0^t \nabla_\sigma e(X_s,\thinspace t-s)\cdot dW_s .
```

$`R_\mathrm{sig}`$ checks the network for consistency against each path, where rough errors do not cancel. Every word adds its own non-negative error, so the loss metrics are nested, $`\mathcal{M}_0 \preceq \dots \preceq \mathcal{M}_N`$.

**Empty word.** $`\Phi_\emptyset = 1`$, so $`\delta\beta_\emptyset = e(x,t)`$. This is the ordinary FK point loss.

**Non-empty words.** These have $`\mathbb{E}[\Phi_w] = 0`$, so they see only the stochastic part of $`Y^\theta - Y`$. Summed over all non-empty words, the Itô isometry gives

```math
\sum_{w\neq\emptyset}\langle \hat p_w, \hat p_w\rangle\thinspace\delta\beta_w^2 = \int_0^t \mathbb{E}\big[\lvert\nabla_\sigma e(X_s,\thinspace t-s)\rvert^2\big]\thinspace ds .
```

The non-empty words therefore supervise the gradient of the network's error along every simulated path, at every intermediate time, without explicit gradient labels. The point loss supervises only the value at $`(x,t)`$. On the solution manifold, the empty word carries the mean of $`e_0(X_t)`$, which is all an FK point label sees. The other words carry its variance, which for a general diffusion is the carré du champ integral (Bakry, Gentil & Ledoux, 2014).

The truncation $`W_N`$ sets the order of the supervision: level $`n`$ measures the $`(n-1)`$-th Malliavin derivative of the gradient error. For the heat equation on the solution manifold, level $`n`$ equals $`\frac{(2\nu t)^n}{n!}\lVert\nabla^n e(x,t)\rVert_F^2`$. So $`R_\mathrm{sig}`$ is an $`H^N`$ Sobolev loss with Parseval weights.

### Preconditioning effect

Let $`\mathcal{L}`$ be self-adjoint in $`L^2(\pi)`$ with $`\mathcal{L}\varphi_k = -\lambda_k\varphi_k`$, and write $`e(\cdot,t) = \sum_k a_k\varphi_k`$. On the solution manifold, the point loss is $`\sum_k a_k^2`$ while the full $`R_\mathrm{sig}`$ is $`\sum_k e^{2t\lambda_k}a_k^2`$. The FK average damps mode $`k`$ by $`e^{-t\lambda_k}`$, so the point loss gives a vanishing gradient for rough error modes. $`R_\mathrm{sig}`$ restores it. The truncated metric satisfies

```math
I = \mathcal{M}_0 \preceq \mathcal{M}_N \preceq \mathcal{M}_\infty = e^{-2t\mathcal{L}},
```

and for the heat equation it is the multiplier $`w_N(\lambda) = \sum_{n\le N}(2t\lambda)^n/n!`$ with $`\lambda = \nu\lvert\xi\rvert^2`$.

As an idealisation, suppose $`K`$ shares the modes of $`\mathcal{L}`$ with eigenvalues $`g(\lambda_k)`$. Mode $`k`$ then converges at rate $`g(\lambda_k)\thinspace w_N(\lambda_k)`$ instead of $`g(\lambda_k)`$. Under spectral bias $`g`$ decreases with roughness (Rahaman et al., 2019), while $`w_N`$ increases. If $`g\thinspace w_N`$ stays non-increasing on the trained band $`\lambda \le \lambda_{\max}`$, then

```math
\kappa(K\mathcal{M}_N) = \frac{\kappa(K)}{w_N(\lambda_{\max})}.
```

If $`w_N`$ grows faster than $`g`$ decays, $`g\thinspace w_N`$ turns upward and a large $`N`$ makes $`\kappa`$ worse. The truncation level is therefore the order of the preconditioner. Whether a finite $`N`$ lowers $`\kappa`$ for a real network is an empirical question, which the experiments below test.

### Computational cost

- **Number of words.** Words grow like $`(1+d)^N`$, while the symmetric jets that matter for the heat equation grow only like $`\binom{d+N}{N}`$. Orthogonalisation is free, because the Gram matrix is block-diagonal over $`\sim_0`$ classes (Chevyrev et al., 2026). The cost lies in evaluating words, so the plan is to keep $`N \le 3`$ and subsample words with importance weights, which keeps the estimate of $`R_\mathrm{sig}`$ unbiased.
- **Noise at higher orders.** A single level-$`n`$ coefficient has Monte Carlo variance of order $`n!/(t^n M)`$. The Parseval weight $`t^n/n!`$ cancels this per word, but the summed noise still grows with the number of words, roughly as $`d^n/M`$. Estimating $`\delta\beta_w`$ from $`Y^\theta - Y`$ on the same paths cancels the label noise at the solution, up to time discretisation.
- **Network-side cost.** The model coefficients need derivatives of $`u_\theta`$ along every simulated path, contracted with the tangent flow when the drift is non-trivial. Nested reverse-mode autograd is exponential in the order, so forward- or Taylor-mode differentiation is used instead (Bettencourt et al., 2019). Expected cost is roughly 2–4× the per-step cost of plain FK at $`N = 2`$.
- **Bias.** Derivative labels near $`\partial\Omega`$ inherit, and likely amplify, the $`O(\sqrt{\Delta t})`$ exit-overshoot bias. On Euler paths the martingale identity, and with it the noise cancellation, holds only approximately. Per-level bias will be reported against distance to the boundary.

## Planned experiments

The full proposal is in [experiments_proposals/sighierarchyfkpinn.tex](experiments_proposals/sighierarchyfkpinn.tex).

### Hypotheses

- **H1. Path supervision conditions training.** With noise-free targets, adding $`R_\mathrm{sig}`$ to the FK loss lowers $`\kappa(K\mathcal{M})`$ and the error at matched steps. The gain grows with $`w_N(\lambda_{\max})`$, that is with the roughness of $`u`$, the horizon $`t`$ and the level $`N`$, until $`g\thinspace w_N`$ turns upward. *Test:* D against A\*, swept over frequency, $`t`$ and $`N`$. *Falsifier:* D ≈ A\* at the highest frequency, or a measured $`\kappa`$ ratio that does not grow with $`w_N(\lambda_{\max})`$.
- **H2. Parseval weights.** The Parseval weights match the best per-level weights found by tuning. *Test:* D against D′ at equal tuning budget. *Falsifier:* D′ beats D beyond seed noise. The weights then save tuning but carry no further benefit.
- **H3. Monte Carlo targets keep the gain.** Estimated on the same paths as $`Y^\theta`$, Monte Carlo targets retain most of the noise-free gain. *Test:* B against D, and B against $`\mathrm{B}_\mathrm{off}`$. *Falsifier:* B ≈ A0. A gap B > $`\mathrm{B}_\mathrm{off}`$ confirms that the same-path cancellation removes the label noise.
- **H4. Signature-specific value.** The signature estimator beats estimators that supply the same information without signatures. *Test:* B against C (pathwise derivatives) and against F (the untruncated path residual). *Falsifier:* B ≈ C and B ≤ F on every benchmark. The contribution is then the preconditioner interpretation and its weights, not the signature estimator.

Arm E is the control for all four. If E ≈ B, the gain comes from regularisation rather than information.

### Arms

| Arm | Added loss term | Targets | Tests |
|---|---|---|---|
| P | None (residual and boundary terms only) | – | Context: Table 1 of Tepakbong et al. (2026) |
| A0 | FK point loss | Monte Carlo point labels | Published baseline |
| A\* | FK point loss | Exact point labels | Reference for H1 |
| D | $`R_\mathrm{sig}`$, $`N\in\lbrace 1,2,3\rbrace`$ | Exact derivatives (heat, OU); large-$`M`$ reference otherwise | H1; gate for the rest |
| D′ | Derivative loss, tuned $`\lambda_n`$ | Exact derivatives | H2 |
| B | $`R_\mathrm{sig}`$ | Monte Carlo, from $`Y^\theta - Y`$ on the same paths | Method under test; H3, H4 |
| $`\mathrm{B}_\mathrm{off}`$ | $`R_\mathrm{sig}`$ | Monte Carlo, $`\hat\beta_w`$ precomputed offline | H3 |
| C | Derivative loss, Parseval weights | Monte Carlo pathwise derivatives $`\nabla^n u_0(X_t)`$ with the flow Jacobian | H4 |
| F | $`\mathbb{E}[(Y^\theta - Y)^2]`$, untruncated | Monte Carlo, same paths | H4; FBSNN-type (Raissi, 2018) |
| E | As B | B's targets permuted across $`x`$ within $`t`$-bins | Control |

### Benchmarks

The benchmarks are ordered by how far they depart from the setting of the theory: an endpoint payoff, a fixed horizon, and a generator that commutes with $`\nabla_\sigma`$. Each one tests a specific claim.

| Benchmark | Why | Tests | Arms |
|---|---|---|---|
| Heat, $`d=1`$, $`u_0=\sin(\pi x)`$ | Closed-form check of the hierarchy and of Stroock's identity. A unit test, not evidence. | – | All |
| Anisotropic heat, $`d\in\lbrace 2,5,10\rbrace`$ | Per-letter weights $`2\nu_i t\thinspace\xi_i^2`$, and noise growth with the number of words | H1, H3 | All |
| Ornstein–Uhlenbeck, $`d\in\lbrace 1,2,5\rbrace`$ | Linear drift. Time-letter words are non-zero, so the time truncation matters. Closed form via Mehler. | H1, H4 | All |
| Manufactured parabolic, $`u^*=\sin\pi x+\varepsilon\sin k\pi x`$, $`k\in\lbrace 1,4,16\rbrace`$ | Direct test of the predicted scaling of the gain with $`w_N(\lambda_{\max})`$ | H1, H2 | All |
| Poisson (Tepakbong et al., 2026) | Comparability with the incumbent. Elliptic, so there is no fixed horizon and the Parseval weights are undefined. | H1 | P, A0, A\*, D′ |
| Mean exit time and committor (Tepakbong et al., 2026) | Stopped payoffs and boundary layers. The committor payoff is discontinuous, so pathwise derivatives (arm C) do not exist. | H4 | P, A0, A\*; B after the stopped-path extension |

For the parabolic benchmarks, $`\nu`$ and $`T`$ are chosen with $`2\nu T\pi^2k^2 \gtrsim 1`$. Otherwise $`w_N \approx 1`$ and no effect can appear. $`w_N(\lambda_{\max})`$ is reported for each benchmark, so that the prediction of H1 is quantitative.

## Also under investigation: signature control variates

The FK label is a Monte Carlo average, so its noise limits how much FK supervision can help. Some coordinates of the driver's Itô signature have expectations known in closed form. For example, a word ending in a Brownian letter gives a stopped Itô integral, which has mean zero when the stopping time is bounded. These coordinates come for free with the paths already simulated and can be correlated with the path payoff, which makes them candidate control variates: they could reduce the variance of the FK label without simulating more paths. How much variance they remove, and how they interact with the signature-hierarchy loss, is being studied.

## Status

- [x] PINN and FK-PINN baselines (arms P and A0): parabolic problems (heat, advection–diffusion–reaction) and the four elliptic benchmarks of Tepakbong et al. (2026) (Poisson, Schrödinger, mean exit time, committor)
- [x] Euler–Maruyama FK path generation, for fixed horizons and exit times
- [x] Batched, differentiable truncated signatures in PyTorch (Stratonovich)
- [x] Experiment proposal for the signature hierarchy (hypotheses H1–H4, arms, benchmarks)
- [ ] Time-augmented Itô signatures along Euler–Maruyama paths
- [ ] Orthogonal signature basis (Gram–Schmidt under the Wiener measure)
- [ ] $`R_\mathrm{sig}`$ loss with Parseval weights, using Taylor-mode derivatives of $`u_\theta`$ along paths
- [ ] Remaining parabolic benchmarks: anisotropic heat, Ornstein–Uhlenbeck, manufactured solution
- [ ] Exact-label arms (A\*, D, D′), Monte Carlo arms (B, $`\mathrm{B}_\mathrm{off}`$, C, F) and control (E)
- [ ] Signature control variates for FK labels
- [ ] Paper draft (`paper/`)

## Repository layout

```
src/
  pdes.py          parabolic PDE specs shared by the PINN (autodiff) and FK (SDE) routes
  elliptic.py      elliptic Dirichlet problems with exit-time FK representations
  domains.py       geometries for elliptic problems: membership, sampling, reflection
  sde.py           Euler–Maruyama integration and FK path generation
  signatures.py    truncated path signatures in pure PyTorch (batched, differentiable)
  models.py        plain MLP, held fixed across methods so only the loss changes
  losses/
    residual.py    autodiff PINN residual and initial/boundary terms
    fk_loss.py     FK loss: regress the network onto a Monte Carlo FK estimate
    fk_pinn.py     FK-PINN loss for elliptic problems, after Tepakbong et al. (2026)
  reference.py     reference solutions for 2D elliptic problems (exact or finite differences)
  metrics.py       error metrics and Monte Carlo diagnostics
  plot_style.py    shared figure style
experiments_proposals/   experiment proposals (LaTeX)
figures/                 figures for presentations and the paper
```

## Setup

Developed with Python 3.11.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Modules in `src/` import each other by top-level name (for example `from losses.residual import gradient`), so add `src/` to the Python path:

```bash
export PYTHONPATH=src            # PowerShell: $env:PYTHONPATH = "src"
```

## References

- Bakry, D., Gentil, I. & Ledoux, M. (2014). *Analysis and Geometry of Markov Diffusion Operators*. Springer.
- Bettencourt, J., Johnson, M. J. & Duvenaud, D. (2019). Taylor-mode automatic differentiation for higher-order derivatives in JAX. *NeurIPS 2019 Workshop on Program Transformations for Machine Learning*.
- Chevyrev, I., Ferrucci, E., Lee, D., Lyons, T., Oberhauser, H. & Tapia, N. (2026). Orthogonal polynomials on path-space. [arXiv:2602.18808](https://arxiv.org/abs/2602.18808).
- Chevyrev, I. & Kormilitzin, A. (2016). A primer on the signature method in machine learning. [arXiv:1603.03788](https://arxiv.org/abs/1603.03788).
- De Ryck, T., Bonnet, F., Mishra, S. & de Bézenac, E. (2024). An operator preconditioning perspective on training in physics-informed machine learning. *International Conference on Learning Representations (ICLR)*.
- Han, J., Jentzen, A. & E, W. (2018). Solving high-dimensional partial differential equations using deep learning. *Proceedings of the National Academy of Sciences*, 115(34), 8505–8510. [arXiv:1707.02568](https://arxiv.org/abs/1707.02568).
- Han, J. & Lee, Y. (2026). An analysis of the derivative-free loss method for solving PDEs. *Research in the Mathematical Sciences*, 13, 47. [doi:10.1007/s40687-026-00633-y](https://doi.org/10.1007/s40687-026-00633-y).
- Jacot, A., Gabriel, F. & Hongler, C. (2018). Neural tangent kernel: convergence and generalization in neural networks. *Advances in Neural Information Processing Systems (NeurIPS)*.
- Lyons, T. & McLeod, A. D. (2025). Signature methods in machine learning. [arXiv:2206.14674](https://arxiv.org/abs/2206.14674).
- Rahaman, N., Baratin, A., Arpit, D., Draxler, F., Lin, M., Hamprecht, F. A., Bengio, Y. & Courville, A. (2019). On the spectral bias of neural networks. [arXiv:1806.08734](https://arxiv.org/abs/1806.08734).
- Raissi, M. (2018). Forward–backward stochastic neural networks: deep learning of high-dimensional partial differential equations. [arXiv:1804.07010](https://arxiv.org/abs/1804.07010).
- Raissi, M., Perdikaris, P. & Karniadakis, G. E. (2019). Physics-informed neural networks: a deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686–707.
- Tepakbong, N., Hu, H., Liu, C. & Zhou, X. (2026). Taming the loss landscape of PINNs with noisy Feynman–Kac supervision: operator preconditioning and non-asymptotic error bounds. *Proceedings of the 43rd International Conference on Machine Learning (ICML)*, PMLR 306. [arXiv:2606.00643](https://arxiv.org/abs/2606.00643).
