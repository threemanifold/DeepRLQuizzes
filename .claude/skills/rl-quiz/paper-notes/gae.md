# High-Dimensional Continuous Control Using Generalized Advantage Estimation
Schulman et al., arXiv 1506.02438 (notes from HTML v6). Relevant to Q2, Q3, Q16, Q29, Q30, Q32, Q33, Q36, Q37.

## Definitions (Sec 3)
- TD residual: δ_t^V = r_t + γV(s_{t+1}) − V(s_t).
- Eq. 14, k-step: Â_t^(k) = Σ_{l=0}^{k−1} γ^l δ_{t+l} = −V(s_t) + r_t + … + γ^{k−1} r_{t+k−1} + γ^k V(s_{t+k}).
- Eq. 16, GAE: Â_t^{GAE(γ,λ)} = (1−λ)(Â^(1) + λÂ^(2) + λ²Â^(3) + …) = Σ_{l≥0} (γλ)^l δ_{t+l}.
- Eqs. 17–18: λ=0 gives δ_t; λ=1 gives Σ γ^l r_{t+l} − V(s_t).
- The paper says the construction is "closely analogous" to TD(λ): TD(λ) estimates V, GAE estimates A. It does not use the term "λ-return".

## Bias (Sec 2–3)
- γ-just (Def. 1, Eq. 7): E[Â_t ∇log π] = E[A^{π,γ} ∇log π]. Prop. 1: Q_t − b_t with E[Q_t|s_t,a_t] = Q^{π,γ} and a baseline b_t that depends only on the past.
- δ^V is γ-just only when V = V^{π,γ}. GAE(γ,1) is γ-just for any V but has high variance. GAE(γ,0) has low variance but is biased unless V is exact.
- γ < 1 biases the policy gradient regardless of V. λ < 1 introduces bias only when V is inaccurate. The best λ is much lower than the best γ.
- Sec 4 reward shaping: shaped reward r̃ = r + γΦ(s') − Φ(s). With Φ = V, GAE is the γλ-discounted sum of shaped rewards (Eq. 25).
  The response function (Eqs. 26–27) shows that terms with l ≫ 1/(1−γλ) are effectively ignored.

## Value fitting (Sec 5)
- Eq. 28: regress V_φ(s_n) onto V̂_t = Σ_l γ^l r_{t+l}, the Monte Carlo / TD(1) target.
- Footnote 2: the alternative TD(λ) target V̂^λ_t = V_old(s_t) + Σ_l (γλ)^l δ_{t+l}, i.e. the λ-return = V_old + GAE. The authors saw no difference from λ = 1.
- Eqs. 29–30: trust region on the value fit, (1/N) Σ ||V_φ − V_old||² / (2σ²) ≤ ε, solved with conjugate gradient.
- Sec 6.1: compute advantages with V_old before refitting V; fitting V first would bias the estimate (an overfit V gives a zero gradient).

## Empirical ranges
- Cart-pole: γ ∈ [0.96, 0.99], λ ∈ [0.92, 0.99]. 3D biped: γ ∈ [0.99, 0.995], λ ∈ [0.96, 0.99]. Quadruped: γ = 0.995, λ = 0.96 best.
- Sec 7: an intermediate λ in [0.9, 0.99] is usually best. λ = 0 has prohibitively large bias (App A.2).
