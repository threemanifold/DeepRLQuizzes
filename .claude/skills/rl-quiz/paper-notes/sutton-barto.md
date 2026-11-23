# Reinforcement Learning: An Introduction (Sutton & Barto, 2nd ed.), Ch. 12: λ-returns
Written from the standard textbook definitions and checked numerically (2026-11-23). Relevant to Q16, Q29, Q30.

Episode with terminal time T. Rewards are r_t, …, r_{T−1}, and V(s_T) = 0 at the terminal state.
- n-step return: G_t^{(n)} = Σ_{k=0}^{n−1} γ^k r_{t+k} + γ^n V(s_{t+n}) for t+n < T; for t+n ≥ T, G_t^{(n)} = G_t (the MC return).
- Eq. 12.3 (episodic λ-return): G_t^λ = (1−λ) Σ_{n=1}^{T−t−1} λ^{n−1} G_t^{(n)} + λ^{T−t−1} G_t.
  The weights sum to 1; all the mass after the episode ends goes to G_t.
- Recursive form: G_t^λ = r_t + γ[(1−λ)V(s_{t+1}) + λ G_{t+1}^λ], with boundary G_T^λ = 0 (and V(s_T) = 0), so G_{T−1}^λ = r_{T−1}.
  Truncated but non-terminal rollouts bootstrap with G_T^λ := V(s_T) instead.
- TD-error form: G_t^λ − V(s_t) = Σ_{l=0}^{T−t−1} (γλ)^l δ_{t+l} = GAE Â_t. Recursively, Â_t = δ_t + γλ Â_{t+1}.
- λ = 0: G_t^λ = r_t + γV(s_{t+1}), the TD(0) target. λ = 1: G_t^λ = G_t, the Monte Carlo return.
- Equivalence proof: use G_t^{(n)} = r_t + γ G_{t+1}^{(n−1)} with G_{t+1}^{(0)} = V(s_{t+1}) and G_t = r_t + γG_{t+1}, then regroup the sum.
- Critic target: G_t^λ = V_old(s_t) + Â_t (verl/PPO "returns = advantages + values").
