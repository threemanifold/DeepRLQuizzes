# RL Questions

Reviewed questions only; solutions are omitted. Questions are added as we approve them.

1. **State-value function.** Define \(V^\pi(s)\) as an expected discounted sum of rewards, with precise conditioning and discount exponents. Use the convention that action \(A_t\) in state \(S_t\) leads to reward \(R_{t+1}\) and state \(S_{t+1}\). Specify the summation bounds for an infinite-horizon task and for an episode ending at terminal time \(T\).

2. Under what conditions is the infinite-horizon discounted return finite?

3. Write the Bellman expectation equation for \(V^\pi(s)\), both as an expectation and as an explicit sum.

4. Define \(Q^\pi(s,a)\) as an expected return. What role does \(\pi\) play in this definition?

5. Can \(Q^\pi(s,a)\) be defined when \(\pi(a\mid s)=0\)?

6. Write the Bellman expectation equation for \(Q^\pi(s,a)\), both as an expectation and as an explicit sum.

7. Let \(Q^*\) denote the action-value function of an optimal policy. Define \(Q^*(s,a)\) in terms of \(Q^\pi(s,a)\) and write its Bellman optimality equation.

8. Using the existence of a policy optimal from every state, derive the Bellman optimality equation for \(Q^*\).

9. What is a stationary policy? How does stationarity differ from determinism?

10. How does a finite horizon affect whether an optimal policy can be stationary?

11. Write the tabular TD(0) update and define the TD error.

12. How should TD bootstrapping handle true termination versus time-limit truncation?

13. Write the tabular SARSA update.

14. Write the tabular Q-learning update.

15. What are the behavior and target policies in Q-learning? Is it on-policy or off-policy, and why?

16. What are the behavior and target policies in SARSA? Is it on-policy or off-policy, and why?

17. Write the \(n\)-step bootstrapped return.

18. Write the finite-episode \(\lambda\)-return, including the final return's weight.

19. What targets do \(\lambda=0\) and \(\lambda=1\) recover? Which involve bootstrapping?

20. Define \(A^\pi(s,a)\). What is its expectation over \(a\sim\pi(\cdot\mid s)\)?

21. With an exact critic \(V^\pi\), what is \(\mathbb E[\delta_t\mid S_t=s,A_t=a]\)?

22. How is GAE defined in terms of the \(\lambda\)-return and the value estimate?

23. Derive the expression for GAE as a weighted sum of TD residuals.

24. For GAE computed from an \(H\)-step rollout:

    1. Can it be computed before the episode ends? Write the truncated TD-residual sum.
    2. Give the backward recursion and its boundary conditions.
    3. Write the equivalent mixture of \(n\)-step returns, including the final weight.
    4. With an exact critic and on-policy continuation, what is its expected value conditioned on \(S_t=s,A_t=a\)? Does it depend on \(\lambda\) or \(H\)?

25. If the critic is \(V=V^\pi+e\), express \(\mathbb E[\delta_t\mid S_t=s,A_t=a]\) in terms of the true advantage and critic error.

26. With \(V=V^\pi+e\), what is the expected full-episode GAE at \(\lambda=1\), conditioned on \(S_t=s,A_t=a\)? Assume true termination with \(V(S_T)=0\).

27. For a fixed state \(s\), prove:

    1. \(\mathbb E_{a\sim\pi_\theta(\cdot\mid s)}[\nabla_\theta\log\pi_\theta(a\mid s)]=0\).
    2. \(\mathbb E_{a\sim\pi_\theta(\cdot\mid s)}[b(s)\nabla_\theta\log\pi_\theta(a\mid s)]=0\) for any state-only baseline \(b(s)\).

28. Starting from \(J(\theta)=\mathbb E_{\tau\sim p_\theta}[G_0]\), derive the REINFORCE gradient in four steps. Assume the initial-state distribution and environment dynamics and rewards have no explicit dependence on \(\theta\).

    1. Show that \(\nabla_\theta J=\mathbb E[G_0\nabla_\theta\log p_\theta(\tau)]\). Why can \(G_0(\tau)\) be held fixed when differentiating?
    2. Factor the trajectory probability and express \(\nabla_\theta\log p_\theta(\tau)\) as a sum of policy log-probability gradients. Explain which terms disappear and why.
    3. Split \(G_0\) into past rewards and discounted reward-to-go. Condition on the history before \(A_t\) to remove the past-reward contribution and show that \(\nabla_\theta J=\mathbb E[\sum_t\gamma^tG_t\nabla_\theta\log\pi_\theta(A_t\mid S_t)]\).
    4. For trajectories of fixed length \(H\), replace the time sum by an expectation over a uniformly sampled time step. What factor is needed for an unbiased estimate?

29. With \(\gamma=1\) and a terminal binary correctness reward, write the objective and REINFORCE gradient in terms of token log probabilities.

30. Write the formula for GRPO's group-relative advantage. How does it differ from using raw binary rewards?

31. For GRPO:

    1. Write the clipped objective. Define its probability ratio and distinguish the policies used in the ratio and KL penalty. Explain the roles of clipping and the KL penalty.
    2. How does summing versus averaging token contributions affect the weighting of completions of different lengths?

32. For the infinite-horizon objective with \(0\leq\gamma<1\), derive the policy gradient theorem from the REINFORCE reward-to-go expression:

    1. Use the tower property, conditioning on \((S_t,A_t)\), to replace \(G_t\) by \(Q^{\pi_\theta}(S_t,A_t)\) inside the gradient expectation. Justify moving the policy score outside the inner expectation.
    2. Independently sample a time index \(K\) with \(\Pr(K=t)=(1-\gamma)\gamma^t\). Replace the discounted time sum by an expectation at time \(K\), retaining the correct factor.
    3. Define \(d_\gamma^{\pi_\theta}(s)=\Pr(S_K=s)\), express it as a discounted sum of state probabilities, and write the resulting policy gradient theorem.

33. In policy-gradient methods, consider replacing Monte Carlo returns with bootstrapped \(\lambda\)-returns in the formula

    \(\nabla_\theta J=\mathbb E_{\pi_\theta}[\sum_t\gamma^tG_t\nabla_\theta\log\pi_\theta(A_t\mid S_t)]\), where \(J(\theta)=\mathbb E_{\pi_\theta}[G_0]\).

    1. Give a sufficient conditional-expectation condition for replacing \(G_t\) by an estimator \(Y_t\) inside the policy gradient. Prove it using the tower property. Is matching unconditional expectations sufficient?
    2. Do on-policy \(\lambda\)-returns satisfy this condition with an exact critic? What can change with an approximate critic?
    3. Instead redefine the objective as \(\widetilde J(\theta)=\mathbb E_{\pi_\theta}[G_0^\lambda]\), with a frozen critic. Must its gradient match that of the original objective? Is critic exactness at the current \(\theta\) sufficient?
    4. If the critic depends explicitly on \(\theta\), derive \(\nabla_\theta\mathbb E_{\pi_\theta}[G_{0,\theta}^\lambda]\).
