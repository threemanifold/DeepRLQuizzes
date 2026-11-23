# Source papers for the RL-for-LLMs quiz

`key` is the file name used in `paper-notes/<key>.md`. The arXiv IDs below were checked against arXiv on 2026-11-23. For the non-arXiv entries, if the link fails, search for the title.

| key | Paper | Source |
|---|---|---|
| instructgpt | Training language models to follow instructions with human feedback (Ouyang et al., 2022) | arXiv 2203.02155 |
| ppo | Proximal Policy Optimization Algorithms (Schulman et al., 2017) | arXiv 1707.06347 |
| gae | High-Dimensional Continuous Control Using Generalized Advantage Estimation (Schulman et al., 2015) | arXiv 1506.02438 |
| kl-blog | Approximating KL Divergence (J. Schulman, blog post): the k1/k2/k3 estimators | http://joschu.net/blog/kl-approx.html |
| kl-design | On the Design of KL-Regularized Policy Gradient Algorithms for LLM Reasoning (2025) | arXiv 2505.17508 |
| dpo | Direct Preference Optimization: Your Language Model is Secretly a Reward Model (Rafailov et al., 2023) | arXiv 2305.18290 |
| rloo | Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs (Ahmadian et al., 2024) | arXiv 2402.14740 |
| kool | Buy 4 REINFORCE Samples, Get a Baseline for Free! (Kool, van Hoof, Welling, 2019), the original leave-one-out baseline | OpenReview r1lgTGL5DE |
| grpo | DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (Shao et al., 2024), which introduces GRPO | arXiv 2402.03300 |
| drgrpo | Understanding R1-Zero-Like Training: A Critical Perspective (Liu et al., 2025), "Dr. GRPO": length and std-normalisation biases | arXiv 2503.20783 |
| dapo | DAPO: An Open-Source LLM Reinforcement Learning System at Scale (Yu et al., 2025) | arXiv 2503.14476 |
| gspo | Group Sequence Policy Optimization (Zheng et al., 2025) | arXiv 2507.18071 |
| cispo | MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention (2025), which introduces CISPO | arXiv 2506.13585 |
| vcppo | What's Behind PPO's Collapse in Long-CoT? Value Optimization Holds the Secret (Yuan et al., 2025), VC-PPO | arXiv 2503.01491 |
| vapo | VAPO: Efficient and Reliable Reinforcement Learning for Advanced Reasoning Tasks (Yue et al., 2025) | arXiv 2504.05118 |
| hybridflow | HybridFlow: A Flexible and Efficient RLHF Framework (Sheng et al., 2024), the verl paper | arXiv 2409.19256 |
| areal | AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning (Fu et al., 2025) | arXiv 2505.24298 |
| async-rlhf | Asynchronous RLHF: Faster and More Efficient Off-Policy RL for Language Models (Noukhovitch et al., 2024) | arXiv 2410.18252 |
| batch-inv | Batch size-invariance for policy optimization (Hilton et al., 2021), the origin of the decoupled proximal-policy objective | arXiv 2110.00641 |
| apex | Distributed Prioritized Experience Replay (Horgan et al., 2018), Ape-X | arXiv 1803.00933 |
| r2d2 | Recurrent Experience Replay in Distributed Reinforcement Learning (Kapturowski et al., ICLR 2019), R2D2 | OpenReview r1lyTjAqYX |
| impala | IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures (Espeholt et al., 2018), V-trace and truncated IS | arXiv 1802.01561 |
| pipelinerl | PipelineRL: Faster On-policy Reinforcement Learning for Long Sequence Generation (2025) | arXiv 2509.19128 |
| flashattn | FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (Dao et al., 2022) | arXiv 2205.14135 |
| sutton-barto | Reinforcement Learning: An Introduction (Sutton & Barto, 2nd ed.), chapter 12 on λ-returns | http://incompleteideas.net/book/the-book-2nd.html |

## Question-to-source index (`Quizzes/LLM-RL.md`)

"First principles" means there is no single source paper: derive the answer and check it yourself. A key in brackets is useful background, not the main source.

| Questions | Sources |
|---|---|
| 1 | instructgpt |
| 2–3 | instructgpt, ppo, gae |
| 4 | areal, async-rlhf, batch-inv |
| 5 | kl-blog, kl-design, [grpo] |
| 6 | dpo |
| 7 | rloo, kool |
| 8–9 | rloo, grpo, drgrpo |
| 10 | rloo, grpo |
| 11 | grpo, dapo |
| 12 | grpo, dapo, drgrpo |
| 13–14 | rloo, grpo |
| 15 | instructgpt, vcppo |
| 16 | gae, sutton-barto |
| 17–22 | dapo |
| 23–25 | gspo |
| 26 | ppo |
| 27–28 | cispo |
| 29–30 | gae, sutton-barto |
| 31 | first principles |
| 32 | vcppo, gae |
| 33 | gae (first principles) |
| 34–35 | vcppo |
| 36–37 | vapo, gae |
| 38–46 | hybridflow (Q45: a set of 4 models has Bell number B₄ = 15 partitions) |
| 47 | areal |
| 48 | areal, apex |
| 49–53 | areal |
| 54 | r2d2, areal |
| 55 | areal, pipelinerl |
| 56 | areal, batch-inv |
| 57 | areal |
| 58 | areal (a hypothetical variant: reason from first principles) |
| 59–65 | first principles, [hybridflow] |
| 66 | flashattn |
| 67 | first principles |
| 68–75 | pipelinerl |
| 76 | pipelinerl, impala, areal |
| 77 | first principles, [hybridflow] |
| 78–79 | first principles, [pipelinerl, areal] |
| 80 | first principles, [areal] |
| 81–83 | first principles |
| 84 | first principles, [pipelinerl] (ESS = (Σw)² / Σw², normalised ESS = ESS / n) |
| 85 | first principles, [pipelinerl] |
