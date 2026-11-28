# DeepRLQuizzes

A collection of quizzes covering many parts of deep learning and deep reinforcement learning, from algorithms to system design. Each quiz is a Markdown question bank. Solutions are left out on purpose, so you can work through the questions yourself or use them for an interactive quiz session.

## Table of Contents

| Quiz | Questions | Topics |
| --- | --- | --- |
| [Basic RL](Quizzes/Basic-RL.md) | 33 | Value functions, Bellman equations, optimality, TD learning, SARSA, policy gradients, λ-returns |
| [RL for LLMs](Quizzes/LLM-RL.md) | 85 | RLHF, PPO, GRPO, DPO and related methods; RL systems and distributed training; calculation exercises |
| [Distributed Systems](Quizzes/DistributedSystems.md) | 24 | Data parallelism, ZeRO/FSDP, tensor, pipeline and sequence parallelism, gradient bucketing, activation checkpointing, distributed inference |

## Interactive quizzing

If you use [Claude Code](https://claude.com/claude-code), the `rl-quiz` skill in `.claude/skills/` runs a tutor-style quiz from these banks. Start it with `/rl-quiz` or ask "quiz me".
