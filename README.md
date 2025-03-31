# Policy Optimization with DAgger and Reward-Weighted Regression

This repository implements a deep imitation learning framework that uses the DAgger (Dataset Aggregation) algorithm combined with reward-weighted regression for expert-guided policy optimization in a goal-reaching environment.

## Overview

The objective is to train a deep neural network policy to navigate an environment where the agent must reach and stop at a goal location within a fixed episode length. The challenge lies in overcoming the state distribution shift that occurs when the agent deviates from the expert’s demonstrations.

## The DAgger Algorithm

DAgger mitigates the compounding errors of behavior cloning by iteratively aggregating data from both expert demonstrations and the agent’s own interactions. The algorithm proceeds as follows:

1. **Initialization:**  
   Begin with a dataset of expert demonstrations:  
   $$ D_0 = \{(s, \pi^*(s)) \} $$  
   where \( \pi^*(s) \) is the expert action for state \( s \).

2. **Iteration:**  
   For \( i = 1, 2, \dots, N \):
   - Train the policy \( \pi_i \) on the aggregated dataset \( D_i \).
   - Execute \( \pi_i \) in the environment to collect a set of states \( S_i \).
   - Query the expert for actions on these states:  
     $$ \{(s, \pi^*(s)) : s \in S_i\} $$
   - Aggregate the data:  
     $$ D_{i+1} = D_i \cup \{(s, \pi^*(s)) : s \in S_i\}. $$

3. **Objective:**  
   Minimize the expected loss between the predicted and expert actions:
   $$ \mathcal{L}(\pi) = \mathbb{E}_{s \sim D} \left[ \ell(\pi(s), \pi^*(s)) \right] $$
   where \( \ell(\cdot,\cdot) \) is a suitable loss function (e.g., mean squared error).

## Optimized DAgger with Beta Scheduling

In the optimized variant, a beta parameter \( \beta \) controls the probability of querying the expert:
- **High \( \beta \):** Greater reliance on the expert.
- **Low \( \beta \):** Increased reliance on the learned policy.

This beta scheduling strategy reduces unnecessary expert queries while ensuring the policy still receives sufficient corrective feedback. Comparative performance is visualized in the provided plots:
- **with_beta.png:** Performance with beta scheduling.
- **without_beta.png:** Baseline performance without beta scheduling.

## Reward-Weighted Regression

Reward-weighted regression further refines policy learning by assigning weights to samples based on their rewards. Higher-quality experiences are given greater importance during training, enabling the model to focus on trajectories that yield better outcomes.

## Results and Analysis

- **Expert Queries vs. Success Rate:**  
  Analysis shows that the optimized DAgger (with beta scheduling) achieves comparable or improved success rates while significantly reducing the number of expert queries.
- **Mathematical Insight:**  
  By reducing dependency on the expert and emphasizing higher-reward trajectories, the policy better learns recovery behaviors and robust performance in complex scenarios.

## Conclusion

This project demonstrates a comprehensive implementation of deep imitation learning using DAgger and reward-weighted regression. The optimized approach balances expert input with autonomous learning, leading to efficient policy improvement for challenging, goal-reaching tasks. Future work may explore further adaptive strategies to minimize expert dependency while maximizing learning efficiency.
