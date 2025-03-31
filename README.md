# Policy Optimization with DAgger and Reward-Weighted Regression

## Abstract

This repository implements a deep imitation learning framework using the DAgger (Dataset Aggregation) algorithm combined with reward-weighted regression for expert-guided policy optimization in a goal-reaching environment.

## Introduction

Behavior cloning often suffers from compounding errors when the agent encounters states outside the expert demonstration distribution. To address this, the DAgger algorithm iteratively aggregates expert data with the agent’s own experiences, refining the policy over successive iterations. In addition, reward-weighted regression emphasizes higher-quality trajectories during training, further enhancing performance.

## Methodology

### DAgger Algorithm

We initialize with a dataset of expert demonstrations:
$$
D_0 = \{(s, \pi^*(s))\}
$$
where \(\pi^*(s)\) is the expert action for state \(s\). For iterations \(i = 1, 2, \dots, N\), the following steps are performed:

1. **Policy Training:**  
   Train the policy \(\pi_i\) on the aggregated dataset \(D_i\).

2. **Data Collection:**  
   Execute \(\pi_i\) in the environment to collect states \(S_i\) and query the expert for corresponding actions:
   $$
   \{(s, \pi^*(s)) : s \in S_i\}
   $$

3. **Data Aggregation:**  
   Update the dataset as:
   $$
   D_{i+1} = D_i \cup \{(s, \pi^*(s)) : s \in S_i\}.
   $$

The overall training objective is to minimize:
$$
\mathcal{L}(\pi) = \mathbb{E}_{s \sim D} \left[\ell\big(\pi(s), \pi^*(s)\big)\right],
$$
where \(\ell\) is typically the mean squared error loss.

### Optimized DAgger with Beta Scheduling

In the optimized variant, a beta parameter \(\beta\) controls the probability of querying the expert:
- **High \(\beta\):** Increased reliance on expert guidance.
- **Low \(\beta\):** Greater reliance on the learned policy.

This beta scheduling reduces redundant expert queries while still ensuring sufficient corrective feedback.

### Reward-Weighted Regression

Reward-weighted regression refines policy learning by assigning weights to samples based on their rewards. The weighted loss is formulated as:
$$
\mathcal{L}_{\text{weighted}}(\pi) = \mathbb{E}_{s \sim D} \left[w(s) \, \ell\big(\pi(s), \pi^*(s)\big)\right],
$$
where \(w(s)\) is a weight proportional to the reward associated with state \(s\). This prioritizes high-quality trajectories during training.

## Experiments and Analysis

Performance is evaluated by comparing the number of expert queries against the success rate of the policy:
- **Baseline DAgger:** Standard data aggregation without beta scheduling.
- **Optimized DAgger:** Incorporates beta scheduling to reduce unnecessary queries.

Experimental results indicate that the optimized approach maintains or improves success rates while significantly reducing expert query counts.

## Conclusion

This work demonstrates an effective implementation of the DAgger algorithm enhanced with reward-weighted regression. By balancing expert input with autonomous learning, the approach improves policy performance in challenging goal-reaching tasks. Future work may explore further adaptive strategies to reduce expert dependency while maximizing learning efficiency.
