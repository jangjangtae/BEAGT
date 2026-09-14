# BEAGT

**Overcoming Exploration Stagnation in Reinforcement Learning-Based Automated Game Testing**

This repository contains code used to study **exploration stagnation** in reinforcement-learning-based automated game testing.

## Overview

In automated game testing, an RL agent can quickly converge to familiar trajectories and lose exploration capacity. BEAGT addresses this by detecting stagnation from recent reward changes and temporarily increasing the exploration level of the policy.

The study evaluates this idea with DQN-based agents using **ε-greedy** and **softmax exploration**.

## Environments

### CartPole
Location-based fault scenarios are embedded into the environment and evaluated with repeated test episodes.

### MsPacman
The study also evaluates exploration behavior in MsPacman using the RELINE research environment.

Reference environment: https://github.com/RosaliaTufano/rlgameauthors

## Method

BEAGT monitors recent reward progress. When the reward change falls below a stagnation threshold, the exploration parameter is temporarily increased:

- **ε-greedy**: increase ε
- **Softmax exploration**: increase temperature

The goal is not simply to maximize task reward, but to recover exploration when the policy becomes overly concentrated on already-known trajectories.

## Repository Structure

```text
BEAGT/
├── CartPole/
│   ├── BEAGT_train.py
│   ├── RELINE_train.py
│   └── best_model/
├── pre-experiment/
│   └── env.py
└── readme.md
```

## Research Contribution

My contribution to this work included:

- constructing location-based fault scenarios for CartPole and MsPacman,
- implementing and evaluating dynamic exploration control with DQN,
- comparing BEAGT with existing and random exploration baselines,
- analyzing fault-discovery behavior after policy training.

## Publication

**Tae-Hyeon Jang**, Yeajin Lee, Hyunseok Kim,  
*Overcoming Exploration Stagnation in Reinforcement Learning-Based Automated Game Testing*,  
Journal of Digital Contents Society, 2025.  
**First Author**

## Notes

This repository is provided as research code for the published study. Environment dependencies and versions may require adjustment on newer systems.
