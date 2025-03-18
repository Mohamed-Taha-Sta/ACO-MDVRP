# Multi-Depot Vehicle Routing Problem Solver

This repository contains an implementation of an Ant Colony Optimization (ACO) algorithm for solving the Multi-Depot Vehicle Routing Problem (MDVRP).

## Problem Description

The MDVRP is an extension of the classic Vehicle Routing Problem (VRP) where multiple depots are available to serve customers. The goal is to:

- Assign customers to depots
- Determine the optimal routes for vehicles based at each depot
- Minimize the total travel distance and number of vehicles used
- Respect vehicle capacity constraints

## Algorithm

The implementation uses Ant Colony Optimization (ACO), a metaheuristic inspired by the foraging behavior of ants:

- Ants construct solutions by probabilistically selecting nodes based on pheromone trails and heuristic information
- Pheromone trails are updated based on solution quality
- The algorithm balances exploration of new solutions with exploitation of known good solutions

## Features

- Customer-to-depot assignment based on pheromone trails and distance
- Construction of routes using ACO principles
- Local search improvement using 2-opt heuristic
- Bi-objective optimization (minimizing both distance and vehicle count)
- Visualization of solutions with color-coded routes

## Usage

```python
# Create ACO solver
aco = MDVRP_ACO(
    file_path="test_data/p01.txt",
    num_ants=30,
    alpha=1.0,
    beta=5.0,
    rho=0.5,
    q0=0.9,
    max_iterations=100
)

# Run the algorithm
solution, cost = aco.run()

# Visualize the solution
aco.visualize_solution(solution)
```

## Parameters

- `alpha`: Weight of the pheromone information
- `beta`: Weight of the heuristic information
- `rho`: Pheromone evaporation rate
- `q0`: Probability of choosing the best option (exploitation vs. exploration)
- `num_ants`: Number of ants in the colony
- `max_iterations`: Maximum number of iterations

## Input Format

The solver accepts standard MDVRP benchmark problem instances. The format includes:
- Number of vehicles per depot, customers, and depots
- Vehicle capacities
- Customer coordinates and demands
- Depot coordinates

## Visualization

The solution visualizer displays:
- Customer locations (blue dots)
- Depot locations (red squares)
- Color-coded routes
- Summary statistics including total cost, vehicles used, and distance

## Dependencies

- NumPy: For numerical operations
- Matplotlib: For visualization
- Random: For probabilistic selections
- Copy: For solution manipulation