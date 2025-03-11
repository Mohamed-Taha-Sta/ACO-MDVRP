# MDVRP-ACO Solution Structure

## Core Solution Structure

In the MDVRP-ACO algorithm, a solution is represented as a nested list structure:

```python
solution = [
    [depot_id, customer1, customer5, customer3, depot_id],
    [depot_id, customer2, customer4, depot_id],
    [depot_id2, customer7, customer6, depot_id2],
    # more routes...
]
```

Where:
- Each inner list represents one complete vehicle route
- Each route starts and ends at the same depot
- The nodes between are customers visited in sequence
- All entries are indices corresponding to positions in the coordinate list

## Solution Construction Algorithm

The solution is built through these key steps:

1. **Customer-Depot Assignment**:
   - Each customer is initially assigned to a depot based on distance and pheromone levels
   - This creates customer clusters for each depot

2. **Route Construction**:
   - For each depot, the algorithm builds routes using assigned customers
   - Routes are created until all customers are served or vehicle limit is reached
   - Each route respects vehicle capacity constraints

3. **Node Selection Process**:
   - For each route position, select next customer using:
     - Pheromone trails (τ) - representing learned knowledge
     - Heuristic information (η) - usually inverse of distance
     - Combined as probability: p ∝ (τᵅ)(ηᵝ)
   - Uses q₀ parameter to balance exploitation vs. exploration

4. **Solution Evaluation**:
   - Total cost = sum of all route distances
   - Each route cost = sum of distances between consecutive nodes

5. **Solution Improvement**:
   - Multiple "ants" build independent solutions
   - Best solutions receive more pheromone reinforcement
   - Local search (2-opt) further improves promising routes

The algorithm gradually converges on optimal or near-optimal solutions through iterative learning from the collective ant colony experience.