# ACO Algorithm for MDVRP - Pseudocode

## Initialization
```
function Initialize(file_path, num_ants, alpha, beta, rho, q0, max_iterations):
    Parse MDVRP data file
    Calculate distance matrix between all nodes
    Calculate heuristic information (inverse of distances)
    Initialize pheromone matrix with uniform values
    best_solution ← null
    best_cost ← infinity
```

## Main ACO Loop
```
function Run():
    for iteration = 1 to max_iterations:
        solutions ← []
        
        # Each ant constructs a solution
        for ant = 1 to num_ants:
            solution, cost ← ConstructSolution(ant)
            solutions.Add((solution, cost))
            
            # Update best solution
            if cost < best_cost:
                best_solution ← solution
                best_cost ← cost
        
        # Update pheromones
        UpdatePheromones(solutions)
        
    return best_solution, best_cost
```

## Solution Construction
```
function ConstructSolution(ant_idx):
    # Initialize routes for each depot
    routes ← [[] for each depot]
    unassigned_customers ← list of all customers
    
    # Assign customers to depots
    for each customer in unassigned_customers:
        Calculate probabilities for each depot based on distance
        Select depot using probability rules
        Add customer to selected depot's list
    
    # Construct routes for each depot
    final_routes ← []
    total_cost ← 0
    
    for each depot and its assigned customers:
        depot_routes ← ConstructDepotRoutes(depot, customers, max_load)
        total_cost += CalculateRoutesCost(depot, depot_routes)
        
        for each route in depot_routes:
            final_route ← [depot] + route + [depot]
            final_routes.Add(final_route)
    
    return final_routes, total_cost
```

## Depot Route Construction
```
function ConstructDepotRoutes(depot, customers, max_load):
    routes ← []
    remaining_customers ← customers.copy()
    
    while remaining_customers is not empty:
        route ← []
        current_load ← 0
        
        # Select first customer
        Calculate probabilities for each customer based on pheromone and distance
        Select first_customer based on probability rules
        Add first_customer to route
        current_load += demand[first_customer]
        Remove first_customer from remaining_customers
        current_node ← first_customer
        
        # Build rest of route
        while current_node is not null:
            next_node ← SelectNextNode(current_node, remaining_customers, current_load, max_load)
            
            if next_node is not null:
                Add next_node to route
                current_load += demand[next_node]
                Remove next_node from remaining_customers
                current_node ← next_node
            else:
                current_node ← null
        
        if route is not empty:
            routes.Add(route)
    
    return routes
```

## Next Node Selection
```
function SelectNextNode(current_node, candidates, current_load, max_load):
    feasible_candidates ← []
    
    # Find candidates that don't exceed capacity
    for each candidate in candidates:
        if current_load + demand[candidate] <= max_load:
            feasible_candidates.Add(candidate)
    
    if feasible_candidates is empty:
        return null
    
    # Calculate probabilities based on pheromone and distance
    probs ← []
    for each candidate in feasible_candidates:
        tau ← pheromone[current_node][candidate]
        eta ← heuristic[current_node][candidate]
        prob ← (tau^alpha) * (eta^beta)
        probs.Add(prob)
    
    Normalize probabilities
    
    # Select next node
    if random() < q0:
        # Greedy selection (exploitation)
        return feasible_candidates[index of max value in probs]
    else:
        # Probabilistic selection (exploration)
        Select using roulette wheel selection based on probs
        return selected candidate
```

## Pheromone Update
```
function UpdatePheromones(solutions):
    # Evaporation
    pheromone ← (1 - rho) * pheromone
    
    # Deposit new pheromones
    for each (solution, cost) in solutions:
        delta_tau ← 1 / cost
        
        for each route in solution:
            for i = 0 to length(route) - 2:
                pheromone[route[i]][route[i+1]] += delta_tau
                pheromone[route[i+1]][route[i]] += delta_tau  # Symmetric
```
