# Simplified MDVRP-ACO Algorithm Pseudocode

## 1. Initialization
```
function Initialize(data_file):
    Read problem data (customers, depots, capacities)
    Calculate distances between all nodes
    Initialize pheromone matrix with small values
    best_solution = null
    best_cost = infinity
```

## 2. Main ACO Loop
```
function Run():
    for iteration = 1 to max_iterations:
        solutions = []
        
        for ant = 1 to num_ants:
            solution, distance, vehicles = BuildSolution()
            objective_value = w1 * vehicles + w2 * distance
            solutions.Add(solution, objective_value)
            
            if objective_value < best_cost:
                best_solution = solution
                best_cost = objective_value
        
        UpdatePheromones(solutions)
    
    return best_solution
```

## 3. Solution Construction
```
function BuildSolution():
    unassigned_customers = all_customers
    solution = []
    vehicles_used = 0
    
    // Try to assign customers to each depot's vehicles
    for each depot:
        for each vehicle:
            // Decide whether to use this vehicle
            if should_use_vehicle():
                route = [depot]
                capacity = depot_capacity
                
                // Add customers to route while capacity allows
                while unassigned_customers not empty:
                    next = SelectNextCustomer(current, unassigned_customers, capacity)
                    if next is null:
                        break
                    
                    route.Add(next)
                    capacity -= demand[next]
                    unassigned_customers.Remove(next)
                
                route.Add(depot)  // Return to depot
                
                if route has customers:
                    solution.Add(route)
                    vehicles_used++
    
    // Handle any remaining customers with new routes
    while unassigned_customers not empty:
        // Create additional routes as needed
        
    total_distance = CalculateDistance(solution)
    
    return solution, total_distance, vehicles_used
```

## 4. Customer Selection
```
function SelectNextCustomer(current, candidates, remaining_capacity):
    // For each candidate customer
    for each customer in candidates:
        if demand[customer] > remaining_capacity:
            probability = 0
        else:
            // Calculate selection probability based on:
            // - Pheromone level (tau)
            // - Distance (eta = 1/distance)
            probability = (tau^alpha) * (eta^beta)
    
    // Select customer based on probabilities
    return selected_customer  // or null if none are feasible
```

## 5. Pheromone Update
```
function UpdatePheromones(solutions):
    // Evaporation
    for each edge (i,j):
        pheromone[i][j] *= (1 - evaporation_rate)
    
    // Deposit new pheromones
    for each solution:
        deposit = Q / solution_cost
        
        for each route in solution:
            for each edge (i,j) in route:
                pheromone[i][j] += deposit
```

## 6. Evaluate Solution
```
function CalculateDistance(solution):
    total = 0
    
    for each route in solution:
        for i = 0 to length(route) - 2:
            total += distance[route[i]][route[i+1]]
    
    return total
```
