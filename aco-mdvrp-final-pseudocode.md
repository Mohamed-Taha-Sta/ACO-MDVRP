# MDVRP-ACO Algorithm Pseudocode

## 1. Initialization
```
function Initialize(file_path, parameters):
    Read problem data (customers, depots, capacities)
    Calculate distance matrix between all nodes
    Initialize pheromone matrix with small uniform values
    Set weights w₁, w₂ for the objective function
    best_solution = null
    best_cost = infinity
```
**Explanation:** Setup phase that reads the problem data, creates the initial pheromone matrix, sets the objective function weights, and prepares tracking variables.

## 2. Main ACO Loop
```
function Run():
    for iteration = 1 to max_iterations:
        solutions = []
        
        for ant = 1 to num_ants:
            solution, cost = ConstructSolution()
            Apply LocalSearch(solution, cost)
            solutions.Add((solution, cost))
            
            if cost < best_cost:
                best_solution = solution
                best_cost = cost
        
        UpdatePheromones(solutions)
    
    return best_solution, best_cost
```
**Explanation:** The core iteration process where each ant builds a solution, local improvements are applied, and the pheromone trails are updated based on solution quality.

## 3. Solution Construction
```
function ConstructSolution():
    // Assign customers to depots based on pheromone and distance
    depot_customers = AssignCustomersToDepots()
    
    solution = []
    total_cost = 0
    used_vehicles = 0
    
    // Build routes for each depot
    for each depot and its assigned customers:
        depot_routes = ConstructDepotRoutes(depot, customers, max_load, max_vehicles)
        
        // Add routes to solution
        for each route in depot_routes:
            vehicle_id = used_vehicles + 1
            final_route = [depot] + route + [depot]
            solution.Add((vehicle_id, final_route))
            total_cost += CalculateRouteCost(final_route, vehicle_id)
            used_vehicles += 1
    
    return solution, total_cost
```
**Explanation:** Solution construction happens in two stages - first assigning customers to depots, then building optimized routes for each depot's customers. Each route is assigned a vehicle ID for tracking costs.

## 4. Route Construction
```
function ConstructDepotRoutes(depot, customers, max_load, max_vehicles):
    routes = []
    remaining_customers = customers.copy()
    
    // Create routes until all customers served or vehicle limit reached
    while remaining_customers and routes.length < max_vehicles:
        route = []
        current_load = 0
        current_node = depot
        
        // Build route incrementally
        while remaining_customers:
            next_node = SelectNextNode(current_node, remaining_customers, current_load, max_load)
            
            if next_node is not null:
                route.Add(next_node)
                current_load += demand[next_node]
                remaining_customers.Remove(next_node)
                current_node = next_node
            else:
                break  // Return to depot
        
        if route not empty:
            routes.Add(route)
    
    return routes
```
**Explanation:** Routes are constructed one by one for each depot, adding customers incrementally until capacity constraints or vehicle limits are reached.

## 5. Next Node Selection
```
function SelectNextNode(current_node, candidates, current_load, max_load):
    // Find candidates that don't exceed capacity
    feasible_candidates = [c for c in candidates if current_load + demand[c] <= max_load]
    
    if feasible_candidates is empty:
        return null  // Return to depot
    
    // Calculate selection probabilities
    for each candidate in feasible_candidates:
        tau = pheromone[current_node][candidate]  // Pheromone strength
        eta = 1/distance[current_node][candidate]  // Heuristic (inverse distance)
        probability = (tau^alpha) * (eta^beta) / sumOfAllAttainableNodesProbabilities
    
    // Decision rule: exploitation vs exploration
    if random() < q0:  // Exploitation (greedy)
        return candidate with highest probability
    else:  // Exploration (probabilistic)
        return candidate selected according to probabilities
```
**Explanation:** Selects the next customer to visit based on pheromone levels and distances, with a balance between exploitation (choosing the best option) and exploration (trying alternatives).

## 6. Cost Calculation
```
function CalculateRouteCost(route, vehicle_id):
    // Initialize cost components
    fixed_vehicle_cost = w₁  // Cost of using a vehicle
    total_distance_cost = 0
    
    // Calculate distance cost for the route
    for i = 0 to length(route) - 2:
        from_node = route[i]
        to_node = route[i+1]
        total_distance_cost += w₂ * distance[from_node][to_node]
    
    // Total cost includes both components
    return fixed_vehicle_cost + total_distance_cost
```
**Explanation:** Calculates the cost of a route according to the objective function, which includes both a fixed cost for using a vehicle (w₁) and the weighted distance traveled (w₂).

## 7. Pheromone Update
```
function UpdatePheromones(solutions):
    // Evaporation
    for i, j in pheromone matrix:
        pheromone[i][j] = (1 - rho) * pheromone[i][j]
    
    // Deposit new pheromones
    for solution, cost in solutions:
        delta_tau = 1 / cost  // Better solutions deposit more
        
        for vehicle_id, route in solution:
            for i = 0 to length(route) - 2:
                pheromone[route[i]][route[i+1]] += delta_tau
```
**Explanation:** Pheromone trails are updated by first evaporating existing pheromones (forgetting) and then depositing new pheromones based on solution quality (learning).

## 8. Local Search
```
function LocalSearch(solution, cost):
    improved = true
    
    while improved:
        improved = false
        
        for vehicle_id, route in solution:
            for i, j in route (i < j):
                // Try 2-opt exchange (reverse segment between i and j)
                new_route = route[:i+1] + reverse(route[i+1:j]) + route[j:]
                
                // Calculate new route cost with the objective function
                new_route_cost = CalculateRouteCost(new_route, vehicle_id)
                old_route_cost = CalculateRouteCost(route, vehicle_id)
                
                if new_route_cost < old_route_cost:
                    solution[vehicle_id] = new_route
                    cost = cost - old_route_cost + new_route_cost
                    improved = true
    
    return solution, cost
```
**Explanation:** Improves solutions using 2-opt local search, which tries reversing route segments to find shorter paths. The improved cost calculation now uses the complete objective function.