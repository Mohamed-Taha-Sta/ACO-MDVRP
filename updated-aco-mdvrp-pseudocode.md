# MDVRP-ACO Algorithm Pseudocode (Updated Version)

## 1. Initialization
```
function Initialize(data_file, parameters):
    Parse problem data (customers, depots, demands, capacities)
    Calculate distance matrix between all nodes (customers and depots)
    Initialize pheromone matrix with small uniform values
    Initialize heuristic information (1/distance)
    Set best_solution = null
    Set best_objective_value = infinity
    Set best_distance = infinity
    Set best_vehicles_used = infinity
```
**Explanation:** Setup phase that reads the problem data, calculates distances, creates the initial pheromone matrix, and prepares tracking variables for both single and multi-objective optimization.

## 2. Main ACO Loop
```
function Run():
    for iteration = 1 to max_iterations:
        all_solutions = []
        all_distances = []
        all_vehicle_counts = []
        all_objective_values = []
        
        for ant = 1 to num_ants:
            solution, distance, vehicles_used = ConstructSolution()
            
            objective_value = vehicle_cost_weight * vehicles_used + 
                              distance_cost_weight * distance
            
            all_solutions.append(solution)
            all_distances.append(distance)
            all_vehicle_counts.append(vehicles_used)
            all_objective_values.append(objective_value)
        
        // Find best solution in this iteration
        best_idx = argmin(all_objective_values)
        iteration_best_solution = all_solutions[best_idx]
        iteration_best_objective = all_objective_values[best_idx]
        iteration_best_distance = all_distances[best_idx]
        iteration_best_vehicles = all_vehicle_counts[best_idx]
        
        // Update best solution if necessary
        if iteration_best_objective < best_objective_value:
            best_solution = iteration_best_solution
            best_objective_value = iteration_best_objective
            best_distance = iteration_best_distance
            best_vehicles_used = iteration_best_vehicles
        
        // Save this iteration's solution for history
        solution_history.append(iteration data)
        
        // Update pheromone trails
        UpdatePheromones(all_solutions, all_objective_values)
    
    return best_solution, best_objective_value, best_distance, best_vehicles_used
```
**Explanation:** The core iteration process where each ant builds a solution, solutions are evaluated based on both distance and vehicle count, and pheromone trails are updated. It now includes a bi-objective function that balances vehicle usage and distance traveled.

## 3. Solution Construction
```
function ConstructSolution():
    solution = []
    vehicles_used = 0
    
    // Initialize customer-depot assignment tracking
    customer_depot_assignment = zeros(num_customers, num_depots)
    
    // Initialize vehicle usage tracking
    vehicle_usage = zeros(num_depots * vehicles_per_depot)
    
    // Start with all customers unassigned
    unassigned_customers = list(0 to num_customers-1)
    
    // For each depot, assign customers and build routes
    for depot_idx = 0 to num_depots-1:
        depot_node_idx = num_customers + depot_idx
        
        // For each vehicle in the depot
        for vehicle_idx = 0 to vehicles_per_depot-1:
            global_vehicle_idx = depot_idx * vehicles_per_depot + vehicle_idx
            
            // Skip if no more customers to assign
            if unassigned_customers is empty:
                break
            
            // Probabilistic decision to use this vehicle
            use_vehicle_prob = min(1.0, len(unassigned_customers) / (num_customers * 0.5))
            if random() > use_vehicle_prob:
                continue
            
            // Start a new route from the depot
            route = [depot_node_idx]
            remaining_capacity = depot_capacities[depot_idx]
            current_node = depot_node_idx
            route_has_customers = false
            
            // Add customers to the route
            while unassigned_customers is not empty:
                // Calculate probabilities for next node
                selection_probabilities = CalculateSelectionProbabilities(
                    current_node, unassigned_customers, remaining_capacity)
                
                // If no valid next nodes, break
                if sum(selection_probabilities) == 0:
                    break
                
                // Select next node based on probabilities
                next_node = random_choice(unassigned_customers, weights=selection_probabilities)
                
                // Update route and capacity
                route.append(next_node)
                remaining_capacity -= customers[next_node]['demand']
                unassigned_customers.remove(next_node)
                current_node = next_node
                route_has_customers = true
                
                // Update customer-depot assignment
                customer_depot_assignment[next_node, depot_idx] = 1
                
                // If vehicle is full, return to depot
                if remaining_capacity <= 0:
                    break
            
            // Complete the route by returning to the depot
            route.append(depot_node_idx)
            
            // Add route to solution if it has customers
            if route_has_customers:
                solution.append(route)
                vehicle_usage[global_vehicle_idx] = 1
                vehicles_used += 1
    
    // Handle any remaining unassigned customers
    while unassigned_customers is not empty:
        // Choose a depot randomly
        depot_idx = random(0, num_depots-1)
        depot_node_idx = num_customers + depot_idx
        
        // Find next available vehicle
        global_vehicle_idx = depot_idx * vehicles_per_depot
        while global_vehicle_idx < (depot_idx + 1) * vehicles_per_depot and 
              vehicle_usage[global_vehicle_idx] == 1:
            global_vehicle_idx += 1
        
        // If no available vehicles, try another depot
        if global_vehicle_idx >= (depot_idx + 1) * vehicles_per_depot:
            continue
        
        // Start a new route from the depot
        route = [depot_node_idx]
        remaining_capacity = depot_capacities[depot_idx]
        current_node = depot_node_idx
        route_has_customers = false
        
        // Add customers to the route (same logic as above)
        // ...
        
        // Complete route and add to solution
        // ...
    
    // Calculate total distance of the solution
    total_distance = CalculateSolutionDistance(solution)
    
    return solution, total_distance, vehicles_used
```
**Explanation:** The solution construction process now focuses on assigning vehicles from multiple depots efficiently, with probabilistic decisions about vehicle usage and customer assignments based on capacity constraints.

## 4. Selection Probability Calculation
```
function CalculateSelectionProbabilities(current_node, candidate_nodes, remaining_capacity):
    probabilities = []
    
    for node in candidate_nodes:
        // Skip if demand exceeds remaining capacity
        if customers[node]['demand'] > remaining_capacity:
            probabilities.append(0)
        else:
            // Calculate probability based on pheromone and heuristic
            pheromone_value = pheromone_trails[current_node, node]
            heuristic_value = heuristic_info[current_node, node]
            probability = (pheromone_value ^ pheromone_weight) * 
                          (heuristic_value ^ heuristic_weight)
            probabilities.append(probability)
    
    // Normalize probabilities
    total = sum(probabilities)
    if total > 0:
        probabilities = [p / total for p in probabilities]
    
    return probabilities
```
**Explanation:** Calculates the probability of selecting each candidate node as the next stop on a route, based on pheromone levels, heuristic information (inverse distance), and capacity constraints.

## 5. Pheromone Update
```
function UpdatePheromones(solutions, objective_values):
    // Evaporation
    pheromone_trails *= (1 - evaporation_rate)
    
    // Deposit new pheromones
    for solution, obj_value in zip(solutions, objective_values):
        if obj_value > 0:  // Avoid division by zero
            deposit = pheromone_deposit / obj_value
            
            for route in solution:
                for i = 0 to length(route) - 2:
                    pheromone_trails[route[i], route[i+1]] += deposit
                    pheromone_trails[route[i+1], route[i]] += deposit  // Symmetric update
```
**Explanation:** Pheromone trails are updated by first evaporating existing pheromones and then depositing new pheromones based on solution quality. Better solutions (lower objective values) deposit more pheromone.

## 6. Solution Evaluation
```
function CalculateSolutionDistance(solution):
    total_distance = 0
    
    for route in solution:
        route_distance = 0
        for i = 0 to length(route) - 2:
            route_distance += distances[route[i], route[i+1]]
        total_distance += route_distance
    
    return total_distance

function CalculateObjectiveFunction(solution):
    // Extract decision variables
    arc_usage, vehicle_usage, customer_depot_assignment = ExtractDecisionVariables(solution)
    
    // Calculate number of vehicles used
    vehicles_used = sum(vehicle_usage)
    
    // Calculate total distance
    total_distance = 0
    for k = 0 to len(solution)-1:
        for i = 0 to num_nodes-1:
            for j = 0 to num_nodes-1:
                total_distance += distances[i, j] * arc_usage[i, j, k]
    
    // Calculate objective function with weights
    objective_value = (vehicle_cost_weight * vehicles_used +
                      distance_cost_weight * total_distance)
    
    return objective_value, vehicles_used, total_distance
```
**Explanation:** Evaluates solutions using a bi-objective function that considers both the total distance traveled and the number of vehicles used, with configurable weights for each component.

## 7. Extract Decision Variables
```
function ExtractDecisionVariables(solution):
    // Initialize decision variables
    arc_usage = zeros(num_nodes, num_nodes, len(solution))
    vehicle_usage = zeros(len(solution))
    customer_depot_assignment = zeros(num_customers, num_depots)
    
    // Fill in the decision variables
    for k, route in enumerate(solution):
        // This vehicle is used
        vehicle_usage[k] = 1
        
        // Get the depot for this route
        depot_node = route[0]
        depot_idx = depot_node - num_customers
        
        // For each arc in the route
        for i = 0 to length(route) - 2:
            from_node = route[i]
            to_node = route[i+1]
            
            // Set arc_usage = 1 for this arc and vehicle
            arc_usage[from_node, to_node, k] = 1
            
            // If the from_node is a customer, assign it to the depot
            if from_node < num_customers:
                customer_depot_assignment[from_node, depot_idx] = 1
    
    return arc_usage, vehicle_usage, customer_depot_assignment
```
**Explanation:** Extracts the mathematical optimization model's decision variables from the solution representation, which is useful for evaluating the solution's feasibility and calculating objective functions.

## 8. Visualization
```
function VisualizeSolution(solution):
    // Plot customers and depots
    for each customer:
        Plot customer location
        
    for each depot:
        Plot depot location
    
    // Plot routes with different colors
    colors = GenerateColors(len(solution))
    
    for i, route in enumerate(solution):
        route_x = []
        route_y = []
        
        for node in route:
            if node < num_customers:
                // Customer node
                x = customers[node]['x']
                y = customers[node]['y']
            else:
                // Depot node
                depot_idx = node - num_customers
                x = depots[depot_idx]['x']
                y = depots[depot_idx]['y']
            
            route_x.append(x)
            route_y.append(y)
        
        Plot route using colors[i]
    
    // Add title and save figure
    Calculate statistics and add to plot title
    Save figure to file
    
    // Plot convergence graphs
    Plot objective function over iterations
    Plot total distance over iterations
    Plot vehicles used over iterations
```
**Explanation:** Visualizes the solution with routes, customers, and depots, and generates convergence plots to show how the solution quality improves over iterations.
