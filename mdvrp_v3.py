import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


class MDVRP_ACO:
    """
    Ant Colony Optimization for Multi-Depot Vehicle Routing Problem (MDVRP).

    This class implements the ACO algorithm to solve the MDVRP, which aims to find optimal routes
    for a fleet of vehicles from multiple depots to serve a set of customers while minimizing
    total distance and number of vehicles used.
    """

    def __init__(self, data_file, num_ants=10, pheromone_weight=1.0, heuristic_weight=2.0,
                 evaporation_rate=0.1, pheromone_deposit=100, max_iterations=100,
                 vehicle_cost_weight=1.0, distance_cost_weight=1.0):
        """
        Initialize the MDVRP ACO algorithm.

        Parameters:
        - data_file: Path to the data file containing problem instance
        - num_ants: Number of ants in the colony
        - pheromone_weight (alpha): Weight of pheromone trails in decision making
        - heuristic_weight (beta): Weight of heuristic information in decision making
        - evaporation_rate (rho): Rate at which pheromone evaporates
        - pheromone_deposit (Q): Amount of pheromone deposited by ants
        - max_iterations: Maximum number of iterations for the algorithm
        - vehicle_cost_weight (w1): Weight for the number of vehicles used in objective function
        - distance_cost_weight (w2): Weight for the total distance in objective function
        """
        self.num_ants = num_ants
        self.pheromone_weight = pheromone_weight
        self.heuristic_weight = heuristic_weight
        self.evaporation_rate = evaporation_rate
        self.pheromone_deposit = pheromone_deposit
        self.max_iterations = max_iterations
        self.vehicle_cost_weight = vehicle_cost_weight
        self.distance_cost_weight = distance_cost_weight

        # Parse the data file
        self.parse_data_file(data_file)

        # Initialize pheromone trails between all nodes (customers and depots)
        self.num_nodes = self.num_customers + self.num_depots
        self.pheromone_trails = np.ones((self.num_nodes, self.num_nodes))

        # Calculate distances between all nodes
        self.calculate_distances()

        # Initialize heuristic information (1/distance)
        self.heuristic_info = 1.0 / (self.distances + np.eye(self.num_nodes))

        # Best solution found so far
        self.best_solution = None
        self.best_objective_value = float('inf')
        self.best_distance = float('inf')
        self.best_vehicles_used = float('inf')

        # Track solution history for analysis
        self.solution_history = []

    def parse_data_file(self, file_path):
        """
        Parse the MDVRP data file format.

        Reads data from the file and extracts:
        - Problem parameters
        - Depot capacities
        - Customer locations and demands
        - Depot locations
        """
        with open(file_path, 'r') as file:
            lines = file.readlines()

            # First line: problem parameters
            params = lines[0].strip().split()
            self.problem_type = int(params[0])
            self.vehicles_per_depot = int(params[1])
            self.num_customers = int(params[2])
            self.num_depots = int(params[3])

            # Depot capacity information
            self.depot_capacities = []
            for i in range(1, self.num_depots + 1):
                capacity = float(lines[i].strip().split()[1])
                self.depot_capacities.append(capacity)

            # Customer data
            self.customers = []
            for i in range(self.num_depots + 1, self.num_depots + self.num_customers + 1):
                data = lines[i].strip().split()
                customer_id = int(data[0])
                x_coord = float(data[1])
                y_coord = float(data[2])
                demand = float(data[4])
                self.customers.append({
                    'id': customer_id,
                    'x': x_coord,
                    'y': y_coord,
                    'demand': demand
                })

            # Depot coordinates
            self.depots = []
            for i in range(self.num_depots + self.num_customers + 1, self.num_depots + self.num_customers + self.num_depots + 1):
                data = lines[i].strip().split()
                depot_id = int(data[0])
                x_coord = float(data[1])
                y_coord = float(data[2])
                self.depots.append({
                    'id': depot_id,
                    'x': x_coord,
                    'y': y_coord
                })

    def calculate_distances(self):
        """
        Calculate the Euclidean distances between all nodes (customers and depots).

        Creates a distance matrix where distances[i][j] is the distance from node i to node j.
        """
        self.distances = np.zeros((self.num_nodes, self.num_nodes))

        # Extract coordinates for all nodes (customers and depots)
        coordinates = []

        # Customer coordinates (0 to num_customers-1)
        for customer in self.customers:
            coordinates.append((customer['x'], customer['y']))

        # Depot coordinates (num_customers to num_customers+num_depots-1)
        for depot in self.depots:
            coordinates.append((depot['x'], depot['y']))

        # Calculate distances
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i != j:
                    self.distances[i, j] = np.sqrt((coordinates[i][0] - coordinates[j][0])**2 +
                                                   (coordinates[i][1] - coordinates[j][1])**2)

    def run(self):
        """
        Run the ACO algorithm for the specified number of iterations.

        Returns:
        - best_solution: The best solution found (list of routes)
        - best_objective_value: The best objective function value
        - best_distance: The total distance of the best solution
        - best_vehicles_used: The number of vehicles used in the best solution
        """
        for iteration in range(self.max_iterations):
            # Solutions for all ants in this iteration
            all_solutions = []
            all_distances = []
            all_vehicle_counts = []
            all_objective_values = []

            # Each ant builds a solution
            for ant in range(self.num_ants):
                solution, distance, vehicles_used = self.construct_solution()
                all_solutions.append(solution)
                all_distances.append(distance)
                all_vehicle_counts.append(vehicles_used)

                # Calculate objective function value using the bi-objective formula
                objective_value = (self.vehicle_cost_weight * vehicles_used +
                                   self.distance_cost_weight * distance)
                all_objective_values.append(objective_value)

            # Find best solution in this iteration
            best_idx = np.argmin(all_objective_values)
            iteration_best_solution = all_solutions[best_idx]
            iteration_best_objective = all_objective_values[best_idx]
            iteration_best_distance = all_distances[best_idx]
            iteration_best_vehicles = all_vehicle_counts[best_idx]

            # Update best solution if necessary
            if iteration_best_objective < self.best_objective_value:
                self.best_solution = iteration_best_solution
                self.best_objective_value = iteration_best_objective
                self.best_distance = iteration_best_distance
                self.best_vehicles_used = iteration_best_vehicles
                print(f"Iteration {iteration+1}: New best solution found:")
                print(f"  - Objective value: {self.best_objective_value:.2f}")
                print(f"  - Total distance: {self.best_distance:.2f}")
                print(f"  - Vehicles used: {self.best_vehicles_used}")

            # Save this iteration's best solution for history
            self.solution_history.append({
                'iteration': iteration + 1,
                'objective': iteration_best_objective,
                'distance': iteration_best_distance,
                'vehicles': iteration_best_vehicles
            })

            # Update pheromone trails
            self.update_pheromones(all_solutions, all_objective_values)

            # Optionally, print current best solution
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration+1}: Best objective so far = {self.best_objective_value:.2f}")

        print(f"\nFinal best solution:")
        print(f"  - Objective value: {self.best_objective_value:.2f}")
        print(f"  - Total distance: {self.best_distance:.2f}")
        print(f"  - Vehicles used: {self.best_vehicles_used}")

        return (self.best_solution, self.best_objective_value,
                self.best_distance, self.best_vehicles_used)

    def construct_solution(self):
        """
        Construct a solution for one ant.

        A solution consists of a set of routes, where each route starts and ends at a depot.

        Returns:
        - solution: A list of routes, where each route is a list of node indices
        - total_distance: The total distance of the solution
        - vehicles_used: The number of vehicles used in the solution
        """
        # Initialize solution
        solution = []
        vehicles_used = 0

        # Customer to depot assignment matrix (z_id)
        customer_depot_assignment = np.zeros((self.num_customers, self.num_depots))

        # Vehicle usage indicator (y_k)
        vehicle_usage = np.zeros(self.num_depots * self.vehicles_per_depot)

        # Unassigned customers
        unassigned_customers = list(range(self.num_customers))

        # Each depot has multiple vehicles
        for depot_idx in range(self.num_depots):
            depot_node_idx = self.num_customers + depot_idx

            # For each vehicle in the depot
            for vehicle_idx in range(self.vehicles_per_depot):
                global_vehicle_idx = depot_idx * self.vehicles_per_depot + vehicle_idx

                # Skip if no more customers to assign
                if not unassigned_customers:
                    break

                # Decide whether to use this vehicle (probabilistic decision)
                # Higher probability of using a vehicle if there are more customers left
                use_vehicle_prob = min(1.0, len(unassigned_customers) / (self.num_customers * 0.5))
                if random.random() > use_vehicle_prob:
                    continue

                # Start a new route from the depot
                route = [depot_node_idx]
                remaining_capacity = self.depot_capacities[depot_idx]
                current_node = depot_node_idx
                route_has_customers = False

                while unassigned_customers:
                    # Calculate probabilities for next node
                    selection_probabilities = self.calculate_selection_probabilities(
                        current_node, unassigned_customers, remaining_capacity)

                    # If no valid next nodes, break
                    if not any(selection_probabilities):
                        break

                    # Select next node based on probabilities
                    next_node = random.choices(
                        unassigned_customers, weights=selection_probabilities, k=1)[0]

                    # Update route and remaining capacity
                    route.append(next_node)
                    remaining_capacity -= self.customers[next_node]['demand']
                    unassigned_customers.remove(next_node)
                    current_node = next_node
                    route_has_customers = True

                    # Update customer-depot assignment
                    customer_depot_assignment[next_node, depot_idx] = 1

                    # If vehicle is full, return to depot
                    if remaining_capacity <= 0:
                        break

                # Complete the route by returning to the depot
                route.append(depot_node_idx)

                # Add route to solution if it's not just depot-depot
                if route_has_customers:
                    solution.append(route)
                    vehicle_usage[global_vehicle_idx] = 1
                    vehicles_used += 1

        # If there are still unassigned customers, create additional routes
        while unassigned_customers:
            # Choose a depot randomly
            depot_idx = random.randint(0, self.num_depots - 1)
            depot_node_idx = self.num_customers + depot_idx

            # Start a new route from the depot
            route = [depot_node_idx]
            remaining_capacity = self.depot_capacities[depot_idx]
            current_node = depot_node_idx
            route_has_customers = False

            # Find next available vehicle index
            global_vehicle_idx = depot_idx * self.vehicles_per_depot
            while (global_vehicle_idx < (depot_idx + 1) * self.vehicles_per_depot and
                   vehicle_usage[global_vehicle_idx] == 1):
                global_vehicle_idx += 1

            # If no available vehicles, try another depot
            if global_vehicle_idx >= (depot_idx + 1) * self.vehicles_per_depot:
                continue

            while unassigned_customers:
                # Calculate probabilities for next node
                selection_probabilities = self.calculate_selection_probabilities(
                    current_node, unassigned_customers, remaining_capacity)

                # If no valid next nodes, break
                if not any(selection_probabilities):
                    break

                # Select next node based on probabilities
                next_node = random.choices(
                    unassigned_customers, weights=selection_probabilities, k=1)[0]

                # Update route and remaining capacity
                route.append(next_node)
                remaining_capacity -= self.customers[next_node]['demand']
                unassigned_customers.remove(next_node)
                current_node = next_node
                route_has_customers = True

                # Update customer-depot assignment
                customer_depot_assignment[next_node, depot_idx] = 1

                # If vehicle is full, return to depot
                if remaining_capacity <= 0:
                    break

            # Complete the route by returning to the depot
            route.append(depot_node_idx)

            # Add route to solution if it's not just depot-depot
            if route_has_customers:
                solution.append(route)
                vehicle_usage[global_vehicle_idx] = 1
                vehicles_used += 1

        # Calculate total distance of the solution
        total_distance = self.calculate_solution_distance(solution)

        return solution, total_distance, vehicles_used

    def calculate_selection_probabilities(self, current_node, candidate_nodes, remaining_capacity):
        """
        Calculate selection probabilities for the next node based on pheromone and heuristic information.

        Parameters:
        - current_node: The current node index
        - candidate_nodes: List of candidate node indices
        - remaining_capacity: Remaining vehicle capacity

        Returns:
        - probabilities: List of selection probabilities for each candidate node
        """
        probabilities = []

        for node in candidate_nodes:
            # Skip if demand exceeds remaining capacity
            if self.customers[node]['demand'] > remaining_capacity:
                probabilities.append(0)
            else:
                # Calculate probability based on pheromone and heuristic
                pheromone_value = self.pheromone_trails[current_node, node]
                heuristic_value = self.heuristic_info[current_node, node]
                probability = (pheromone_value ** self.pheromone_weight) * (heuristic_value ** self.heuristic_weight)
                probabilities.append(probability)

        # Normalize probabilities
        total = sum(probabilities)
        if total > 0:
            probabilities = [p / total for p in probabilities]

        return probabilities

    def calculate_solution_distance(self, solution):
        """
        Calculate the total distance of a solution.

        Parameters:
        - solution: A list of routes, where each route is a list of node indices

        Returns:
        - total_distance: The total distance of the solution
        """
        total_distance = 0

        for route in solution:
            route_distance = 0
            for i in range(len(route) - 1):
                route_distance += self.distances[route[i], route[i+1]]
            total_distance += route_distance

        return total_distance

    def update_pheromones(self, solutions, objective_values):
        """
        Update pheromone trails based on the quality of solutions.

        Parameters:
        - solutions: List of solutions
        - objective_values: List of objective function values
        """
        # Evaporation
        self.pheromone_trails *= (1 - self.evaporation_rate)

        # Deposit new pheromones
        for solution, obj_value in zip(solutions, objective_values):
            if obj_value > 0:  # Avoid division by zero
                deposit = self.pheromone_deposit / obj_value

                for route in solution:
                    for i in range(len(route) - 1):
                        self.pheromone_trails[route[i], route[i+1]] += deposit
                        self.pheromone_trails[route[i+1], route[i]] += deposit  # Symmetric

    def extract_decision_variables(self, solution):
        """
        Extract the decision variables from the solution:
        - x_ijk: 1 if vehicle k traverses arc (i,j), 0 otherwise
        - y_k: 1 if vehicle k is used, 0 otherwise
        - z_id: 1 if customer i is assigned to depot d, 0 otherwise

        Parameters:
        - solution: A list of routes, where each route is a list of node indices

        Returns:
        - arc_usage: 3D array of arc usage (x_ijk)
        - vehicle_usage: 1D array of vehicle usage (y_k)
        - customer_depot_assignment: 2D array of customer-depot assignments (z_id)
        """
        # Initialize decision variables
        arc_usage = np.zeros((self.num_nodes, self.num_nodes, len(solution)))
        vehicle_usage = np.zeros(len(solution))
        customer_depot_assignment = np.zeros((self.num_customers, self.num_depots))

        # Fill in the decision variables
        for k, route in enumerate(solution):
            # This vehicle is used
            vehicle_usage[k] = 1

            # Get the depot for this route
            depot_node = route[0]
            depot_idx = depot_node - self.num_customers

            # For each arc in the route
            for i in range(len(route) - 1):
                from_node = route[i]
                to_node = route[i+1]

                # Set arc_usage = 1 for this arc and vehicle
                arc_usage[from_node, to_node, k] = 1

                # If the from_node is a customer, assign it to the depot
                if from_node < self.num_customers:
                    customer_depot_assignment[from_node, depot_idx] = 1

        return arc_usage, vehicle_usage, customer_depot_assignment

    def calculate_objective_function(self, solution):
        """
        Calculate the objective function value using the bi-objective formula.
        Parameters:
        - solution: A list of routes, where each route is a list of node indices

        Returns:
        - objective_value: The objective function value
        - vehicles_used: The number of vehicles used
        - total_distance: The total distance of the solution
        """
        # Extract decision variables
        arc_usage, vehicle_usage, customer_depot_assignment = self.extract_decision_variables(solution)

        # Calculate number of vehicles used
        vehicles_used = np.sum(vehicle_usage)

        # Calculate total distance
        total_distance = 0
        for k in range(len(solution)):
            for i in range(self.num_nodes):
                for j in range(self.num_nodes):
                    total_distance += self.distances[i, j] * arc_usage[i, j, k]

        # Calculate objective function
        objective_value = (self.vehicle_cost_weight * vehicles_used +
                           self.distance_cost_weight * total_distance)

        return objective_value, vehicles_used, total_distance

    def visualize_solution(self, solution):
        """
        Visualize the solution using matplotlib.

        Parameters:
        - solution: A list of routes, where each route is a list of node indices
        """
        plt.figure(figsize=(12, 10))

        # Plot customers
        for customer in self.customers:
            plt.scatter(customer['x'], customer['y'], c='blue', s=30)
            plt.text(customer['x'] + 0.5, customer['y'] + 0.5, str(customer['id']), fontsize=8)

        # Plot depots
        for depot in self.depots:
            plt.scatter(depot['x'], depot['y'], c='red', marker='s', s=100)
            plt.text(depot['x'] + 0.5, depot['y'] + 0.5, str(depot['id']), fontsize=10)

        # Plot routes with different colors
        colors = plt.cm.jet(np.linspace(0, 1, len(solution)))

        for i, route in enumerate(solution):
            route_x = []
            route_y = []

            for node in route:
                if node < self.num_customers:
                    # Customer node
                    route_x.append(self.customers[node]['x'])
                    route_y.append(self.customers[node]['y'])
                else:
                    # Depot node
                    depot_idx = node - self.num_customers
                    route_x.append(self.depots[depot_idx]['x'])
                    route_y.append(self.depots[depot_idx]['y'])

            plt.plot(route_x, route_y, c=colors[i], linewidth=1.5)

        # Extract decision variables
        arc_usage, vehicle_usage, customer_depot_assignment = self.extract_decision_variables(solution)

        # Calculate statistics
        total_vehicles = np.sum(vehicle_usage)
        total_distance = self.calculate_solution_distance(solution)
        objective_value = (self.vehicle_cost_weight * total_vehicles +
                           self.distance_cost_weight * total_distance)

        plt.title(f'MDVRP Solution\nObjective: {objective_value:.2f} '
                  f'(vehicle_weight={self.vehicle_cost_weight}, distance_weight={self.distance_cost_weight})\n'
                  f'Vehicles: {total_vehicles}/{self.vehicles_per_depot * self.num_depots}, Distance: {total_distance:.2f}')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.savefig('mdvrp_solution.png', dpi=300)
        plt.close()

        # Plot convergence
        self.plot_convergence()

    def plot_convergence(self):
        """
        Plot the convergence of the algorithm.
        """
        iterations = [sol['iteration'] for sol in self.solution_history]
        objectives = [sol['objective'] for sol in self.solution_history]
        distances = [sol['distance'] for sol in self.solution_history]
        vehicles = [sol['vehicles'] for sol in self.solution_history]

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

        # Plot objective function
        ax1.plot(iterations, objectives, 'b-', linewidth=2)
        ax1.set_ylabel('Objective Value')
        ax1.set_title('Convergence of Objective Function')
        ax1.grid(True)

        # Plot total distance
        ax2.plot(iterations, distances, 'g-', linewidth=2)
        ax2.set_ylabel('Total Distance')
        ax2.set_title('Convergence of Total Distance')
        ax2.grid(True)

        # Plot vehicles used
        ax3.plot(iterations, vehicles, 'r-', linewidth=2)
        ax3.set_ylabel('Vehicles Used')
        ax3.set_xlabel('Iteration')
        ax3.set_title('Convergence of Vehicles Used')
        ax3.grid(True)

        plt.tight_layout()
        plt.savefig('convergence.png', dpi=300)
        plt.close()


# Example usage
if __name__ == "__main__":
    # ACO parameters
    num_ants = 20
    pheromone_weight = 1.0      # Alpha: importance of pheromone trails
    heuristic_weight = 2.0      # Beta: importance of heuristic information
    evaporation_rate = 0.1      # Rho: pheromone evaporation rate
    pheromone_deposit = 100     # Q: pheromone deposit factor
    max_iterations = 100
    vehicle_cost_weight = 10.0  # Weight for number of vehicles (higher means more important)
    distance_cost_weight = 1.0  # Weight for total distance

    # Create and run the algorithm
    aco = MDVRP_ACO(
        data_file="test_data/p01.txt",
        num_ants=num_ants,
        pheromone_weight=pheromone_weight,
        heuristic_weight=heuristic_weight,
        evaporation_rate=evaporation_rate,
        pheromone_deposit=pheromone_deposit,
        max_iterations=max_iterations,
        vehicle_cost_weight=vehicle_cost_weight,
        distance_cost_weight=distance_cost_weight
    )

    solution, obj_value, distance, vehicles = aco.run()

    # Visualize the solution
    aco.visualize_solution(solution)

    # Print solution details
    print("\nDetailed Solution:")
    for i, route in enumerate(solution):
        route_str = []
        total_load = 0
        for node in route:
            if node < aco.num_customers:
                customer_id = aco.customers[node]['id']
                demand = aco.customers[node]['demand']
                total_load += demand
                route_str.append(f"C{customer_id}(d={demand})")
            else:
                depot_idx = node - aco.num_customers
                route_str.append(f"D{aco.depots[depot_idx]['id']}")

        route_distance = 0
        for j in range(len(route) - 1):
            route_distance += aco.distances[route[j], route[j+1]]

        print(f"Route {i+1} (Vehicle {i+1}):")
        print(f"  Path: {' -> '.join(route_str)}")
        print(f"  Total load: {total_load}")
        print(f"  Distance: {route_distance:.2f}")

    # Extract decision variables for verification
    arc_usage, vehicle_usage, customer_depot_assignment = aco.extract_decision_variables(solution)

    print("\nDecision Variables Summary:")
    print(f"vehicle_usage (y_k): {sum(vehicle_usage)} vehicles used out of {len(vehicle_usage)} possible")

    # Count customer-depot assignments
    for d in range(aco.num_depots):
        customers_assigned = sum(customer_depot_assignment[:, d])
        print(f"customer_depot_assignment (z_id): {customers_assigned} customers assigned to depot {aco.depots[d]['id']}")