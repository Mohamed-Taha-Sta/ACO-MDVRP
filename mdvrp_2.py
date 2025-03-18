import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


class MDVRP_ACO:
    def __init__(self, data_file, num_ants=10, alpha=1.0, beta=2.0, rho=0.1, Q=100, max_iterations=100, w1=1.0, w2=1.0):
        """
        Initialize the MDVRP ACO algorithm.

        Parameters:
        - data_file: path to the data file
        - num_ants: number of ants in the colony
        - alpha: importance of pheromone trails
        - beta: importance of heuristic information
        - rho: pheromone evaporation rate
        - Q: pheromone deposit factor
        - max_iterations: maximum number of iterations
        - w1: weight for the number of vehicles used
        - w2: weight for the total distance traveled
        """
        self.num_ants = num_ants
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.max_iterations = max_iterations
        self.w1 = w1  # Weight for number of vehicles
        self.w2 = w2  # Weight for total distance

        # Parse the data file
        self.parse_data_file(data_file)

        # Initialize pheromone trails between all nodes (customers and depots)
        self.n_nodes = self.n_customers + self.n_depots
        self.pheromone = np.ones((self.n_nodes, self.n_nodes))

        # Calculate distances between all nodes
        self.calculate_distances()

        # Initialize heuristic information (1/distance)
        self.heuristic = 1.0 / (self.distances + np.eye(self.n_nodes))

        # Best solution found so far
        self.best_solution = None
        self.best_cost = float('inf')
        self.best_vehicles_used = float('inf')

        # Track solution history for analysis
        self.solution_history = []

    def parse_data_file(self, file_path):
        """Parse the MDVRP data file."""
        with open(file_path, 'r') as file:
            lines = file.readlines()

            # First line: problem parameters
            params = lines[0].strip().split()
            self.problem_type = int(params[0])
            self.vehicles_per_depot = int(params[1])
            self.n_customers = int(params[2])
            self.n_depots = int(params[3])

            # Depot capacity information
            self.depot_capacities = []
            for i in range(1, self.n_depots + 1):
                capacity = float(lines[i].strip().split()[1])
                self.depot_capacities.append(capacity)

            # Customer data
            self.customers = []
            for i in range(self.n_depots + 1, self.n_depots + self.n_customers + 1):
                data = lines[i].strip().split()
                customer_id = int(data[0])
                x = float(data[1])
                y = float(data[2])
                demand = float(data[4])
                self.customers.append({
                    'id': customer_id,
                    'x': x,
                    'y': y,
                    'demand': demand
                })

            # Depot coordinates
            self.depots = []
            for i in range(self.n_depots + self.n_customers + 1, self.n_depots + self.n_customers + self.n_depots + 1):
                data = lines[i].strip().split()
                depot_id = int(data[0])
                x = float(data[1])
                y = float(data[2])
                self.depots.append({
                    'id': depot_id,
                    'x': x,
                    'y': y
                })

    def calculate_distances(self):
        """Calculate the Euclidean distances between all nodes."""
        self.distances = np.zeros((self.n_nodes, self.n_nodes))

        # Extract coordinates for all nodes (customers and depots)
        coordinates = []

        # Customer coordinates (0 to n_customers-1)
        for customer in self.customers:
            coordinates.append((customer['x'], customer['y']))

        # Depot coordinates (n_customers to n_customers+n_depots-1)
        for depot in self.depots:
            coordinates.append((depot['x'], depot['y']))

        # Calculate distances
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i != j:
                    self.distances[i, j] = np.sqrt((coordinates[i][0] - coordinates[j][0])**2 +
                                                   (coordinates[i][1] - coordinates[j][1])**2)

    def run(self):
        """Run the ACO algorithm."""
        for iteration in range(self.max_iterations):
            # Solutions for all ants in this iteration
            solutions = []
            costs = []
            vehicles_counts = []
            objective_values = []

            # Each ant builds a solution
            for ant in range(self.num_ants):
                solution, distance, vehicles_used = self.construct_solution()
                solutions.append(solution)
                costs.append(distance)
                vehicles_counts.append(vehicles_used)

                # Calculate objective function value using the bi-objective formula
                objective_value = self.w1 * vehicles_used + self.w2 * distance
                objective_values.append(objective_value)

            # Find best solution based on combined objective
            best_idx = np.argmin(objective_values)

            # Update best solution if necessary
            if objective_values[best_idx] < self.best_cost:
                self.best_solution = solutions[best_idx]
                self.best_cost = objective_values[best_idx]
                self.best_distance = costs[best_idx]
                self.best_vehicles_used = vehicles_counts[best_idx]
                print(f"Iteration {iteration+1}: New best solution found:")
                print(f"  - Objective value: {self.best_cost:.2f}")
                print(f"  - Total distance: {self.best_distance:.2f}")
                print(f"  - Vehicles used: {self.best_vehicles_used}")

            # Save this iteration's best solution for history
            self.solution_history.append({
                'iteration': iteration + 1,
                'objective': objective_values[best_idx],
                'distance': costs[best_idx],
                'vehicles': vehicles_counts[best_idx]
            })

            # Update pheromone trails
            self.update_pheromones(solutions, objective_values)

            # Optionally, print current best solution
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration+1}: Best objective so far = {self.best_cost:.2f}")

        print(f"\nFinal best solution:")
        print(f"  - Objective value: {self.best_cost:.2f}")
        print(f"  - Total distance: {self.best_distance:.2f}")
        print(f"  - Vehicles used: {self.best_vehicles_used}")

        return self.best_solution, self.best_cost, self.best_distance, self.best_vehicles_used

    def construct_solution(self):
        """Construct a solution for one ant."""
        # Initialize solution
        solution = []
        vehicles_used = 0

        # Customer to depot assignment matrix (z_id)
        customer_depot_assignment = np.zeros((self.n_customers, self.n_depots))

        # Vehicle usage indicator (y_k)
        vehicle_usage = np.zeros(self.n_depots * self.vehicles_per_depot)

        # Unassigned customers
        unassigned_customers = list(range(self.n_customers))

        # Each depot has multiple vehicles
        for depot_idx in range(self.n_depots):
            depot_node_idx = self.n_customers + depot_idx

            # For each vehicle in the depot
            for vehicle in range(self.vehicles_per_depot):
                vehicle_idx = depot_idx * self.vehicles_per_depot + vehicle

                # Skip if no more customers to assign
                if not unassigned_customers:
                    break

                # Decide whether to use this vehicle (probabilistic decision)
                # Higher probability of using a vehicle if there are more customers left
                use_vehicle_prob = min(1.0, len(unassigned_customers) / (self.n_customers * 0.5))
                if random.random() > use_vehicle_prob:
                    continue

                # Start a new route from the depot
                route = [depot_node_idx]
                remaining_capacity = self.depot_capacities[depot_idx]
                current_node = depot_node_idx
                route_has_customers = False

                while unassigned_customers:
                    # Calculate probabilities for next node
                    probabilities = self.calculate_probabilities(current_node, unassigned_customers, remaining_capacity)

                    # If no valid next nodes, break
                    if not any(probabilities):
                        break

                    # Select next node based on probabilities
                    next_node = random.choices(unassigned_customers, weights=probabilities, k=1)[0]

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
                    vehicle_usage[vehicle_idx] = 1
                    vehicles_used += 1

        # If there are still unassigned customers, create additional routes
        while unassigned_customers:
            # Choose a depot randomly
            depot_idx = random.randint(0, self.n_depots - 1)
            depot_node_idx = self.n_customers + depot_idx

            # Start a new route from the depot
            route = [depot_node_idx]
            remaining_capacity = self.depot_capacities[depot_idx]
            current_node = depot_node_idx
            route_has_customers = False

            # Find next available vehicle index
            vehicle_idx = depot_idx * self.vehicles_per_depot
            while vehicle_idx < (depot_idx + 1) * self.vehicles_per_depot and vehicle_usage[vehicle_idx] == 1:
                vehicle_idx += 1

            # If no available vehicles, try another depot
            if vehicle_idx >= (depot_idx + 1) * self.vehicles_per_depot:
                continue

            while unassigned_customers:
                # Calculate probabilities for next node
                probabilities = self.calculate_probabilities(current_node, unassigned_customers, remaining_capacity)

                # If no valid next nodes, break
                if not any(probabilities):
                    break

                # Select next node based on probabilities
                next_node = random.choices(unassigned_customers, weights=probabilities, k=1)[0]

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
                vehicle_usage[vehicle_idx] = 1
                vehicles_used += 1

        # Calculate total distance of the solution
        total_distance = self.calculate_solution_distance(solution)

        return solution, total_distance, vehicles_used

    def calculate_probabilities(self, current_node, candidate_nodes, remaining_capacity):
        """Calculate probabilities for the next node."""
        probabilities = []

        for node in candidate_nodes:
            # Skip if demand exceeds remaining capacity
            if self.customers[node]['demand'] > remaining_capacity:
                probabilities.append(0)
            else:
                # Calculate probability based on pheromone and heuristic
                pheromone = self.pheromone[current_node, node]
                heuristic = self.heuristic[current_node, node]
                probability = (pheromone ** self.alpha) * (heuristic ** self.beta)
                probabilities.append(probability)

        # Normalize probabilities
        total = sum(probabilities)
        if total > 0:
            probabilities = [p / total for p in probabilities]

        return probabilities

    def calculate_solution_distance(self, solution):
        """Calculate the total distance of a solution."""
        total_distance = 0

        for route in solution:
            route_distance = 0
            for i in range(len(route) - 1):
                route_distance += self.distances[route[i], route[i+1]]
            total_distance += route_distance

        return total_distance

    def update_pheromones(self, solutions, objective_values):
        """Update pheromone trails."""
        # Evaporation
        self.pheromone *= (1 - self.rho)

        # Deposit new pheromones
        for solution, obj_value in zip(solutions, objective_values):
            if obj_value > 0:  # Avoid division by zero
                deposit = self.Q / obj_value

                for route in solution:
                    for i in range(len(route) - 1):
                        self.pheromone[route[i], route[i+1]] += deposit
                        self.pheromone[route[i+1], route[i]] += deposit  # Symmetric

    def extract_decision_variables(self, solution):
        """
        Extract the decision variables from the solution:
        - x_ij^k: 1 if vehicle k traverses arc (i,j), 0 otherwise
        - y_k: 1 if vehicle k is used, 0 otherwise
        - z_id: 1 if customer i is assigned to depot d, 0 otherwise
        """
        # Initialize decision variables
        # We'll use a simplified index for vehicles: consecutive numbers
        x_ijk = np.zeros((self.n_nodes, self.n_nodes, len(solution)))
        y_k = np.zeros(len(solution))
        z_id = np.zeros((self.n_customers, self.n_depots))
        # Fill in the decision variables
        for k, route in enumerate(solution):
            # This vehicle is used
            y_k[k] = 1

            # Get the depot for this route
            depot_node = route[0]
            depot_idx = depot_node - self.n_customers

            # For each arc in the route
            for i in range(len(route) - 1):
                from_node = route[i]
                to_node = route[i+1]

                # Set x_ij^k = 1 for this arc and vehicle
                x_ijk[from_node, to_node, k] = 1

                # If the from_node is a customer, assign it to the depot
                if from_node < self.n_customers:
                    z_id[from_node, depot_idx] = 1

        return x_ijk, y_k, z_id

    def calculate_objective_function(self, solution):
        """Calculate the objective function value using the bi-objective formula."""
        # Extract decision variables
        x_ijk, y_k, z_id = self.extract_decision_variables(solution)

        # Calculate number of vehicles used
        vehicles_used = np.sum(y_k)

        # Calculate total distance
        total_distance = 0
        for k in range(len(solution)):
            for i in range(self.n_nodes):
                for j in range(self.n_nodes):
                    total_distance += self.distances[i, j] * x_ijk[i, j, k]

        # Calculate objective function
        objective_value = self.w1 * vehicles_used + self.w2 * total_distance

        return objective_value, vehicles_used, total_distance

    def visualize_solution(self, solution):
        """Visualize the solution using matplotlib."""
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
                if node < self.n_customers:
                    # Customer node
                    route_x.append(self.customers[node]['x'])
                    route_y.append(self.customers[node]['y'])
                else:
                    # Depot node
                    depot_idx = node - self.n_customers
                    route_x.append(self.depots[depot_idx]['x'])
                    route_y.append(self.depots[depot_idx]['y'])

            plt.plot(route_x, route_y, c=colors[i], linewidth=1.5)

        # Extract decision variables
        x_ijk, y_k, z_id = self.extract_decision_variables(solution)

        # Calculate statistics
        total_vehicles = np.sum(y_k)
        total_distance = self.calculate_solution_distance(solution)
        objective_value = self.w1 * total_vehicles + self.w2 * total_distance

        plt.title(f'MDVRP Solution\nObjective: {objective_value:.2f} (w1={self.w1}, w2={self.w2})\n'
                  f'Vehicles: {total_vehicles}, Distance: {total_distance:.2f}')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.savefig('mdvrp_solution.png', dpi=300)
        plt.close()

        # Plot convergence
        self.plot_convergence()

    def plot_convergence(self):
        """Plot the convergence of the algorithm."""
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
    alpha = 1.0      # Importance of pheromone
    beta = 2.0       # Importance of heuristic information
    rho = 0.1        # Pheromone evaporation rate
    Q = 100          # Pheromone deposit factor
    max_iterations = 100
    w1 = 10.0        # Weight for number of vehicles (adjust this based on importance)
    w2 = 1.0         # Weight for total distance

    # Create and run the algorithm
    aco = MDVRP_ACO(
        data_file="test_data/p01.txt",
        num_ants=num_ants,
        alpha=alpha,
        beta=beta,
        rho=rho,
        Q=Q,
        max_iterations=max_iterations,
        w1=w1,
        w2=w2
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
            if node < aco.n_customers:
                customer_id = aco.customers[node]['id']
                demand = aco.customers[node]['demand']
                total_load += demand
                route_str.append(f"C{customer_id}(d={demand})")
            else:
                depot_idx = node - aco.n_customers
                route_str.append(f"D{aco.depots[depot_idx]['id']}")

        route_distance = 0
        for j in range(len(route) - 1):
            route_distance += aco.distances[route[j], route[j+1]]

        print(f"Route {i+1} (Vehicle {i+1}):")
        print(f"  Path: {' -> '.join(route_str)}")
        print(f"  Total load: {total_load}")
        print(f"  Distance: {route_distance:.2f}")

    # Extract decision variables for verification
    x_ijk, y_k, z_id = aco.extract_decision_variables(solution)

    print("\nDecision Variables Summary:")
    print(f"y_k (vehicle usage): {sum(y_k)} vehicles used out of {len(y_k)} possible")

    # Count customer-depot assignments
    for d in range(aco.n_depots):
        customers_assigned = sum(z_id[:, d])
        print(f"z_id: {customers_assigned} customers assigned to depot {aco.depots[d]['id']}")