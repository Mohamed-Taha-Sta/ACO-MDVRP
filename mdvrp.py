import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
import random
import copy


class MDVRP_ACO:
	def __init__(self, file_path, num_ants=30, alpha=1.0, beta=5.0, rho=0.5, q0=0.9, max_iterations=100):
		"""
		Initialize the ACO algorithm for MDVRP.

		Args:
			file_path (str): Path to the MDVRP data file
			num_ants (int): Number of ants in the colony
			alpha (float): Pheromone importance factor
			beta (float): Heuristic information importance factor
			rho (float): Pheromone evaporation rate
			q0 (float): Probability of choosing the best next node
			max_iterations (int): Maximum number of iterations
		"""
		self.file_path = file_path
		self.num_ants = num_ants
		self.alpha = alpha
		self.beta = beta
		self.rho = rho
		self.q0 = q0
		self.max_iterations = max_iterations

		# Parse data file
		self.parse_data_file()

		# Initialize pheromone matrix
		initial_pheromone = 1.0 / (self.num_customers * self.num_depots)
		self.pheromone = np.ones((self.num_nodes, self.num_nodes)) * initial_pheromone

		# Calculate distance matrix
		self.distance_matrix = np.zeros((self.num_nodes, self.num_nodes))
		for i in range(self.num_nodes):
			for j in range(self.num_nodes):
				self.distance_matrix[i][j] = euclidean(
					(self.coordinates[i][0], self.coordinates[i][1]),
					(self.coordinates[j][0], self.coordinates[j][1])
				)

		# Calculate heuristic information (inverse of distance)
		self.heuristic = 1.0 / (self.distance_matrix + np.finfo(float).eps)

		# Best solution so far
		self.best_solution = None
		self.best_cost = float('inf')

	def parse_data_file(self):
		"""Parse the MDVRP data file format as described."""
		with open(self.file_path, 'r') as f:
			lines = f.readlines()

		# Parse first line
		first_line = lines[0].strip().split()
		self.problem_type = int(first_line[0])
		self.num_vehicles = int(first_line[1])
		self.num_customers = int(first_line[2])
		self.num_depots = int(first_line[3])

		# Verify problem type
		if self.problem_type != 2:
			raise ValueError("This is not a MDVRP (type 2) problem.")

		# Parse depot information
		self.max_durations = []
		self.max_loads = []
		for i in range(1, self.num_depots + 1):
			depot_info = lines[i].strip().split()
			self.max_durations.append(float(depot_info[0]))
			self.max_loads.append(float(depot_info[1]))

		# Parse customer and depot information
		self.customer_ids = []
		self.coordinates = []
		self.service_durations = []
		self.demands = []
		self.depot_indices = []

		# Line index to start parsing customers
		line_idx = self.num_depots + 1

		# Parse customers
		for i in range(line_idx, line_idx + self.num_customers):
			customer_info = lines[i].strip().split()
			customer_id = int(customer_info[0])
			x = float(customer_info[1])
			y = float(customer_info[2])
			service_duration = float(customer_info[3])
			demand = float(customer_info[4])

			self.customer_ids.append(customer_id)
			self.coordinates.append((x, y))
			self.service_durations.append(service_duration)
			self.demands.append(demand)

		# Parse depots
		for i in range(line_idx + self.num_customers, line_idx + self.num_customers + self.num_depots):
			depot_info = lines[i].strip().split()
			depot_id = int(depot_info[0])
			x = float(depot_info[1])
			y = float(depot_info[2])

			self.depot_indices.append(len(self.coordinates))
			self.coordinates.append((x, y))
			# Depots have no demand or service duration
			self.service_durations.append(0)
			self.demands.append(0)

		# Total number of nodes (customers + depots)
		self.num_nodes = len(self.coordinates)

	def construct_solution(self, ant_idx):
		"""
		Construct a solution for a single ant.

		Args:
			ant_idx (int): Index of the ant

		Returns:
			list: Routes for each depot
			float: Total cost of the solution
		"""
		# Routes for each depot
		routes = [[] for _ in range(self.num_depots)]

		# Copy of customer list to assign
		unassigned_customers = list(range(self.num_customers))

		# Assign customers to depots
		for customer in unassigned_customers:
			# Calculate probabilities for each depot
			depot_probabilities = []
			sum_heuristic = 0

			for depot_idx, depot in enumerate(self.depot_indices):
				# Distance from depot to customer
				distance = self.distance_matrix[depot][customer]
				heuristic_value = 1.0 / (distance + 1e-10)
				sum_heuristic += heuristic_value
				depot_probabilities.append((depot_idx, heuristic_value))

			# Normalize probabilities
			for i in range(len(depot_probabilities)):
				depot_idx, heuristic = depot_probabilities[i]
				depot_probabilities[i] = (depot_idx, heuristic / sum_heuristic)

			# Select depot based on probabilities
			if random.random() < self.q0:
				# Greedy selection
				selected_depot = max(depot_probabilities, key=lambda x: x[1])[0]
			else:
				# Probabilistic selection
				cum_prob = 0
				rand_num = random.random()
				selected_depot = None

				for depot_idx, prob in depot_probabilities:
					cum_prob += prob
					if rand_num <= cum_prob:
						selected_depot = depot_idx
						break

				if selected_depot is None:
					selected_depot = depot_probabilities[-1][0]

			# Add customer to selected depot's unrouted list
			routes[selected_depot].append(customer)

		# Construct routes for each depot using ACO
		total_cost = 0
		final_routes = []

		for depot_idx, customers in enumerate(routes):
			if not customers:
				continue

			depot = self.depot_indices[depot_idx]
			max_load = self.max_loads[depot_idx]

			# Split customers into routes using ACO
			depot_routes = self.construct_depot_routes(depot, customers, max_load)
			total_cost += self.calculate_routes_cost(depot, depot_routes)

			# Add the depot's routes to the final solution
			for route in depot_routes:
				if route:  # Only add non-empty routes
					final_route = [depot] + route + [depot]
					final_routes.append(final_route)

		return final_routes, total_cost

	def construct_depot_routes(self, depot, customers, max_load):
		"""
		Construct routes for a single depot using ACO.

		Args:
			depot (int): Index of the depot
			customers (list): List of customer indices assigned to this depot
			max_load (float): Maximum vehicle load capacity

		Returns:
			list: List of routes (each route is a list of customer indices)
		"""
		routes = []
		remaining_customers = customers.copy()

		while remaining_customers:
			route = []
			current_load = 0
			current_node = None

			# Choose first customer for the route
			candidates = remaining_customers.copy()
			if candidates:
				# Calculate probabilities for each customer
				probs = []
				for candidate in candidates:
					# Pheromone and heuristic values
					tau = self.pheromone[depot][candidate]
					eta = self.heuristic[depot][candidate]
					prob = (tau ** self.alpha) * (eta ** self.beta)
					probs.append(prob)

				# Normalize probabilities
				sum_probs = sum(probs)
				if sum_probs > 0:
					probs = [p / sum_probs for p in probs]

				# Select first customer
				if random.random() < self.q0:
					# Greedy selection
					first_customer = candidates[np.argmax(probs)]
				else:
					# Probabilistic selection
					cum_prob = 0
					rand_num = random.random()
					first_customer = candidates[-1]  # Default

					for i, prob in enumerate(probs):
						cum_prob += prob
						if rand_num <= cum_prob:
							first_customer = candidates[i]
							break

				current_node = first_customer
				route.append(current_node)
				current_load += self.demands[current_node]
				remaining_customers.remove(current_node)

			# Build the rest of the route
			while current_node is not None:
				next_node = self.select_next_node(current_node, remaining_customers, current_load, max_load)

				if next_node is not None:
					route.append(next_node)
					current_load += self.demands[next_node]
					remaining_customers.remove(next_node)
					current_node = next_node
				else:
					current_node = None

			if route:
				routes.append(route)

		return routes

	def select_next_node(self, current_node, candidates, current_load, max_load):
		"""
		Select the next node to visit using ACO rules.

		Args:
			current_node (int): Current node index
			candidates (list): List of candidate node indices
			current_load (float): Current load of the vehicle
			max_load (float): Maximum vehicle load capacity

		Returns:
			int or None: Selected next node index or None if no valid node
		"""
		feasible_candidates = []

		for candidate in candidates:
			# Check if adding this customer exceeds vehicle capacity
			if current_load + self.demands[candidate] <= max_load:
				feasible_candidates.append(candidate)

		if not feasible_candidates:
			return None

		# Calculate selection probabilities
		probs = []
		for candidate in feasible_candidates:
			# Pheromone and heuristic values
			tau = self.pheromone[current_node][candidate]
			eta = self.heuristic[current_node][candidate]
			prob = (tau ** self.alpha) * (eta ** self.beta)
			probs.append(prob)

		# Normalize probabilities
		sum_probs = sum(probs)
		if sum_probs > 0:
			probs = [p / sum_probs for p in probs]
		else:
			probs = [1.0 / len(feasible_candidates)] * len(feasible_candidates)

		# Select next node
		if random.random() < self.q0:
			# Greedy selection
			return feasible_candidates[np.argmax(probs)]
		else:
			# Probabilistic selection
			cum_prob = 0
			rand_num = random.random()

			for i, prob in enumerate(probs):
				cum_prob += prob
				if rand_num <= cum_prob:
					return feasible_candidates[i]

			return feasible_candidates[-1]  # Default if none selected

	def calculate_routes_cost(self, depot, routes):
		"""
		Calculate the total cost of routes for a depot.

		Args:
			depot (int): Depot index
			routes (list): List of routes (each route is a list of customer indices)

		Returns:
			float: Total cost of the routes
		"""
		total_cost = 0

		for route in routes:
			if not route:
				continue

			# Cost from depot to first customer
			cost = self.distance_matrix[depot][route[0]]

			# Cost between consecutive customers
			for i in range(len(route) - 1):
				cost += self.distance_matrix[route[i]][route[i + 1]]

			# Cost from last customer to depot
			cost += self.distance_matrix[route[-1]][depot]

			total_cost += cost

		return total_cost

	def update_pheromones(self, solutions):
		"""
		Update pheromone levels based on solutions.

		Args:
			solutions (list): List of (routes, cost) tuples
		"""
		# Evaporate pheromones
		self.pheromone = (1 - self.rho) * self.pheromone

		# Add new pheromones based on solutions
		for solution, cost in solutions:
			if cost == 0:
				continue

			delta_tau = 1.0 / cost

			for route in solution:
				for i in range(len(route) - 1):
					self.pheromone[route[i]][route[i + 1]] += delta_tau
					self.pheromone[route[i + 1]][route[i]] += delta_tau  # Symmetric

	def run(self):
		"""Run the ACO algorithm for MDVRP."""
		for iteration in range(self.max_iterations):
			solutions = []

			# Each ant constructs a solution
			for ant in range(self.num_ants):
				solution, cost = self.construct_solution(ant)
				solutions.append((solution, cost))

				# Update best solution
				if cost < self.best_cost:
					self.best_solution = solution
					self.best_cost = cost

			# Update pheromones
			self.update_pheromones(solutions)

			# Print progress
			if iteration % 10 == 0:
				print(f"Iteration {iteration}: Best cost = {self.best_cost:.2f}")

		return self.best_solution, self.best_cost

	def visualize_solution(self, solution):
		"""
		Visualize the solution.

		Args:
			solution (list): List of routes
		"""
		plt.figure(figsize=(10, 8))

		# Plot customers
		for i in range(self.num_customers):
			plt.plot(self.coordinates[i][0], self.coordinates[i][1], 'ko', markersize=5)
			plt.text(self.coordinates[i][0] + 0.3, self.coordinates[i][1] + 0.3, str(i), fontsize=8)

		# Plot depots
		for depot in self.depot_indices:
			plt.plot(self.coordinates[depot][0], self.coordinates[depot][1], 'rs', markersize=10)
			plt.text(self.coordinates[depot][0] + 0.3, self.coordinates[depot][1] + 0.3,
			         f"D{self.depot_indices.index(depot)}", fontsize=10)

		# Plot routes
		colors = plt.cm.tab20(np.linspace(0, 1, len(solution)))

		for i, route in enumerate(solution):
			color = colors[i % len(colors)]

			for j in range(len(route) - 1):
				x1, y1 = self.coordinates[route[j]]
				x2, y2 = self.coordinates[route[j + 1]]
				plt.plot([x1, x2], [y1, y2], '-', color=color, linewidth=1.5)

		plt.title(f"MDVRP Solution (Cost: {self.best_cost:.2f})")
		plt.xlabel("X Coordinate")
		plt.ylabel("Y Coordinate")
		plt.grid(True, linestyle='--', alpha=0.7)
		plt.tight_layout()
		plt.show()


# Example usage
if __name__ == "__main__":
	# Replace with your actual file path
	file_path = "test_data/p02"

	# Create ACO solver
	aco = MDVRP_ACO(
		file_path=file_path,
		num_ants=30,
		alpha=1.0,
		beta=5.0,
		rho=0.5,
		q0=0.9,
		max_iterations=100
	)

	# Run ACO algorithm
	solution, cost = aco.run()

	# Print final solution
	print(f"Final solution cost: {cost:.2f}")
	print("Number of routes:", len(solution))

	# Visualize solution
	aco.visualize_solution(solution)