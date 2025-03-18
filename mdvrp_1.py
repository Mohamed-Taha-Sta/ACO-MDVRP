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
		self.num_vehicles_per_depot = int(first_line[1])  # Now correctly understood as vehicles per depot
		self.num_customers = int(first_line[2])
		self.num_depots = int(first_line[3])

		# Verify problem type
		if self.problem_type != 2:
			raise ValueError("This is not a MDVRP (type 2) problem.")

		# Parse depot information
		self.max_durations = []
		self.max_loads = []
		self.vehicles_per_depot = []  # Store number of vehicles for each depot
		for i in range(1, self.num_depots + 1):
			depot_info = lines[i].strip().split()
			self.max_durations.append(float(depot_info[0]))
			self.max_loads.append(float(depot_info[1]))
			self.vehicles_per_depot.append(self.num_vehicles_per_depot)  # Each depot gets the specified number

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

	def assign_customers_to_depots(self):
		"""
		Assign customers to depots based on pheromone trails and distance.

		Returns:
			list: Lists of customers assigned to each depot
		"""
		depot_customers = [[] for _ in range(self.num_depots)]
		unassigned_customers = list(range(self.num_customers))

		while unassigned_customers:
			# For each unassigned customer, find the best depot
			assignments = []

			for customer in unassigned_customers:
				best_depot = None
				best_value = -float('inf')

				for depot_idx, depot in enumerate(self.depot_indices):
					# Calculate combined pheromone and heuristic value
					tau = self.pheromone[customer][depot]
					eta = self.heuristic[customer][depot]
					value = (tau ** self.alpha) * (eta ** self.beta)

					if value > best_value:
						best_value = value
						best_depot = depot_idx

				assignments.append((customer, best_depot, best_value))

			# Sort assignments by value (highest first)
			assignments.sort(key=lambda x: x[2], reverse=True)

			# Take the best assignment
			customer, depot_idx, _ = assignments[0]
			depot_customers[depot_idx].append(customer)
			unassigned_customers.remove(customer)

		return depot_customers

	def construct_solution(self, ant_idx):
		"""
		Construct a solution for a single ant.

		Args:
			ant_idx (int): Index of the ant

		Returns:
			list: Routes for each depot
			float: Total cost of the solution
		"""
		# Get initial customer-to-depot assignments based on pheromone and distance
		depot_customers = self.assign_customers_to_depots()

		# Construct routes for each depot
		total_cost = 0
		final_routes = []

		for depot_idx, customers in enumerate(depot_customers):
			if not customers:
				continue

			depot = self.depot_indices[depot_idx]
			max_load = self.max_loads[depot_idx]
			max_vehicles = self.vehicles_per_depot[depot_idx]

			# Split customers into routes using ACO, respecting vehicle limit
			depot_routes = self.construct_depot_routes(depot, customers, max_load, max_vehicles)

			# If we couldn't fit all customers within vehicle constraints, solution is infeasible
			if depot_routes is None:
				return [], float('inf')  # Return infeasible solution

			route_cost = self.calculate_routes_cost(depot, depot_routes)
			total_cost += route_cost

			# Add the depot's routes to the final solution
			for route in depot_routes:
				if route:  # Only add non-empty routes
					final_route = [depot] + route + [depot]
					final_routes.append(final_route)

		return final_routes, total_cost

	def construct_depot_routes(self, depot, customers, max_load, max_vehicles):
		"""
		Construct routes for a single depot using ACO, respecting vehicle limits.
		Routes are constructed by ants starting from the depot and then making decisions based on current node.

		Args:
			depot (int): Index of the depot
			customers (list): List of customer indices assigned to this depot
			max_load (float): Maximum vehicle load capacity
			max_vehicles (int): Maximum number of vehicles available at this depot

		Returns:
			list or None: List of routes or None if no feasible solution
		"""
		routes = []
		remaining_customers = customers.copy()

		# Enforce vehicle limit
		while remaining_customers and len(routes) < max_vehicles:
			route = []
			current_load = 0
			current_node = depot  # Start at depot (important change)

			# Build route by having ant travel from current node each time
			while remaining_customers:
				# Select next customer from current node (not just from depot)
				next_node = self.select_next_node(
					current_node,
					remaining_customers,
					current_load,
					max_load,
					depot  # Pass depot as reference for return check
				)

				if next_node is not None:
					route.append(next_node)
					current_load += self.demands[next_node]
					remaining_customers.remove(next_node)
					current_node = next_node  # Update current position of ant
				else:
					# No more customers can be added to this route
					break

			if route:
				routes.append(route)
			elif not routes:
				# If we can't even create one route, problem is infeasible
				return None

		# Check if all customers were assigned
		if remaining_customers:
			# Not enough vehicles to service all customers
			return None

		return routes

	def select_next_node(self, current_node, candidates, current_load, max_load, depot=None):
		"""
		Select the next node to visit using ACO rules from the current node (not just from depot).

		Now includes consideration of returning to depot if no viable customers remain.

		Args:
			current_node (int): Current node index (customer or depot)
			candidates (list): List of candidate node indices
			current_load (float): Current load of the vehicle
			max_load (float): Maximum vehicle load capacity
			depot (int, optional): Depot index, to evaluate return trip feasibility

		Returns:
			int or None: Selected next node index or None if should return to depot
		"""
		feasible_candidates = []

		for candidate in candidates:
			# Check if adding this customer exceeds vehicle capacity
			if current_load + self.demands[candidate] <= max_load:
				feasible_candidates.append(candidate)

		if not feasible_candidates:
			return None  # Return to depot

		# Calculate probabilities including pheromone influence from current node
		probs = []
		for candidate in feasible_candidates:
			# Pheromone and heuristic values from CURRENT NODE (not depot)
			tau = self.pheromone[current_node][candidate]
			eta = self.heuristic[current_node][candidate]

			# Calculate probability based on ACO formula
			prob = (tau ** self.alpha) * (eta ** self.beta)

			# Add a slight bias toward customers that are closer to depot for better routes
			if depot is not None:
				# Consider ease of return to depot as a small factor
				return_ease = self.heuristic[candidate][depot] ** 0.5  # Square root to reduce influence
				prob *= (1 + 0.1 * return_ease)  # Small 10% boost for easy returns

			probs.append(prob)

		# Normalize probabilities
		sum_probs = sum(probs) + 1e-10  # Avoid division by zero
		probs = [p / sum_probs for p in probs]

		# Select next node
		if random.random() < self.q0:
			# Exploitation (greedy selection)
			return feasible_candidates[np.argmax(probs)]
		else:
			# Exploration (probabilistic selection)
			# Roulette wheel selection
			cum_prob = 0
			rand_val = random.random()

			for i, prob in enumerate(probs):
				cum_prob += prob
				if rand_val <= cum_prob:
					return feasible_candidates[i]

			# Fallback
			return feasible_candidates[-1]

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
			if cost == 0 or cost == float('inf'):  # Skip infeasible solutions
				continue

			delta_tau = 1.0 / cost

			for route in solution:
				for i in range(len(route) - 1):
					self.pheromone[route[i]][route[i + 1]] += delta_tau
					self.pheromone[route[i + 1]][route[i]] += delta_tau  # Symmetric

	def local_search(self, solution, cost):
		"""
		Apply a local search to improve the solution.
		Implements a simple 2-opt heuristic on each route.

		Args:
			solution (list): Current solution
			cost (float): Current solution cost

		Returns:
			tuple: Improved solution and its cost
		"""
		if not solution or cost == float('inf'):
			return solution, cost

		improved = True
		best_solution = copy.deepcopy(solution)
		best_cost = cost

		while improved:
			improved = False

			# Try to improve each route with 2-opt
			for route_idx, route in enumerate(best_solution):
				if len(route) <= 3:  # Skip routes that are too short (depot-customer-depot)
					continue

				# The endpoints are depots, so we only consider internal customers
				for i in range(1, len(route) - 2):
					for j in range(i + 1, len(route) - 1):
						# Skip adjacent nodes
						if j == i + 1:
							continue

						# Create new route with 2-opt exchange
						new_route = route[:i + 1] + route[j:i:-1] + route[j + 1:]

						# Check if new route is better
						old_cost = (self.distance_matrix[route[i - 1]][route[i]] +
						            self.distance_matrix[route[j]][route[j + 1]])
						new_cost = (self.distance_matrix[route[i - 1]][route[j]] +
						            self.distance_matrix[route[i]][route[j + 1]])

						if new_cost < old_cost:
							# Apply the improvement
							new_solution = copy.deepcopy(best_solution)
							new_solution[route_idx] = new_route

							# Recalculate total cost
							new_total_cost = 0
							for r in new_solution:
								depot = r[0]  # First node is depot
								new_total_cost += self.calculate_route_cost(r)

							if new_total_cost < best_cost:
								best_solution = new_solution
								best_cost = new_total_cost
								improved = True

		return best_solution, best_cost

	def calculate_route_cost(self, route):
		"""Calculate cost of a single route."""
		cost = 0
		for i in range(len(route) - 1):
			cost += self.distance_matrix[route[i]][route[i + 1]]
		return cost

	def run(self):
		"""Run the ACO algorithm for MDVRP."""
		for iteration in range(self.max_iterations):
			solutions = []

			# Each ant constructs a solution
			for ant in range(self.num_ants):
				solution, cost = self.construct_solution(ant)

				# Apply local search to improve solution
				if cost < float('inf'):
					solution, cost = self.local_search(solution, cost)

				solutions.append((solution, cost))

				# Update best solution
				if cost < self.best_cost and cost != float('inf'):
					self.best_solution = solution
					self.best_cost = cost
					print(f"New best solution found: {cost:.2f}")

			# Update pheromones
			self.update_pheromones(solutions)

			# Print progress
			if iteration % 10 == 0:
				print(f"Iteration {iteration}: Best cost = {self.best_cost:.2f}")

		if self.best_cost == float('inf'):
			print("No feasible solution found. Try increasing the number of vehicles per depot.")

		return self.best_solution, self.best_cost

	def visualize_solution(self, solution):
		"""
		Visualize the solution.

		Args:
			solution (list): List of routes
		"""
		if not solution:
			print("No solution to visualize")
			return

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

		# Plot routes with different colors for different depots
		# Group routes by depot
		depot_routes = {}
		for i, route in enumerate(solution):
			if route[0] not in depot_routes:
				depot_routes[route[0]] = []
			depot_routes[route[0]].append(route)

		# Get a color map with enough colors
		cmap = plt.cm.get_cmap('tab10', len(self.depot_indices))

		route_idx = 0
		for depot, routes in depot_routes.items():
			depot_idx = self.depot_indices.index(depot)
			base_color = cmap(depot_idx)

			for i, route in enumerate(routes):
				# Adjust alpha for different routes from same depot
				alpha = 0.7 - (0.3 * (i % 3) / len(routes))
				color = list(base_color)
				color[3] = alpha  # Modify alpha

				# Draw route segments
				for j in range(len(route) - 1):
					x1, y1 = self.coordinates[route[j]]
					x2, y2 = self.coordinates[route[j + 1]]
					plt.plot([x1, x2], [y1, y2], '-', color=color, linewidth=1.5)

				# Add directional arrows to show route direction
				for j in range(len(route) - 1):
					if j % 3 == 1:  # Add arrows periodically, not on every segment
						x1, y1 = self.coordinates[route[j]]
						x2, y2 = self.coordinates[route[j + 1]]
						dx, dy = x2 - x1, y2 - y1
						# Normalize and place arrow at middle of segment
						arrow_x, arrow_y = x1 + dx * 0.6, y1 + dy * 0.6
						plt.arrow(arrow_x, arrow_y, dx * 0.05, dy * 0.05,
						          head_width=0.8, head_length=0.8, fc=color, ec=color)

				route_idx += 1

		# Count vehicles used per depot
		vehicles_used = {}
		for route in solution:
			depot = route[0]
			if depot not in vehicles_used:
				vehicles_used[depot] = 0
			vehicles_used[depot] += 1

		# Add legend with vehicle counts
		legend_entries = []
		for depot_idx, depot in enumerate(self.depot_indices):
			if depot in vehicles_used:
				legend_entries.append(
					f"Depot {depot_idx}: {vehicles_used[depot]}/{self.vehicles_per_depot[depot_idx]} vehicles")
			else:
				legend_entries.append(f"Depot {depot_idx}: 0/{self.vehicles_per_depot[depot_idx]} vehicles")

		plt.legend(legend_entries)
		plt.title(f"MDVRP Solution (Cost: {self.best_cost:.2f})")
		plt.xlabel("X Coordinate")
		plt.ylabel("Y Coordinate")
		plt.grid(True, linestyle='--', alpha=0.7)
		plt.tight_layout()
		plt.show()

		# Show stats
		self.print_solution_stats(solution)

	def print_solution_stats(self, solution):
		"""Print detailed statistics about the solution."""
		if not solution:
			return

		print("\n--- Solution Statistics ---")
		print(f"Total cost: {self.best_cost:.2f}")
		print(f"Total routes: {len(solution)}")

		# Analyze by depot
		depot_stats = {}
		for route in solution:
			depot = route[0]
			if depot not in depot_stats:
				depot_stats[depot] = {
					'routes': 0,
					'total_distance': 0,
					'total_demand': 0,
					'customers': 0
				}

			# Calculate route statistics
			route_distance = 0
			route_demand = 0

			for i in range(len(route) - 1):
				route_distance += self.distance_matrix[route[i]][route[i + 1]]
				if i > 0 and i < len(route) - 1:  # Skip depots
					route_demand += self.demands[route[i]]

			depot_stats[depot]['routes'] += 1
			depot_stats[depot]['total_distance'] += route_distance
			depot_stats[depot]['total_demand'] += route_demand
			depot_stats[depot]['customers'] += len(route) - 2  # Subtract depot at both ends

		# Print depot statistics
		print("\nDepot Statistics:")
		for depot_idx, depot in enumerate(self.depot_indices):
			if depot in depot_stats:
				stats = depot_stats[depot]
				print(f"\nDepot {depot_idx}:")
				print(f"  Routes: {stats['routes']}/{self.vehicles_per_depot[depot_idx]}")
				print(f"  Total distance: {stats['total_distance']:.2f}")
				print(f"  Total demand: {stats['total_demand']:.2f}")
				print(f"  Customers served: {stats['customers']}")
				print(f"  Average customers per route: {stats['customers'] / stats['routes']:.2f}")
			else:
				print(f"\nDepot {depot_idx}: No routes")


# Example usage
if __name__ == "__main__":
	# Replace with your actual file path
	file_path = "test_data/p01.txt"

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
	print(f"\nFinal solution cost: {cost:.2f}")
	if solution:
		print("Number of routes:", len(solution))

		# Analyze vehicle usage by depot
		depot_vehicle_count = {depot: 0 for depot in aco.depot_indices}
		for route in solution:
			depot = route[0]  # First node is the depot
			depot_vehicle_count[depot] += 1

		print("\nVehicle usage by depot:")
		for depot_idx, depot in enumerate(aco.depot_indices):
			print(f"Depot {depot_idx}: {depot_vehicle_count[depot]}/{aco.vehicles_per_depot[depot_idx]} vehicles")

		# Visualize solution
		aco.visualize_solution(solution)
	else:
		print("No feasible solution found")