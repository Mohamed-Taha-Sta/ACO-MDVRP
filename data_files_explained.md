# MDVRP Input File Format Explanation (p01.txt)

The `p01.txt` file contains the problem definition for a Multi-Depot Vehicle Routing Problem. Here's the breakdown:

## First Line: Problem Parameters
```
2 4 50 4
```
- `2`: Problem type (2 = MDVRP)
- `4`: Number of vehicles per depot
- `50`: Number of customers
- `4`: Number of depots

## Depot Capacity Information (Lines 2-5)
```
0 80
0 80
0 80
0 80
```
For each depot:
- First value (`0`): Maximum route duration (not used)
- Second value (`80`): Maximum vehicle capacity

## Customer Data (Lines 6-55)
```
 1 37 52 0   7 1 4 1 2 4 8
 2 49 49 0  30 1 4 1 2 4 8
 ...
```
Each customer line contains:
- Customer ID (1-50)
- X-coordinate
- Y-coordinate
- Service duration (`0`)
- Demand (e.g., `7`, `30`)
- Additional parameters (not used in the algorithm)

## Depot Coordinates (Lines 56-59)
```
51 20 20 0   0 0 0
52 30 40 0   0 0 0
53 50 30 0   0 0 0
54 60 50 0   0 0 0
```
Each depot line contains:
- Depot ID (51-54)
- X-coordinate
- Y-coordinate
- Zeros for unused fields

The algorithm uses this information to:
1. Create a network with customers and depots
2. Assign customers to depots
3. Design routes that respect vehicle capacity constraints
4. Minimize total travel distance

The code shows this is properly handled in the `parse_data_file()` method of the `MDVRP_ACO` class.