import random
import numpy as np
import math

# Define environment and obstacles
class Environment:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.obstacles = []

    def add_obstacle(self, obstacle):
        self.obstacles.append(obstacle)

    def is_collision_free(self, node):
        """Check if a point is free from obstacles."""
        for obstacle in self.obstacles:
            if obstacle.contains_point((node.x, node.y)):
                return False
        return True
    

class Node:
    def __init__(self, x, y, parent=None):
        self.x = x
        self.y = y
        self.parent = parent  # parent ref
        self.cost = 0.0 if parent is None else parent.cost + node_distance(self, parent)  # Initialize cost

    def __repr__(self):
        return f"Node(x={self.x}, y={self.y}, cost={self.cost:.2f})"
    
    def distance_to(self, other):
        """Calculate Euclidean distance to another node."""
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5
    


# Sample a random point in the environment and return as node without parent
def sample_random_point(env):
    x_rand = random.uniform(0, env.width)
    y_rand = random.uniform(0, env.height)
    return Node(x_rand, y_rand)

def sample_random_node(nodes):
    return random.choice(nodes)



def get_nearest_node(nodes, random_node):
    nearest_node = nodes[0]
    min_distance = np.linalg.norm([nearest_node.x - random_node.x, nearest_node.y - random_node.y])
    
    for node in nodes[1:]:
        distance = np.linalg.norm([node.x - random_node.x, node.y - random_node.y])
        if distance < min_distance:
            nearest_node = node
            min_distance = distance
    return nearest_node



def extend_tree_towards(xtree, xrand, max_step_size):
    # Compute the direction vector from xtree to xrand
    direction = np.array([xrand.x - xtree.x, xrand.y - xtree.y])
    distance = np.linalg.norm(direction)  # Euclidean distance between xtree and xrand
    
    if distance > max_step_size:
        # Scale the direction to the max step size
        direction = direction / distance * max_step_size
        new_x = xtree.x + direction[0]
        new_y = xtree.y + direction[1]
    else:
        new_x = xrand.x
        new_y = xrand.y
    
    return Node(new_x, new_y, parent=xtree  )  # Return the new node


def check_goal_reached(node, goal_node, goal_radius):
    # Calculate the distance between the current node and the goal
    distance_to_goal = np.linalg.norm([node.x - goal_node.x, node.y - goal_node.y])
    return distance_to_goal <= goal_radius


# def find_neighbors(nodes, new_node, radius):
    # """Return a list of nodes within a specified radius from new_node."""
    # return [node for node in nodes if node_distance(node, new_node) <= radius]
# 

def find_neighbors(nodes, x, gamma=0.75, d=2, max_threshold=0.50, min_threshold=0.0001):
    n = len(nodes)  # Total number of nodes in the tree
    threshold = min(gamma * (math.log(n) / n) ** (1 / d)+0.0001, 10.)  #  threshold distance

    print(threshold)
    
    neighbors = []

    for node in nodes:
        # Calculate Euclidean distance between node and point x
        distance = np.linalg.norm(np.array([node.x, node.y]) - np.array([x.x, x.y])) 
        if distance <= threshold:
            neighbors.append(node)

    while len(neighbors) == 0:
        threshold *= 1.2
        print(f"No neighbors found. New threshold: {threshold}")
        
        # Recalculate neighbors with the updated threshold
        neighbors = []
        for node in nodes:
            distance = np.linalg.norm(np.array([node.x, node.y]) - np.array([x.x, x.y])) 
            if distance <= threshold:
                neighbors.append(node)

        # Optionally, stop if the threshold exceeds a maximum value
        if threshold > max_threshold :  # You can adjust this multiplier as needed
            print("Threshold exceeded maximum limit, stopping search.")
            break
    
    return neighbors

def node_distance(node, goal_node):
    return ((node.x - goal_node.x) ** 2 + (node.y - goal_node.y) ** 2) ** 0.5
