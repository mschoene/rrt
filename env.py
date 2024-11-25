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
    def is_collision_free_path(env, node1, node2):
        """Check if path between two nodes is collision free"""
        # Create a few test points along the line
        points = 3 #cause we want to be efficient, add more if you want to be save
        for i in range(points):
            t = i / (points - 1)
            test_node = Node(
                node1.x + t * (node2.x - node1.x),
                node1.y + t * (node2.y - node1.y)
            )
            if not env.is_collision_free(test_node):
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


def find_neighbors(nodes, x, gamma=2, d=2.5): #.5):
    n = len(nodes)  # Total number of nodes in the tree

    threshold = gamma * (math.log(n) / n) ** (1 / d) 
    
    neighbors = []
    for node in nodes:
        distance = np.linalg.norm(np.array([node.x, node.y]) - np.array([x.x, x.y]))
        if distance <= threshold:
            neighbors.append(node)
    
    return neighbors


def node_distance(node, goal_node):
    return ((node.x - goal_node.x) ** 2 + (node.y - goal_node.y) ** 2) ** 0.5




def env_is_collision_free_path(env, node1, node2):
    """Check if path between two nodes is collision free"""
    # Create a few test points along the line
    points = 10
    for i in range(points):
        t = i / (points - 1)
        test_node = Node(
            node1.x + t * (node2.x - node1.x),
            node1.y + t * (node2.y - node1.y)
        )
        if not env.is_collision_free(test_node):
            return False
    return True


def rewire(new_node, neighbors, env, nodes):
    """
    Rewire the tree through the new node if it provides better paths
    
    Args:
        new_node: Newly added node
        neighbors: List of neighboring nodes
        env: Environment for collision checking
        nodes: Global list of all nodes
    """
    for neighbor in neighbors:
        # Calculate potential new cost through new_node
        potential_cost = new_node.cost + node_distance(new_node, neighbor)
        
        if potential_cost < neighbor.cost:
            # Check if path is collision free
            if env.is_collision_free_path(new_node, neighbor):
                neighbor.parent = new_node
                neighbor.cost = potential_cost
                # Update costs of all descendants
                update_descendant_costs(neighbor, nodes)


def choose_parent(new_node, neighbors, env):
    """
    Choose the best parent for a new node from its neighbors.
    
    Args:
        new_node: Node to find parent for
        neighbors: List of potential parent nodes
        env: Environment object for collision checking
    
    Returns:
        Best parent node and associated cost
    """
    if not neighbors:
        return None, float('inf')
        
    min_cost = float('inf')
    best_parent = None
    
    for potential_parent in neighbors:
        # Calculate new cost through this potential parent
        cost_through_parent = potential_parent.cost + new_node.distance_to(potential_parent)
        
        if cost_through_parent < min_cost:
            # Check if path to this parent is collision-free
            if not env.check_collision_line(new_node, potential_parent):
                min_cost = cost_through_parent
                best_parent = potential_parent

    return best_parent, min_cost


def update_descendant_costs(node, nodes):
    """Update costs of all nodes that have this node as their parent"""
    for descendant in nodes:
        if descendant.parent == node:
            descendant.cost = node.cost + node_distance(node, descendant)
            update_descendant_costs(descendant, nodes)



