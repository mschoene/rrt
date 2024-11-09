import random
import numpy as np

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

# Node Classe
class Node:
    def __init__(self, x, y, parent=None):
        self.x = x
        self.y = y
        self.parent = parent # for assigning 

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

