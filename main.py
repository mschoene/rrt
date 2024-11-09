import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon
import imageio
import os
from copy import copy
from PIL import Image
import gc  # For manual garbage collection
import random
from env import *

# Hyperparameters
GOAL_RADIUS = 0.3  # Threshold distance to goal for termination
STEP_SIZE = 0.5  # You can adjust this value as needed
MAX_STEPS = 50 # let's not go to inifinity and beyond!
GIF_FOLDER = 'randTree_1p1_frames'  # Folder to save frames
SAVE_INTERVAL = 100

def extend_towards_random(xtree, xrand, max_step_size=STEP_SIZE):
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


def check_goal_reached(node, goal_node, goal_radius=GOAL_RADIUS):
    # Calculate the distance between the current node and the goal
    distance_to_goal = np.linalg.norm([node.x - goal_node.x, node.y - goal_node.y])
    return distance_to_goal <= goal_radius



# Set up environment dimensions and obstacles
env = Environment(10, 10)
env.add_obstacle(Circle((3, 2.2), 1.8 ))  # circle
env.add_obstacle(Rectangle((2, 6), 1, 3 ))  # rectancle
env.add_obstacle(Rectangle((5.5, 0.5), 3.5, 3.5))  #  square
env.add_obstacle(Rectangle((7.05, 8.75), 0.5, 2.5, angle=140))  # m line left
env.add_obstacle(Rectangle((6.8, 7.2), 0.5, 1.5, angle=10))  # m middle left
env.add_obstacle(Rectangle((8.5, 7.1), 0.5, 1.7, angle=90))  # m middle right
env.add_obstacle(Rectangle((8.85, 7.3), 0.5, 2.5, angle=140))  # m line right




# Define start and end nodes
nodes = [Node(5, 5)] 
goal_node = Node(9, 9)
start_node = nodes[0]

# Visualize and Save each step
output_folder = "motion_planning_steps"
os.makedirs(output_folder, exist_ok=True)


# Draw environment and obstacles
def draw_environment(env, nodes, goal_node, step, folder_path, goal_reached = False):
    fig, ax = plt.subplots(figsize=(6, 6))
    # Draw obstacles here if needed (using ax.add_patch())
    
    # Draw nodes and edges
    for node in nodes:
        ax.plot(node.x, node.y, 'go')  # 'go' is green dot for nodes
        if node.parent:
            ax.plot([node.x, node.parent.x], [node.y, node.parent.y], 'g-', linewidth=1)
    
    # Draw the start and goal nodes
    plt.plot(nodes[0].x, nodes[0].y, 'bo', markersize=6, label="Start Node")
    plt.plot(goal_node.x, goal_node.y, 'ro', markersize=6, label="Goal Node")
    

    # Highlight the latest node in orange
    new_node = nodes[-1]
    plt.plot(new_node.x, new_node.y, 'o', color='orange', markersize=5)

    # Draw shortest path in red
    draw_shortest_path(ax, nodes, goal_node, goal_reached)
    
    # Add a legend
    plt.legend(loc="upper right")
    

    # Draw obstacles
    for obstacle in env.obstacles:
        ob_cop = copy(obstacle)
        ax.add_patch(ob_cop)

    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_title(f"Step {step}")
    
    # Save the current figure as an image every SAVE_INTERVAL steps
    #if step % SAVE_INTERVAL == 0:
    #filename = os.path.join(folder_path, f"step_{step}.png")
    filename = os.path.join(folder_path, f"step_{step:04d}.png")  # Zero-padded to 4 digits
    plt.savefig(filename, dpi=100)
    plt.close(fig)



def draw_shortest_path(ax, nodes, goal_node, goal_reached ):
    # Find the nearest node to the goal (if the goal isn't reached directly)
    nearest_node = get_nearest_node(nodes, goal_node)
    
    # Trace the path back from the goal (or the nearest node) to the start
    path = []
    current_node = nearest_node
    while current_node is not None:
        path.append(current_node)
        current_node = current_node.parent
    
    # Reverse the path to go from start to goal
    path = path[::-1]

    color = 'b-'
    if goal_reached:
        color = 'r-'

    # Draw the path
    for i in range(len(path) - 1):
        ax.plot([path[i].x, path[i+1].x], [path[i].y, path[i+1].y], color, linewidth=2)



# ex 1.1 rand tree with no actual plan
def planning_algorithm(start_node, goal_node, env, max_steps=MAX_STEPS, folder_path=GIF_FOLDER):
    nodes = [start_node]
    step_count = 0
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    while step_count < max_steps:
        random_node = sample_random_node(nodes) #get a rand node 
        random_point = sample_random_point(env) # get a rand point in the env
        new_node = extend_towards_random(random_node, random_point, max_step_size=STEP_SIZE)
        
        
        # Only add node if it is collision-free
        if env.is_collision_free(new_node):  
            nodes.append(new_node)
        

        draw_environment(env, nodes, goal_node, step=step_count, folder_path=folder_path)

        if check_goal_reached(new_node, goal_node, goal_radius=GOAL_RADIUS):        
            draw_environment(env, nodes, goal_node, step=step_count, folder_path=folder_path, goal_reached = True)

            print("Goal reached!")
            break
    
        step_count += 1
        if step_count % SAVE_INTERVAL == 0:
            gc.collect()  # Trigger garbage collection every SAVE_INTERVAL steps for efficiency
    
    if step_count == max_steps:
        print(f"Reached max steps ({max_steps}) without finding the goal.")
    return nodes


def frame_generator(folder_path):
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.png'):
            yield Image.open(os.path.join(folder_path, filename))


def create_gif(folder_path, gif_name='planning.gif', duration=200):
    first_frame = next(frame_generator(folder_path))  # Get the first frame
    first_frame.save(
        gif_name,
        save_all=True,
        append_images=frame_generator(folder_path),
        duration=duration,
        loop=0
    )
# 
# def create_gif(folder_path, gif_name='planning.gif', duration=200):
    # frames = [Image.open(os.path.join(folder_path, f)) for f in sorted(os.listdir(folder_path)) if f.endswith('.png')]
    # frames[0].save(gif_name, save_all=True, append_images=frames[1:], duration=duration, loop=0)


nodes = planning_algorithm(start_node, goal_node, env, max_steps=MAX_STEPS, folder_path=GIF_FOLDER)
create_gif(GIF_FOLDER, gif_name='planning.gif') #, duration=200)