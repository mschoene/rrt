import numpy as np
from env import *
from PIL import Image
import os
import matplotlib.pyplot as plt
from copy import copy
from matplotlib.patches import Circle

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





def draw_shortest_path(ax, nodes, goal_node, goal_reached, current_shortest_path):
    # Find the nearest node to the goal (if the goal isn't reached directly)
    nearest_node = get_nearest_node(nodes, goal_node)
    
    path = []
    current_node = nearest_node
    path_cost = 0  # Track the cumulative cost of the path

    while current_node is not None:
        path.append(current_node)
        if current_node.parent is not None:
            # Add the actual distance (cost) from the current node to its parent
            path_cost += current_node.cost  # Use node's cost attribute for accurate path cost
        current_node = current_node.parent
    
    # Reverse to go from start to goal
    path = path[::-1]
    
    if path_cost < current_shortest_path:
        current_shortest_path = path_cost
        if goal_reached:
            for i in range(len(path) - 1):
                ax.plot([path[i].x, path[i+1].x], [path[i].y, path[i+1].y], 'r-', linewidth=2)
    
    return current_shortest_path
    

# Draw environment and obstacles
def draw_environment(env, nodes, goal_node, step, folder_path, goal_radius, goal_reached = False, current_shortest_path = float('inf')):
    fig, ax = plt.subplots(figsize=(6, 6))
    # Draw obstacles here if needed (using ax.add_patch())
    
    # Draw nodes and edges
    for node in nodes:
        ax.plot(node.x, node.y, 'go')  # 'go' is green dot for nodes
        if node.parent:
            ax.plot([node.x, node.parent.x], [node.y, node.parent.y], 'g-', linewidth=1)
    
    # Draw the start and goal nodes
    ax.plot(nodes[0].x, nodes[0].y, 'bo', label="Start Node")
    ax.plot(goal_node.x, goal_node.y, 'ro', label="Goal Node")
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.15))#

    goal_radius_circle = Circle((goal_node.x, goal_node.y), goal_radius, color='red', alpha=0.2)
    ax.add_patch(goal_radius_circle)
    # Highlight the latest node in orange
    new_node = nodes[-1]
    plt.plot(new_node.x, new_node.y, 'o', color='orange', markersize=5)

    # Draw shortest path (red if found)
    shortest_path_length = draw_shortest_path(ax, nodes, goal_node, goal_reached, current_shortest_path)


    # Draw obstacles
    for obstacle in env.obstacles:
        ob_cop = copy(obstacle)
        ax.add_patch(ob_cop)

    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    title = f"Step {step}"
    if goal_reached:
        title = title + f"   Path length = {shortest_path_length:.2f}   "
    else:
        title = title + f"   Path length = NaN   "
    ax.set_title(title)
    
    # Save the current figure as an image every SAVE_INTERVAL steps
    filename = os.path.join(folder_path, f"step_{step:04d}.png")  # Zero-padded to 4 digits
    plt.savefig(filename, dpi=100)
    plt.close(fig)

    return shortest_path_length
