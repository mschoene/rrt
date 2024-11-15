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
from utils import *
import gradio as gr
import time


# Hyperparameters
STEP_SIZE = 0.5  # You can adjust this value as needed
GOAL_RADIUS = STEP_SIZE
MAX_STEPS = 50 # let's not go to inifinity and beyond!
GIF_FOLDER = 'randTree_1p1_frames'  # Folder to save frames
FINAL_FRAME_PATH = os.path.join(GIF_FOLDER, "final_frame.png")
REWIRE_RADIUS = 0.7

# Global variables to track simulation state
global_nodes = None
global_step_count = 0
global_goal_reached = False
global_shortest_cost = float('inf')

# Set up environment dimensions and obstacles
env = Environment(10, 10)
env.add_obstacle(Circle((3, 2.2), 1.3 ))  # circle
env.add_obstacle(Rectangle((2, 6), 1, 3 ))  # rectancle
env.add_obstacle(Rectangle((5.5, 0.5), 3, 3))  #  square

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







# Function to run a specific number of steps in the simulation
def run_simulation_streaming(steps, algorithm_choice = "RT", heuristic_probability=0): 
    global global_nodes, global_step_count, global_goal_reached, global_shortest_cost
    
    if global_nodes is None:  # Reset state if no existing nodes
        start_node = Node(5, 5)
        global_nodes = [start_node]
        global_step_count = 0
        global_goal_reached = False
        global_shortest_cost = float('inf')
        # Clear GIF folder
        if os.path.exists(GIF_FOLDER):
            for file in os.listdir(GIF_FOLDER):
                os.remove(os.path.join(GIF_FOLDER, file))
    
    goal_node = Node(9, 9)

    start_time = time.time()  # Start timer

    for _ in range(steps):
        # sample until you find a point without collissions.
        while True:
            random_point = sample_random_point(env)
            parent_node = None

            if algorithm_choice == "RT":
                # Heuristic-based node selection
                if random.random() < heuristic_probability:  # Select nearest node based on heuristic
                    parent_node = get_nearest_node(global_nodes, goal_node)
                else:  # Otherwise, select a random node
                    parent_node = sample_random_node(global_nodes)
                
                # Try extending towards the random point
                new_node = extend_tree_towards(parent_node, random_point, max_step_size=STEP_SIZE)

                # Check if the new node is collision-free; if so, break out of the loop
                if env.is_collision_free(new_node):
                    global_nodes.append(new_node)
                    break  # Exit the resampling loop if the node is valid



            elif algorithm_choice == "RRT":
                # Select the nearest node to the random point in RRT
                #parent_node = min(global_nodes, key=lambda node: straight_line_distance_to_goal(node, random_point))
                parent_node = get_nearest_node(global_nodes, random_point)
                
                # Try extending towards the random point
                new_node = extend_tree_towards(parent_node, random_point, max_step_size=STEP_SIZE)

                # Check if the new node is collision-free; if so, break out of the loop
                if env.is_collision_free(new_node):
                    global_nodes.append(new_node)
                    break  # Exit the resampling loop if the node is valid



            elif algorithm_choice == "RRT*":
                # For RRT*, connect to nearest node but optimize the path with rewiring
                #parent_node = min(nodes, key=lambda node: straight_line_distance_to_goal(node, random_point))
                parent_node = get_nearest_node(global_nodes, random_point)

                new_node = extend_tree_towards(parent_node, random_point, max_step_size=STEP_SIZE)

                if env.is_collision_free(new_node):
                    # Check nearby nodes within a radius to rewire
                    neighbors = find_neighbors(global_nodes, new_node) #, radius=REWIRE_RADIUS)
                    best_parent = min(neighbors, key=lambda n: n.cost + node_distance(n, new_node))

                    # Connect the new node to the best parent
                    new_node.parent = best_parent
                    new_node.cost = best_parent.cost + node_distance(best_parent, new_node)

                    # Rewire neighbors to the new node if it provides a shorter path
                    for neighbor in neighbors:
                        new_cost = new_node.cost + node_distance(new_node, neighbor)
                        if new_cost < neighbor.cost:
                            neighbor.parent = new_node
                            neighbor.cost = new_cost

                    global_nodes.append(new_node)
                    break
                


        if global_goal_reached:
            shortest_path_length = draw_environment(env, global_nodes, goal_node, global_step_count, folder_path=GIF_FOLDER, goal_radius=GOAL_RADIUS, goal_reached=global_goal_reached)
        else: # if it wasn't reached before eval  the new node 
            global_goal_reached = check_goal_reached(new_node, goal_node, goal_radius=GOAL_RADIUS)
            shortest_path_length = draw_environment(env, global_nodes, goal_node, global_step_count, folder_path=GIF_FOLDER, goal_radius=GOAL_RADIUS, goal_reached=global_goal_reached)

        
        global_step_count += 1
        frame_counter = global_step_count - 1
        final_frame_path = os.path.join(GIF_FOLDER, f"step_{frame_counter:04d}.png")
        
        if os.path.exists(final_frame_path):
            frame = Image.open(final_frame_path).convert("RGB")
            frame.save(FINAL_FRAME_PATH)  # Save the final frame explicitly as RGB
            yield frame, None  # Yield the frame and None as a placeholder for GIF
            
        if global_goal_reached:
            if shortest_path_length < global_shortest_cost:
                global_shortest_cost = shortest_path_length
                break
    
        
    end_time = time.time()  # End timer
    elapsed_time = end_time - start_time
    print(f"Simulation ran for {elapsed_time:.2f} seconds.")
    

    # Generate GIF only at the end of the steps
    gif_path = os.path.join(GIF_FOLDER, "simulation.gif")
    create_gif(GIF_FOLDER, gif_name=gif_path, duration=max(global_step_count,200) )
    yield frame, gif_path  # Yield final frame and the GIF path after completion of steps

# Reset function for simulation
def reset_simulation():
    global global_nodes, global_step_count, global_goal_reached, global_shortest_cost
    global_nodes = None
    global_step_count = 0
    global_goal_reached = False
    if os.path.exists(FINAL_FRAME_PATH):
        os.remove(FINAL_FRAME_PATH)
    return None, None, None



# Gradio UI update with slider for heuristic probability
with gr.Blocks() as demo:
    gr.Markdown("### RT/RRT/RRT*")

    algorithm_choice = gr.Radio(["RT", "RRT", "RRT*"], label="Algorithm Choice", value="RT")    

    with gr.Row():
        heuristic_slider = gr.Slider(minimum=0, maximum=1, value=1, step=0.01, label="Heuristic Probability")
    
    with gr.Row():
        add_1_button = gr.Button("Add 1 Step")
        add_10_button = gr.Button("Add 10 Steps")
        add_50_button = gr.Button("Add 50 Steps")
        add_100_button = gr.Button("Add 100 Steps")
        add_500_button = gr.Button("Add 500 Steps")
        reset_button = gr.Button("Reset Simulation")
    
    with gr.Row():
        final_frame_output = gr.Image(type="pil", label="Current graph", streaming=True)
        gif_output = gr.Image(type="filepath", label="GIF")

    #button funcs: pass args
    add_1_button.click(run_simulation_streaming, inputs=[gr.Number(value=1), algorithm_choice], outputs=[final_frame_output, gif_output])
    add_10_button.click(run_simulation_streaming, inputs=[gr.Number(value=10), algorithm_choice], outputs=[final_frame_output, gif_output])
    add_50_button.click(run_simulation_streaming, inputs=[gr.Number(value=50), algorithm_choice], outputs=[final_frame_output, gif_output])
    add_100_button.click(run_simulation_streaming, inputs=[gr.Number(value=100), algorithm_choice], outputs=[final_frame_output, gif_output])
    add_500_button.click(run_simulation_streaming, inputs=[gr.Number(value=500), algorithm_choice], outputs=[final_frame_output, gif_output])
    reset_button.click(reset_simulation, outputs=[final_frame_output, gif_output])


demo.launch()