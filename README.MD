# TAMP Assignment implementation
## Truly understanding sampling-based motion planning

## Overview

This repository implements sampling algorithm to more or less efficiently explore a 2D environment. Gradio is used to 

* RT (with heuristic probability slider)
* RRT
* RRT* 

## Usage



Clone the repository, ceate a conda environment and install the requirements.
```
conda create --name mopla
conda activate mopla 
pip install -r requirements.txt
```

And then just run 
```
python main.py
```

and open the generated link. Play around with the different algorithms and settings. Remember to save a gif of your test if you wish to keep it such as the examples below. 

![](https://github.com/mschoene/rrt/blob/main/gifs/rt.gif)
![](https://github.com/mschoene/rrt/blob/main/gifs/rt_heuristic.gif)

![](https://github.com/mschoene/rrt/blob/main/gifs/rrt.gif)
![](https://github.com/mschoene/rrt/blob/main/gifs/rrt_star.gif)
