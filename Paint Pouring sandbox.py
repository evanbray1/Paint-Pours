import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
from matplotlib import colorbar, use
from scipy.ndimage import gaussian_filter
from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np
import time
import os
import cv2
import paint_pour_tools as pptools
use('TkAgg')
# Show the plots in the iPython console
# %matplotlib inline 


#########USER-DEFINED VARIABLES#########
image_dimensions = [1920, 1080]

display_final_image = True        #Do you want to display the image on the screen?
save_image = True                 #Do you want to save a .png copy of your image?
num_images = 10                    #How many images do you want to produce?

show_intermediate_plots = False  #Do you want to show some intermediate results to help with troubleshooting?
seed=None                        #Set a seed for the random number generator. If None, a seed will be chosen randomly. 

cmap_name = 'inferno'                #Which colormap do you want to use for your images? Use "any" to pick one at random, 'custom' to use a custom one from the block below, or pick one from this list: https://matplotlib.org/stable/tutorials/colors/colormaps.html
output_directory = 'D:/Google Drive/Python Projects/Paint Pouring/Pictures/temp/'

# Some parameters that are specific for producing images with the ever-popular "cells in the foreground" style.

########################################
pptools.generate_paint_pour_images(
    image_dimensions=image_dimensions,
    num_images=num_images,
    display_final_image=display_final_image,
    save_image=save_image,
    show_intermediate_plots=show_intermediate_plots,
    cmap_name=cmap_name,
    output_directory=output_directory,
    seed=seed,
    prominent_cells=True
)