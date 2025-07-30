import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
from matplotlib import colorbar, use
from scipy.ndimage import gaussian_filter
from scipy.spatial import Voronoi, voronoi_plot_2d
#import matplotlib.cm as cm
import numpy as np
import time
import os
import cv2
import paint_pour_tools as pptools
use('TkAgg')
%matplotlib inline
plt.close('all')
start_time = time.time()

#TODO, add a way to overplot cell lines to the perlin noise images

#########USER-DEFINED VARIABLES#########
image_dimensions = [1200,800]

display_final_image = True        #Do you want to display the image on the screen?
save_image = True                 #Do you want to save a .png copy of your image?
num_images = 1                    #How many images do you want to produce?

show_intermediate_plots = True  #Do you want to show some intermediate results to help with troubleshooting?
add_cells = False

cmap_name = 'any'#'any'                 #Which colormap do you want to use for your images? Use "any" to pick one at random, 'custom' to use a custom one from the block below, or pick one from this list: https://matplotlib.org/stable/tutorials/colors/colormaps.html
# output_directory = 'Pictures/to print/'   #The relative directory where the output images will be saved
output_directory = 'D:/Google Drive/Python Projects/Paint Pouring/Pictures/temp/'


########################################

pptools.generate_paint_pour_images(
    image_dimensions=image_dimensions,
    num_images=num_images,
    display_final_image=display_final_image,
    save_image=save_image,
    show_intermediate_plots=show_intermediate_plots,
    add_cells=add_cells,
    cmap_name=cmap_name,
    output_directory=output_directory
)