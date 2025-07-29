import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
from matplotlib import colorbar
from scipy.ndimage import gaussian_filter
from scipy.spatial import Voronoi, voronoi_plot_2d
#import matplotlib.cm as cm
import numpy as np
import time
import os
import cv2
import paint_pour_tools as ppt
# %matplotlib inline
plt.close('all')
start_time = time.time()

#########USER-DEFINED VARIABLES#########
image_dimensions = [1200,800]

display_image = True               #Do you want to display the image on the screen? NOTE, automatically set to false when image_dimensions > [1920,1080], otherwise there's some rendering problems.
save_image = True                 #Do you want to save a .png copy of your image?
num_images = 1                    #How many images do you want to produce?

show_intermediate_plots = False  #Do you want to show some intermediate results to help with troubleshooting?
make_surface_plot = False           #Helpful for diagnostic purposes in case you want to see a low-res surface plot of your image
add_cells = False
display_colormap = True         #Do you want to display your chosen colormap in a separate window?

cmap_name = 'any'#'any'                 #Which colormap do you want to use for your images? Use "any" to pick one at random, 'custom' to use a custom one from the block below, or pick one from this list: https://matplotlib.org/stable/tutorials/colors/colormaps.html
# output_directory = 'Pictures/to print/'   #The relative directory where the output images will be saved
output_directory = 'D:/Google Drive/Python Projects/Paint Pouring/Pictures/temp/'


########################################

ppt.generate_paint_pour_images(
    image_dimensions=image_dimensions,
    num_images=num_images,
    display_image=display_image,
    save_image=save_image,
    show_intermediate_plots=show_intermediate_plots,
    make_surface_plot=make_surface_plot,
    add_cells=add_cells,
    display_colormap=display_colormap,
    cmap_name=cmap_name,
    output_directory=output_directory
)

end_time=time.time()
elapsed_time = round(end_time - start_time,2)   #calculate the amount of time that has elapsed since program start, and print it
print('Elapsed Time: '+str(elapsed_time)+' seconds')