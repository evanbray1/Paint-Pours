import paint_pour_tools as pptools
from matplotlib import use
import matplotlib.pyplot as plt
use('TkAgg')
# Show the plots in the iPython console
# %matplotlib inline 


#########USER-DEFINED VARIABLES#########
image_dimensions = [500, 1000]

display_final_image = True        #Do you want to display the image on the screen?
save_image = True                 #Do you want to save a .png copy of your image?
num_images = 1                    #How many images do you want to produce?

show_intermediate_plots = False  #Do you want to show some intermediate results to help with troubleshooting?
seed=None                        #Set a seed for the random number generator. If None, a seed will be chosen randomly. 

cmap_name = 'custom'                #Which colormap do you want to use for your images? Use "any" to pick one at random, 'custom' to use a custom one from the block below, or pick one from this list: https://matplotlib.org/stable/tutorials/colors/colormaps.html
output_directory = 'D:/Google Drive/Python Projects/Paint Pouring/Pictures/sample_output/'

########################################
for i in range(num_images):
    plt.close('all') # Close all existing plots before starting a new image
    plt.pause(0.1)  # In VSCode specifically, a short pause is needed to ensure interactive plot windows actually close
    print('Currently making image ', i+1, ' of ', num_images)

    pptools.generate_paint_pour_image(
        image_dimensions=image_dimensions,
        display_final_image=display_final_image,
        save_image=save_image,
        show_intermediate_plots=show_intermediate_plots,
        cmap_name=cmap_name,
        custom_cmap_colors=['#dadfdb','#a2544c', '#e4bda2', '#f18c6b', '#5c3c37', '#ce9896', '#7a291c', '#ce3d47'],
        num_levels=90,
        octave_powers=[1, 0.1, 0.0, 0.005],
        stretch_value=4,
        output_directory=output_directory,
        seed=seed,
        prominent_cells=False
    )