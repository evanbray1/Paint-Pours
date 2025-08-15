import paint_pour_tools as pptools
from matplotlib import use
import matplotlib.pyplot as plt
use('TkAgg')
# Show the plots in the iPython console
# %matplotlib inline

# --- HOW TO USE THIS SANDBOX ---
# 1. Adjust the parameters below to experiment with different paint pour styles.
# 2. For each image, a new PaintPour object is created and .generate() is called.
# 3. All results, including colormaps and noise maps, are accessible as attributes of the PaintPour object.
# 4. Images are saved and/or displayed according to your settings.
# 5. For advanced usage, pass additional arguments to PaintPour or modify the class in paint_pour_tools.py.
#
# Example: To access the noise map for analysis, use: paint_pour.noise_map
# Example: To change the colormap, set custom_cmap_colors or cmap_name.

######### USER-DEFINED VARIABLES #########
image_dimensions = [500, 1000]
display_final_image = True        # Display the image on the screen?
save_image = True                 # Save a .png copy of your image?
num_images = 1                    # How many images to produce?
show_intermediate_plots = False   # Show intermediate results for troubleshooting?
seed = None                       # Set a seed for reproducibility. If None, a seed is chosen randomly.
cmap_name = 'custom'              # Colormap: 'any', 'custom', or a matplotlib colormap name
output_directory = 'D:/Google Drive/Python Projects/Paint Pouring/Pictures/sample_output/'
custom_cmap_colors = ['#dadfdb','#a2544c', '#e4bda2', '#f18c6b', '#5c3c37', '#ce9896', '#7a291c', '#ce3d47']
num_levels = 90
octave_powers = [1, 0.1, 0.0, 0.005]
stretch_value = 4
prominent_cells = False

##########################################
for i in range(num_images):
    plt.close('all') # Close all existing plots before starting a new image
    plt.pause(0.1)  # In VSCode, a short pause ensures plot windows close properly
    print(f'Currently making image {i+1} of {num_images}')

    # Create a PaintPour object with your chosen parameters
    paint_pour = pptools.PaintPour(
        image_dimensions=image_dimensions,
        display_final_image=display_final_image,
        save_image=save_image,
        show_intermediate_plots=show_intermediate_plots,
        cmap_name=cmap_name,
        custom_cmap_colors=custom_cmap_colors,
        num_levels=num_levels,
        octave_powers=octave_powers,
        stretch_value=stretch_value,
        output_directory=output_directory,
        seed=seed,
        prominent_cells=prominent_cells
    )
    # Generate the paint pour image
    paint_pour.generate()

    # Example: Access the noise map or colormaps for further analysis
    # print(paint_pour.noise_map)
    # print(paint_pour.base_colormap)