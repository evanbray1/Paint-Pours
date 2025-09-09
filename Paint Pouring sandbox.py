import paint_pour_tools as pptools
from matplotlib import use
import matplotlib.pyplot as plt
# Depending on which IDE you're using, you might need to change this line, or comment it out entirely.
use('QtAgg')
# Show the plots in the iPython console
# %matplotlib inline

# Test comment for test branch

# --- HOW TO USE THIS SANDBOX ---
# 1. Adjust the parameters below to experiment with different paint pour styles.
# 2. For each image, a new PaintPour object is created and .generate() is called.
# 3. All results, including colormaps and noise maps, are accessible as attributes of the PaintPour object.
# 4. Images are saved and/or displayed according to your settings.
# 5. For advanced usage, pass additional arguments to PaintPour or modify the class in paint_pour_tools.py.
#
# Example: To access the noise map for analysis, use: paint_pour.noise_map
# Example: To change the colormap, set custom_cmap_colors or cmap_name.

# ######## USER-DEFINED VARIABLES #########
image_dimensions = [800, 800]
display_final_image = True        # Display the image on the screen?
save_image = True                # Save a .png copy of your image?
num_images = 1                    # How many images to produce?
show_intermediate_plots = True  # Show intermediate results for troubleshooting?
# Set a seed for reproducibility. If None, a seed is chosen randomly.
seed = None
cmap_name = 'any'              # Colormap: 'any', 'custom', or a matplotlib colormap name
output_directory = './outputs/'


##########################################
for i in range(num_images):
    plt.close('all')  # Close all existing plots before starting a new image
    # In VSCode, a short pause ensures plot windows close properly
    plt.pause(0.1)
    print(f'Currently making image {i + 1} of {num_images}')

    # Create a PaintPour class object with your chosen parameters
    paint_pour = pptools.PaintPour(
        image_dimensions=image_dimensions,
        display_final_image=display_final_image,
        save_image=save_image,
        show_intermediate_plots=show_intermediate_plots,
        cmap_name=cmap_name,
        output_directory=output_directory,
        seed=seed,
    )

    # Generate the paint pour
    paint_pour_image = paint_pour.generate()
    
