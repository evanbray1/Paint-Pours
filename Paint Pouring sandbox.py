import paint_pour_tools as pptools
from matplotlib import use
# import numpy as np
# import matplotlib.pyplot as plt

# Depending on which IDE you're using, you might need to change this line, or comment it out entirely.
# Common options: 'QtAgg', 'TkAgg', 'Agg' (for headless)
use('QtAgg')
# Show the plots in the iPython console
# %matplotlib inline

# --- HOW TO USE THIS SANDBOX ---
# 1. Adjust the parameters below to experiment with different paint pour styles.
# 2. The generate_paint_pour_images() function handles creating multiple images efficiently.
# 3. All results, including colormaps and noise maps, are accessible as attributes of the PaintPour objects.
# 4. Images are saved and/or displayed according to your settings.
# 5. For advanced usage, pass additional arguments or modify the class in paint_pour_tools.py.


# ######## USER-DEFINED VARIABLES #########
image_dimensions = [2560, 1440]
display_final_image = False        # Display the image on the screen?
save_image = True                # Save a .png copy of your image?
num_images = 1                   # How many images to produce?
show_intermediate_plots = True  # Show intermediate results for troubleshooting?
seed = None                     # Set a seed for reproducibility. If None, a seed is chosen randomly.
base_cmap_name = 'custom'              # Colormap: 'any', 'custom', or a matplotlib colormap name like 'viridis' or 'bone'
output_directory = './outputs/'


##########################################
print(f'Generating {num_images} paint pour images...')

# Generate all images using a convenient function
results = pptools.generate_paint_pour_images(
    num_images=num_images,
    image_dimensions=image_dimensions,
    display_final_image=display_final_image,
    save_image=save_image,
    show_intermediate_plots=show_intermediate_plots,
    base_cmap_name=base_cmap_name,
    output_directory=output_directory,
    # custom_cmap_colors=['#dadfdb', '#a2544c', '#e4bda2', '#f18c6b', '#5c3c37', '#ce9896', '#7a291c', '#ce3d47'],
    seed=seed,
)

print(f'Successfully generated {len(results)} images!')   
