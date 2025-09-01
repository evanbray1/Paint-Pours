import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
import os
import time
import warnings
import csv


class PaintPour:
    def __init__(self,
                 image_dimensions,
                 display_final_image=True,
                 save_image=True,
                 save_metadata=True,
                 show_intermediate_plots=False,
                 add_cells=False,
                 cmap_name='any',
                 custom_cmap_colors=None,
                 output_directory=None,
                 seed=None,
                 octave_powers=None,
                 stretch_value=None,
                 rescaling_exponent=None,
                 num_colormap_levels=None,
                 prominent_cells=False,
                 **kwargs):
        self.image_dimensions = image_dimensions
        self.display_final_image = display_final_image
        self.save_image = save_image
        self.show_intermediate_plots = show_intermediate_plots
        self.add_cells = add_cells
        self.cmap_name = cmap_name
        self.custom_cmap_colors = custom_cmap_colors
        self.output_directory = output_directory
        self.seed = seed
        self.octave_powers = octave_powers
        self.stretch_value = stretch_value
        self.rescaling_exponent = rescaling_exponent
        self.num_colormap_levels = num_colormap_levels
        self.prominent_cells = prominent_cells
        self.save_metadata = save_metadata
        self.kwargs = kwargs

        # Output attributes
        self.base_colormap = None
        self.final_colormap = None
        self.noise_map = None
        self.noise_octaves = None  # 3D array: each 2D slice is one octave
        self.paint_pour_surface = None

        # Intelligently pick some basic parameters if the user didn't specify them
        self.assign_unspecified_parameters()

    def generate(self):
        start_time = time.time()  # Start a timer for performance-tracking purposes
        np.random.seed(self.seed)  # Set the seed for the random number generator

        # Fractal noise and octave slices
        # Generate noise and octaves
        noise_map_tuple, vector_info = fractal_noise(
            image_dimensions=self.image_dimensions, 
            relative_powers=self.octave_powers, 
            stretch=self.stretch_value,
            show_fractal_noise_plot=self.show_intermediate_plots,
            show_perlin_noise_plots=self.show_intermediate_plots,
            return_octaves=True)
        self.noise_map = noise_map_tuple[0]
        self.noise_octaves = noise_map_tuple[1]

        # Apply some log-scaling to the noise map
        self.paint_pour_surface = log_rescaler(input_values=self.noise_map, 
                                               exponent=self.rescaling_exponent, 
                                               show_plot=self.show_intermediate_plots)

        # Generate a colormap for this image
        self.pick_paint_pour_colormap()

        # Add cells if needed
        if self.add_cells and not self.prominent_cells:
            warnings.warn('WARNING: the add_cells parameter isnt fully implemented yet. Try setting prominent_cells=True instead for now')
        if self.add_cells:
            include_perimeter_regions = np.random.choice([True, False])
            self.gauss_smoothing_sigma = 6
            self.threshold_percentile = 70
            num_voronoi_points = int(np.random.choice([100, 250, 600]) * self.image_dimensions[0] * self.image_dimensions[1] / (1920 * 1080))
            cell_field = make_cell_image(self.image_dimensions, num_voronoi_points=num_voronoi_points, show_plots=self.show_intermediate_plots, gauss_smoothing_sigma=self.gauss_smoothing_sigma,
                                         threshold_percentile=self.threshold_percentile, include_perimeter_regions=include_perimeter_regions)
            ind = np.where(cell_field == 1)
            if self.prominent_cells:
                cell_field_coefficient = np.random.choice([0.3, 1.0])
                self.paint_pour_surface += cell_field_coefficient * cell_field
                self.paint_pour_surface = (self.paint_pour_surface - np.nanmin(self.paint_pour_surface)) / (np.nanmax(self.paint_pour_surface) - np.nanmin(self.paint_pour_surface))
            else:
                self.paint_pour_surface[ind] = 1.01
                self.final_colormap.set_over(self.base_colormap(np.random.uniform(0, 1)))

        # Display and save the image (as well a a separate file of its metadata, if desired)
        if self.display_final_image:
            fig, ax = plt.subplots(1, figsize=(self.image_dimensions[0] / 120, self.image_dimensions[1] / 120))
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            fig.add_axes(ax)
            ax.imshow(self.paint_pour_surface, cmap=self.final_colormap, origin='lower', vmin=0, vmax=1)
            plt.show(block=False)
        if self.save_image:
            self.filename = (self.final_colormap.name + '_' + str(self.num_colormap_levels) + 'levels_' + '_'.join(['{:.2f}'.format(i) for i in self.octave_powers[1:]]) +
                             '_stretch' + str(self.stretch_value) + '_exponent' + '{:.0f}'.format(self.rescaling_exponent))
            if self.add_cells:
                self.filename += '_cellfield_voronoi'
            self.filename += '_seed' + str(self.seed) + '.png'
            if self.output_directory is None:
                self.output_directory = os.path.join('./output_data/')
            if not os.path.exists(self.output_directory):
                os.makedirs(self.output_directory)
            fig.savefig(os.path.join(self.output_directory, self.filename), dpi=120)
            print(f'***Image saved to: {self.output_directory}')

        if self.save_metadata is True:
            print(f'Saving metadata to .csv file')
            metadata_filename = self.filename[:-4] + '.csv'
            self.save_metadata_to_csv(filename=metadata_filename)

        elapsed_time = round(time.time() - start_time, 2)
        print('Elapsed Time: ' + str(elapsed_time) + ' seconds')
        return self.paint_pour_surface

    def pick_paint_pour_colormap(self):
        '''Given the parameters of this PaintPour, pick a colormap to use'''

        if self.cmap_name == 'custom':
            if self.custom_cmap_colors is None:
                self.custom_cmap_colors = ['#33192F', '#803D75', '#CF2808', '#FEE16E', '#6AA886', '#5CE5FB', '#1A1941']
            colors = self.custom_cmap_colors.copy()
            self.base_colormap = make_custom_colormap(colors=colors, show_plot=False)
        else:
            self.base_colormap = plt.cm.get_cmap(self.cmap_name)
        if self.show_intermediate_plots:
            plot_colormap(self.base_colormap, title='Your base colormap, ' + self.base_colormap.name)
        if self.num_colormap_levels == 'continuous':
            self.final_colormap = self.base_colormap.copy()
        else:
            colors = np.random.randint(low=0, high=256, size=self.num_colormap_levels)
            nodes = np.sort(np.random.uniform(low=0, high=1, size=len(colors) - 1))
            self.final_colormap = make_custom_segmented_colormap(colors=self.base_colormap(colors), nodes=[0] + list(nodes) + [1], show_plot=self.show_intermediate_plots, cmap_name=self.base_colormap.name)

    def assign_unspecified_parameters(self):
        '''Go through the paramters for this paint pour and intelligently assign the ones
        that were not explicitly specified'''

        if self.seed is None:
            self.seed = np.random.randint(1, int(1e8))

        if self.prominent_cells:
            self.rescaling_exponent = 1
            self.add_cells = True
            self.num_colormap_levels = 'continuous'

        if self.octave_powers is None:
            self.octave_powers = [1,
                                  np.round(np.random.uniform(0.1, 0.5), 1),
                                  np.round(np.random.uniform(0.0, 0.1), 2),
                                  np.random.choice([0.0, 0.01, 0.02, 0.08], p=[0.55, 0.15, 0.15, 0.15])]

        if self.stretch_value is None:
            self.stretch_value = np.random.randint(-2, 3)

        if self.rescaling_exponent is None:
            self.rescaling_exponent = 10 ** np.random.uniform(0.1, 3)

        if self.cmap_name in ['any', None]:
            self.base_colormap = pick_random_colormap(show_plot=False)
            self.cmap_name = self.base_colormap.name

        if self.num_colormap_levels is None:
            self.num_colormap_levels = np.random.choice([30, 40, 50])

    def save_metadata_to_csv(self, filename):
        """
        Saves all attributes that are NOT 2D/3D arrays to a .csv file with the same filename as the accompanying image.
        """
        def is_2d_or_3d_array(value):
            if isinstance(value, np.ndarray):
                return value.ndim >= 2
            # Check for lists-of-lists (2D or 3D, not handling ragged lists robustly)
            if isinstance(value, list):
                if value and isinstance(value[0], list):
                    # Could also check for 3D
                    if value and isinstance(value, list):
                        return True
                    return True
            return False

        attributes = {}  # Initialize a dictionary that we will fill out with metadata

        # Loop through the class attributes one at a time
        for attr in dir(self):
            if attr.startswith('__') or callable(getattr(self, attr)):  # Ignore the methods and Python's built-in attributes
                continue

            value = getattr(self, attr)  # Grab the attribute
            if is_2d_or_3d_array(value):  # Make sure it's not an image or image cube (we don't want to save that here)
                continue

            # For unrepresentable types, do a safe string conversion
            try:
                value_str = str(value)
            except Exception:
                value_str = '<unrepresentable>'
            attributes[attr] = value_str

        # Define the absolute filepath of the file save location
        absolute_filepath = os.path.join(self.output_directory, filename)
        with open(absolute_filepath, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['attribute', 'value'])
            for attribute_name in sorted(attributes):
                writer.writerow([attribute_name, attributes[attribute_name]])

# Rescale an array from 0-1 with a power law, just like you would do in DS9


def power_rescaler(y, exponent):
    """
    Rescale an array from 0-1 with a power law transformation.

    Parameters
    ----------
    y : np.ndarray
        Array of values to rescale.
    exponent : float
        Exponent for the power law rescaling (must be >= 1).

    Returns
    -------
    y_rescaled : np.ndarray
        Power-law rescaled values.
    """
    if exponent < 1:
        print('WARNING, exponent must be >=1')
        os.sys.exit()
    y_rescaled = (exponent**y - 1) / exponent**y
    y_rescaled -= np.min(y_rescaled[y_rescaled != -np.inf])
    y_rescaled /= y_rescaled.max()
    return y_rescaled  

# Define a function for interpolating between two points, which we do a lot here. This is a convenient function because it doesn't have "kinks" at the endpoints like a linear interpolation function would.
# https://en.wikipedia.org/wiki/Smoothstep
# @njit(parallel=True,fastmath=True)   #Like magic, the @njit bit makes the below function run faster by converting it into machine code.


def smootherstep_function(x):
    """
    Compute the smootherstep interpolation for input x.

    Parameters
    ----------
    x : np.ndarray or float
        Input value(s) for interpolation.

    Returns
    -------
    result : np.ndarray or float
        Interpolated value(s) using the smootherstep formula.
    """
    return 6 * x**5 - 15 * x**4 + 10 * x**3

# You should really read the Wikipedia page on Perlin Noise before trying to dissect this function. (https://en.wikipedia.org/wiki/Perlin_noise#Algorithm_detail)
# Seriously, this part is complicated and took a lot of iterations, linear algebra, and troubleshooting. I would just take it at face value if you're not an expert.
# Maybe watch my earlier "How It's Done" video, too...


def perlin_field(image_dimensions, octave, stretch, make_tileable=False, show_plots=False):
    """
    Generate a Perlin noise field for a given image size and octave.

    Parameters
    ----------
    image_dimensions : list or tuple of int
        The [width, height] of the output image in pixels.
    octave : int
        The octave number (controls grid density).
    stretch : int
        Stretch factor for grid shape (positive: more columns, negative: more rows).
    make_tileable : bool, optional
        If True, makes the noise field tileable (default is False).
    show_plots : bool, optional
        If True, displays the generated Perlin noise image with colorbar and grid lines (default is False).

    Returns
    -------
    image : np.ndarray
        The generated Perlin noise image, normalized to [0, 1].
    vector_info : list
        List containing vector grid coordinates and directions used for noise generation.
    """

    # Break the image field up into a grid of NxM (width x height) vertices. Where N,M are defined by the current "octave" and "stretch" parameters. The squares between these vertices shall be referred to as "cells". 
    # The stretch parameter will add more cells in the x or y direction, depending on whether it is positive or negative and it cause the appearnce to be "stretched" like it was rendered for a different resolution and then stretched to fit the desired one.
    # Stretch must be an integer and adds either columns (positive) or rows (negative).
    # The 0th octave will ALWAYS be a 2x2 grid.
    if octave == 0:
        grid_dimensions = [2, 2]
    elif stretch < 0:
        grid_dimensions = [abs(stretch) * 2**(octave), 2**(octave)]
    elif stretch > 0:
        grid_dimensions = [2**(octave), stretch * 2**(octave)]
    else:
        grid_dimensions = [2**(octave), 2**(octave)]

    # print('Dimensions of vector grid: ',grid_dimensions)
    # Build some arrays that store information about the gradient vectors at each grid node (or "vertices")
    if octave == 0:  # If we're doing the 0th octave, the vectors get positioned OUTSIDE the corners of the image. Since the 1st octave still produces too hilly of a surface unto itself.
        vector_coords_x = np.linspace(-0.5 * image_dimensions[0], 1.5 * image_dimensions[0], grid_dimensions[0], dtype=np.float32)  # X coordinates of vertices (Note the units here are not in pixels, but in "# of vertices from the left". I call this the "cell coordinate system").
        vector_coords_y = np.linspace(-0.5 * image_dimensions[1], 1.5 * image_dimensions[1], grid_dimensions[0], dtype=np.float32)  # Y coordinates of vertices (Note the units here are not in pixels, but in "# of vertices from the bottom")
    else:  # If we're doing any other order, the vectors are positioned so the upper leftmost/rightmost one is at the upper left/right pixel of the image, and so on.
        vector_coords_x = np.linspace(0, image_dimensions[0], grid_dimensions[0], dtype=np.float32)  # X coordinates of vertices (Note the units here are not in pixels, but in "# of vertices from the left". I call this the "cell coordinate system").
        vector_coords_y = np.linspace(0, image_dimensions[1], grid_dimensions[1], dtype=np.float32)  # Y coordinates of vertices (Note the units here are not in pixels, but in "# of vertices from the bottom")

    # What DIRECTION is each vector pointing?
    vector_dir_x = (np.random.random((grid_dimensions[1], grid_dimensions[0])) * 2 - 1).astype('float32')  # What direction is the vector pointing at this vertex? (x-coordinate)
    vector_dir_y = (np.random.random((grid_dimensions[1], grid_dimensions[0])) * 2 - 1).astype('float32')  # What direction is the vector pointing at this vertex? (y-coordinate)

    # Normalize the vector at each grid node by finding the magnitude and dividing each component by that value.
    vector_mag = np.sqrt(vector_dir_x**2 + vector_dir_y**2)
    vector_dir_x = vector_dir_x / vector_mag
    vector_dir_y = vector_dir_y / vector_mag

    # Make the vectors "loop". i.e. the vectors at the right edge = the vectors at the left edge, and so on.
    # Helps for Perlin flow fields. Not necessary for paint pours. 
    if make_tileable:
        vector_dir_x[-1, :] = vector_dir_x[0, :]
        vector_dir_x[:, -1] = vector_dir_x[:, 0]
        vector_dir_y[-1, :] = vector_dir_y[0, :]
        vector_dir_y[:, -1] = vector_dir_y[:, 0]

    # Create many sets of coordinates that represent the (x,y) coordinates of the points WITHIN each "cell" where we will calculate a dot product with the vectors at the four corners of that cell.
    # Note that these values are in the cell coordinate system and are equal in length to the image dimensions. If your grid is 3 cells wide by 4 cells tall, then the pixel at the top right of the image will have the coordinate (3,4)
    if octave == 0:  # Because I defined the 0th octave differently than the rest, the coordinates are defined slightly differently in this case.
        grid_points_x = np.linspace(0, grid_dimensions[0] - 1, image_dimensions[0] * 2, endpoint=False)
        grid_points_y = np.linspace(0, grid_dimensions[1] - 1, image_dimensions[1] * 2, endpoint=False)   

        # Define some empty arrays that we'll store stuff in, shortly.
        dot_products = np.zeros((image_dimensions[1] * 2, image_dimensions[0] * 2, 4)).astype('float32')  # Where we'll store the results of the dot products of each grid point with each of the four nearest grid nodes.
        weights = np.zeros((image_dimensions[1] * 2, image_dimensions[0] * 2, 4)).astype('float32')  # Where we'll store the weighting values that we use for interpolating results of the [4]x[Image_width]x[Image_height] dot product array
        image = np.zeros((image_dimensions[1] * 2, image_dimensions[0] * 2)).astype('float32')  # Where we'll store the final image, which we fill in one "cell" at a time.

    else:
        grid_points_x = np.linspace(0, grid_dimensions[0] - 1, image_dimensions[0], endpoint=False)
        grid_points_y = np.linspace(0, grid_dimensions[1] - 1, image_dimensions[1], endpoint=False)

        # Define some empty arrays that we'll store stuff in, shortly.
        dot_products = np.zeros((image_dimensions[1], image_dimensions[0], 4)).astype('float32')  # Where we'll store the results of the dot products of each grid point with each of the four nearest grid nodes.
        weights = np.zeros((image_dimensions[1], image_dimensions[0], 4)).astype('float32')  # Where we'll store the weighting values that we use for interpolating results of the [4]x[Image_width]x[Image_height] dot product array
        image = np.zeros((image_dimensions[1], image_dimensions[0])).astype('float32')  # Where we'll store the final image, which we fill in one "cell" at a time.

    # How many individual cells does our grid consist of? For the 0th and 1st octave, the answer is 1.
    num_cells_x = grid_dimensions[0] - 1
    num_cells_y = grid_dimensions[1] - 1

    # Now we're gonna do a bunch of calculations for the pairs of (x,y) coordinate pairs that fall within each cell, one cell at a time.
    for i in range(num_cells_y):
        for j in range(num_cells_x):
            # Locate the grid points whose x,y coordinates place it in the current cell. These are referred to as the "offset vectors" on Wikipedia. As in "offset from the node of interest". A "node" is "one of the cell corners".
            # So if our node of interest is the bottom left corner of a cell, the "offset vector" for a point in the top right corner of the cell will be [~1,~1]
            cell_coords_x = grid_points_x[np.where((grid_points_x >= j) & (grid_points_x < (j + 1)))] - j  # This is an array of grid points with coordinates that fall in cell (i,j). Think of these coordinate points as "position within the current cell". These coordinate values range from [0,1)
            cell_coords_y = grid_points_y[np.where((grid_points_y >= i) & (grid_points_y < (i + 1)))] - i

            a, b = np.meshgrid(cell_coords_x, cell_coords_y)  # turn those cell coordinates into a meshgrid of points.
            coordinates = np.array([a.ravel(), b.ravel()])  # Reshape those big 'ol 2D arrays into something that's easier for humans to work with.

            # Count some things
            num_points_in_cell = len(coordinates[0])  # number of unique points there are in this cell total
            num_points_in_cell_x = len(cell_coords_x)  # Number of unique x-coordinates there are for this cell
            num_points_in_cell_y = len(cell_coords_y)  # Number of unique y-coordinates there are for this cell

            # Calculate the x,y position of this cell within the overall image, because we're going to fill in the "image" variable one chunk at a time.
            x_low = len(np.where(grid_points_x < j)[0])
            x_high = x_low + num_points_in_cell_x
            y_low = len(np.where(grid_points_y < i)[0])
            y_high = y_low + num_points_in_cell_y

            # Dot product each grid point's offset vector with the gradient vector in the bottom left of the cell.
            # This is the first of four dot products that we will calculate for each cell.
            vector_temp = np.array([np.repeat(vector_dir_x[i, j], num_points_in_cell), np.repeat(vector_dir_y[i, j], num_points_in_cell)])  # Generate a 2xN array where the corner vector's components are repeated N times. This is done so we can calculate all the grid points in this cell simultaneously.
            result_temp = np.reshape(np.sum(coordinates * vector_temp, axis=0), (num_points_in_cell_y, num_points_in_cell_x))  # Perform the dot product of each individual grid point in that cell with that gradient vector.
            dot_products[y_low:y_high, x_low:x_high, 2] = result_temp  # write the result to the appropriate region of the dot product result array
            # Calculate the weight of each offset vector proximity from the bottom left corner vector
            weights[y_low:y_high, x_low:x_high, 2] = np.reshape(smootherstep_function(1 - coordinates[0]) * smootherstep_function(1 - coordinates[1]), (num_points_in_cell_y, num_points_in_cell_x))  

            # Then bottom right
            coordinates_temp = np.array([coordinates[0] - 1, coordinates[1]])  # recalculate the relative position of each point in the cell based on the bottom right corner
            vector_temp = np.array([np.repeat(vector_dir_x[i, j + 1], num_points_in_cell), np.repeat(vector_dir_y[i, j + 1], num_points_in_cell)])
            result_temp = np.reshape(np.sum(coordinates_temp * vector_temp, axis=0), (num_points_in_cell_y, num_points_in_cell_x))
            dot_products[y_low:y_high, x_low:x_high, 3] = result_temp
            weights[y_low:y_high, x_low:x_high, 3] = np.reshape(smootherstep_function(coordinates[0]) * smootherstep_function(1 - coordinates[1]), (num_points_in_cell_y, num_points_in_cell_x))

            # Then top right
            coordinates_temp = np.array([coordinates[0] - 1, coordinates[1] - 1])
            vector_temp = np.array([np.repeat(vector_dir_x[i + 1, j + 1], num_points_in_cell), np.repeat(vector_dir_y[i + 1, j + 1], num_points_in_cell)])
            result_temp = np.reshape(np.sum(coordinates_temp * vector_temp, axis=0), (num_points_in_cell_y, num_points_in_cell_x))
            dot_products[y_low:y_high, x_low:x_high, 0] = result_temp
            weights[y_low:y_high, x_low:x_high, 0] = np.reshape(smootherstep_function(coordinates[0]) * smootherstep_function(coordinates[1]), (num_points_in_cell_y, num_points_in_cell_x))

            # Then top left
            coordinates_temp = np.array([coordinates[0], coordinates[1] - 1])
            vector_temp = np.array([np.repeat(vector_dir_x[i + 1, j], num_points_in_cell), np.repeat(vector_dir_y[i + 1, j], num_points_in_cell)])
            result_temp = np.reshape(np.sum(coordinates_temp * vector_temp, axis=0), (num_points_in_cell_y, num_points_in_cell_x))
            dot_products[y_low:y_high, x_low:x_high, 1] = result_temp
            weights[y_low:y_high, x_low:x_high, 1] = np.reshape(smootherstep_function(1 - coordinates[0]) * smootherstep_function(coordinates[1]), (num_points_in_cell_y, num_points_in_cell_x))

    # Calculate the Perlin noise image by calculating the weighted average of all the individual slices of the dot product array
    image = weights[:, :, 0] * dot_products[:, :, 0] + weights[:, :, 1] * dot_products[:, :, 1] + weights[:, :, 2] * dot_products[:, :, 2] + weights[:, :, 3] * dot_products[:, :, 3]
    # image = np.sum(dot_products,axis=2) # Uncomment this line to show what the noise field looks like without any interpolation. It will look very blocky and illustrate the "cell" structure. 

    # For the 0th octave case, we have to trim the center [image_dimensions] pixels out, because the 0th octave noise is twice as large in both the x- and y-directions.
    if octave == 0:
        image = image[int(image_dimensions[1] / 2):int(image_dimensions[1] + image_dimensions[1] / 2), int(image_dimensions[0] / 2):int(image_dimensions[0] + image_dimensions[0] / 2)]

    # Normalize the image to the range [0,1]
    image -= np.min(image)
    if np.max(image) > 0:
        image /= np.max(image)

    if show_plots:
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(image, origin='lower')
        ax.set_title(f'Perlin Noise - Octave {octave}')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax)
        if octave > 0:
            # Draw a grid of lines to indicate the cell boundaries for this octave
            for x in vector_coords_x:
                ax.axvline(x=x, color='black', linewidth=1, alpha=0.5)
            for y in vector_coords_y:
                ax.axhline(y=y, color='black', linewidth=1, alpha=0.5)
        ax.set_xlim(0, image_dimensions[0])
        ax.set_ylim(0, image_dimensions[1])

        fig.tight_layout()
        plt.show(block=False)

    # Save some handy diagnostic info about the random vectors used to create this particular Perlin noise image
    vector_info = [vector_coords_x, vector_coords_y, vector_dir_x, vector_dir_y]

    return image.astype('float32'), vector_info


def fractal_noise(image_dimensions, relative_powers, stretch, show_perlin_noise_plots=False, show_fractal_noise_plot=False, return_octaves=False):
    """
    Generate a fractal noise image by summing multiple Perlin noise octaves.

    This function works by calling the Perlin_field function multiple times, each time with a different octave, and adding the result to a master image array
    Typically each octave gets multiplied by 1/(2^octave) before adding it to the master image array. 
    However, here we replace 1/(2^octave) with "relative_power" to allow the user to tweak the intensity of different spatial scales.

    Parameters
    ----------
    image_dimensions : list or tuple of int
        The [width, height] of the output image in pixels.
    relative_powers : list of float
        Relative strengths of each Perlin noise octave.
    stretch : int
        Stretch factor for grid shape (positive: more columns, negative: more rows).
    show_perlin_noise_plots : bool, optional
        If True, display Perlin noise plots for each octave (default is False).
    show_fractal_noise_plot : bool, optional
        If True, display the final fractal noise image (default is False).

    Returns
    -------
    image : np.ndarray
        The generated fractal noise image, normalized to [0, 1].
    octave_cube : np.ndarray (optional)
        An image cube containing each octave that makes up the final returned image.
    vectors : list
        List containing vector grid coordinates and vector directions from the last octave.
    """
    num_octaves = len(relative_powers)
    image = np.zeros((image_dimensions[1], image_dimensions[0]))
    vectors = None
    octave_images = []
    for i in range(num_octaves):
        print(f'\t Making octave {i} of {num_octaves}')
        if relative_powers[i] > 0:
            perlin_image, vectors = perlin_field(image_dimensions, i, stretch, show_plots=show_perlin_noise_plots)
            octave_to_add = relative_powers[i] * perlin_image
            image += octave_to_add
            octave_images.append(octave_to_add)
        else:
            octave_images.append(np.zeros_like(image))
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    if show_fractal_noise_plot:
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(image, origin='lower')
        ax.set_title(f'Fractal noise base image\nRelative Powers={np.array(relative_powers)}, Stretch={stretch}')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax)
        fig.tight_layout()
        plt.show(block=False)
    if return_octaves:
        octave_cube = np.stack(octave_images, axis=0)
        return (image.astype('float32'), octave_cube), vectors
    return image.astype('float32'), vectors


def _log_rescale_helper(input_values, exponent):
    """
    Helper function to rescale input values using a logarithmic transformation.

    Parameters
    ----------
    input_values : np.ndarray
        Array of values to rescale.
    exponent : float
        Exponent for the log rescaling.

    Returns
    -------
    rescaled_values : np.ndarray
        Logarithmically rescaled values.
    """
    if exponent != 1:
        rescaled_values = np.log10(exponent * input_values + 1) / np.log10(exponent)
        rescaled_values -= np.min(rescaled_values[rescaled_values != -np.inf])
        rescaled_values /= rescaled_values.max()
        if (exponent < 1) and (exponent > 0):
            rescaled_values = abs(rescaled_values - 1)
    else:
        rescaled_values = input_values.copy()
    return rescaled_values


def log_rescaler(input_values, exponent, show_plot=False):
    """
    Rescale input values using a logarithmic transformation with the given exponent.
    Optionally displays a plot of rescaled values vs input values.

    Parameters
    ----------
    input_values : np.ndarray
        Array of values to rescale.
    exponent : float
        Exponent for the log rescaling.
    show_plot : bool, optional
        If True, show a plot of input vs rescaled values (default is False).

    Returns
    -------
    rescaled_values : np.ndarray
        Logarithmically rescaled values.
    """
    rescaled_values = _log_rescale_helper(input_values, exponent)
    if (show_plot is True) and (exponent != 1):  # No point in showing this plot if the exponent is 1 since it will just be a straight line
        _x = np.linspace(0, 1, 100)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(_x, _log_rescale_helper(_x, exponent))
        ax.set_xlabel('Input Values')
        ax.set_ylabel('Rescaled Values')
        ax.set_title(f'Log Rescaler (exponent={exponent:.1f})')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        fig.tight_layout()
        plt.show(block=False)
    return rescaled_values


def make_voronoi(npoints, width, height):
    """
    Generate a Voronoi diagram from random points within a given width and height.

    Parameters
    ----------
    npoints : int
        Number of random points to generate.
    width : int
        Width of the region.
    height : int
        Height of the region.

    Returns
    -------
    vor : Voronoi
        Voronoi diagram object.
    """
    print('...Making Voronoi object')
    x = np.random.uniform(0, width, npoints)
    y = np.random.uniform(0, height, npoints)
    points = np.array(list(zip(x, y)))
    return Voronoi(points)


def voronoi_to_points(voronoi, spacing):
    """
    Convert Voronoi diagram ridges to a series of (x, y) points with specified spacing.

    Parameters
    ----------
    voronoi : Voronoi
        Voronoi diagram object.
    spacing : float
        Spacing between points along each ridge.

    Returns
    -------
    x : np.ndarray
        Array of x-coordinates of points.
    y : np.ndarray
        Array of y-coordinates of points.
    """
    print('...Converting Voronoi object to discrete points')
    # Converts the lines of a Voronoi object to a series of (x,y) points with a spacing = "spacing".
    ridge_points = np.empty((0, 2), float)
    for i, ridge in enumerate(voronoi.ridge_vertices):
        if -1 not in ridge:
            x0, y0 = voronoi.vertices[ridge[0]]
            x1, y1 = voronoi.vertices[ridge[1]]
            dx = x1 - x0
            dy = y1 - y0
            length = np.sqrt(dx ** 2 + dy ** 2)
            # print('Ridge length is: ',round(length,3))
            N = int(length / spacing)
            # print('Number of nodes: ',N)
            if N > 0:
                t = np.linspace(0, 1, N + 1)
                x = x0 * (1 - t) + x1 * t
                y = y0 * (1 - t) + y1 * t
                ridge_points = np.vstack([ridge_points, np.array([x, y]).T])

    # Trim down the arrays to only include the unique points (some endpoints get counted twice)
    # print('Number of points (pre-trimming)',len(ridge_points[:,0]))
    _, unique_indices = np.unique(ridge_points[:, 0], return_index=True)
    ridge_points = ridge_points[unique_indices]
    # print('Number of points (post-trimming)',len(ridge_points[:,0]))

    return ridge_points[:, 0], ridge_points[:, 1]


def remove_perimeter_regions(thresholded_image):
    """
    Remove regions in a thresholded image that fall partially outside the image border.

    Parameters
    ----------
    thresholded_image : np.ndarray
        Binary image with thresholded regions.

    Returns
    -------
    trimmed_image : np.ndarray
        Image with border regions removed.
    """
    print('...Removing cells that fall partially outside image')
    image_with_border = cv2.copyMakeBorder(thresholded_image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=1)
    flooded_image = cv2.floodFill(image_with_border, None, (0, 0), 0)[1]
    h, w = flooded_image.shape
    trimmed_image = flooded_image[1:h - 1, 1:w - 1]
    return trimmed_image


def get_contour_pixel_areas(image, list_of_contours):
    """
    Calculate the area (in pixels) of each contour in an image.

    Parameters
    ----------
    image : np.ndarray
        Binary image.
    list_of_contours : list
        List of contour arrays.

    Returns
    -------
    areas : np.ndarray
        Array of pixel areas for each contour.
    """
    print('...Calculating area of cells')
    # Determine the areas of the various closed contours in an image
    height, width = image.shape
    areas = []
    for contour in list_of_contours:
        temp_image = np.zeros(image.shape, dtype=np.uint8)
        temp_image = cv2.drawContours(temp_image, [contour], contourIdx=-1, color=1, thickness=-1)
        contour_area = np.count_nonzero(temp_image)
        areas.append(contour_area)
    return np.array(areas)


def remove_small_regions(image, size_threshold):
    """
    Remove regions in a binary image smaller than a given pixel area threshold.

    Parameters
    ----------
    image : np.ndarray
        Binary image.
    size_threshold : int
        Minimum area (in pixels) for regions to keep.

    Returns
    -------
    final_image : np.ndarray
        Image with small regions removed.
    """
    print('...Removing cells smaller than ' + str(int(size_threshold)) + ' pixels in area')
    contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour_areas = get_contour_pixel_areas(image, contours)    # Calculate the area in pixels of each contour in the image
    small_regions = np.where(contour_areas < size_threshold)[0]  # Identify the contours with an area smaller than the threshold

    # Generate a new image comprised of just the small regions, and then XOR it with the original image to remove those regions.
    temp_image = np.zeros(image.shape, dtype=np.uint8)
    for i, region in enumerate(small_regions):
        temp_image = cv2.drawContours(temp_image, contours, contourIdx=region, color=1, thickness=-1)
    final_image = cv2.bitwise_xor(temp_image, image)  
    return final_image


def pick_random_colormap(print_choice=False, show_plot=False):
    """
    Pick a random colormap from matplotlib, avoiding a list of colormaps that the author finds visually unappealing.

    Parameters
    ----------
    print_choice : bool, optional
        If True, print the chosen colormap name (default is False).
    show_plot : bool, optional
        If True, display the chosen colormap (default is False).

    Returns
    -------
    cmap : Colormap
        The randomly chosen matplotlib colormap.
    """
    print('...Picking random colormap')
    # Some colormaps are just bad looking for this kind of art. I list them here so they will be avoided during the random-picking process.
    bad_cmaps = ['flag', 'Accent', 'gist_stern', 'Paired', 'Dark2', 'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20c', 'tab20b', 'binary', 'Pastel1', 'Pastel2', 'gist_yarg', 'gist_gray', 'brg', 'CMRmap', 'gist_ncar', 'gist_rainbow', 'hsv', 'terrain', 'gnuplot2', 'nipy_spectral', 'prism']
    non_reversed_colormaps = [x for x in plt.colormaps() if '_r' not in x]  # Generate a list of all colormaps that don't contain "_r" in their name, indicating they are just a reversed version of another colormap. "Grays" and "Grays_r" look fundamentally the same for this type of art.

    # Pick a random colormap
    cmap = plt.cm.get_cmap(np.random.choice(non_reversed_colormaps))

    # Re-pick the colormap if it randomly chose a Certified Ugly (TM) one the first time. Keep picking new colormaps until it picks one that isn't ugly.
    while any(cmap.name in s for s in bad_cmaps):
        # print('Randomly chose an ugly colormap! Choosing again...')
        cmap = plt.cm.get_cmap(np.random.choice(non_reversed_colormaps))
    if print_choice == True:
        print('...Chosen colormap: ', cmap.name)
    if show_plot == True:
        plot_colormap(cmap, title=f'Your randomly-chosen base colormap, {cmap.name}')
    return cmap


def plot_colormap(cmap, title='Your colormap'):
    """
    Display a matplotlib colormap as a horizontal colorbar.

    Parameters
    ----------
    cmap : Colormap
        Matplotlib colormap to display.
    title : str, optional
        Title for the plot (default is 'Your colormap').

    Returns
    -------
    The figure and axis objects of the colormap plot.
    """
    fig_cmap, ax_cmap = plt.subplots(figsize=(12, 2))
    ax_cmap.imshow(np.outer(np.ones(100), np.arange(0, 1, 0.001)), cmap=cmap, origin='lower', extent=[0, 1, 0, 0.1])
    ax_cmap.set_title(title)
    ax_cmap.set_yticks([])
    fig_cmap.tight_layout()
    return fig_cmap, ax_cmap


def make_custom_colormap(colors=None, nodes=None, show_plot=False):
    """
    Create a custom continuous colormap from a list of colors and nodes.

    Parameters
    ----------
    colors : list, optional
        List of hex codes or RGB tuples for colormap colors (default is a preset list).
    nodes : np.ndarray, optional
        Array of values between 0 and 1 indicating color positions (default is evenly spaced).
        The first and last value must be 0 and 1, respectively.
        For example, if nodes = [0,0.5,1], your colormap will start at color[0], hit color[1] at the middle value, and reach color[2] at the max value
    show_plot : bool, optional
        If True, display the colormap (default is False).

    Returns
    -------
    cmap_custom : LinearSegmentedColormap
        The generated custom colormap.
    """
    print('...Making a custom continuous colormap')
    if colors is None:
        print('WARNING: No input colors specified. Picking some default values....')
        colors = ['#33192F', '#803D75', '#CF2808', '#FEE16E', '#6AA886', '#5CE5FB', '#1A1941']
    if nodes is None:
        print('WARNING: No input nodes specified. Picking an evenly-spaced array....')
        nodes = np.linspace(0, 1.0, len(colors))  
    cmap_custom = LinearSegmentedColormap.from_list('custom', list(zip(nodes, colors)))

    if show_plot == True:
        fig_cmap, ax_cmap = plt.subplots(figsize=(12, 2))
        ax_cmap.imshow(np.outer(np.ones(100), np.arange(0, 1, 0.001)), cmap=cmap_custom, origin='lower', extent=[0, 1, 0, 0.1])
        ax_cmap.set_title('Your custom colormap')
        ax_cmap.set_xticks(ticks=np.array(nodes))
        ax_cmap.set_xticklabels(['{:.2f}'.format(node) for node in nodes], rotation=60)
        ax_cmap.set_yticks([])
        fig_cmap.tight_layout()
    return cmap_custom


def make_custom_segmented_colormap(colors=None, nodes=None, show_plot=False, cmap_name='custom'):
    """
    Create a custom segmented colormap from a list of colors and nodes.

    Parameters
    ----------
    colors : list, optional
        List of RGBA tuples for colormap colors (default is a preset list).
    nodes : np.ndarray, optional
        Array of values between 0 and 1 indicating color positions (default is evenly spaced).
    show_plot : bool, optional
        If True, display the colormap (default is False).
    cmap_name : str, optional
        Name for the colormap (default is 'custom').

    Returns
    -------
    cmap_custom : LinearSegmentedColormap
        The generated custom segmented colormap.
    """
    print('...Making a custom segmented colormap')
    # Colors = a list of tuples for colors you want your colormap to be composed of, in RGBA format.
    # Nodes = a numpy array of values between 0 and 1 that indicate which "position" of the colormap you want each color to be tied to
    #       -The first and last value must be 0 and 1, respectively.
    #       -For example, if nodes = [0, 0.4, 0.8, 1], the first 40% of your colormap will be color[0], the next 40% will be color[1], and the final 20% will be color[2]
    if colors is None:
        print('WARNING: No input colors specified. Picking some default values....')
        colors = [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)]
    if nodes is None:
        print('WARNING: No input nodes specified. Picking an evenly-spaced array....')
        nodes = np.linspace(0, 1.0, len(colors) + 1)  

    # Because we're making a segmented colormap, we must duplicate each color in the colors array, as well as the inner noes of the nodes array.
    # This is because we must specify the color and value at each segment boundary
    # For example, if nodes = [0, 0.4, 0.8, 1], then nodes_new = [0, 0.4, 0.4, 0.8, 0.8, 1]. 
    colors = [tuple(color[:-1]) for color in colors]
    colors_new = []
    nodes_new = []
    for i in range(len(colors)):
        colors_new.extend([colors[i], colors[i]])
        # colors_new.append(colors[i])
    for i in range(len(nodes)):
        nodes_new.extend([nodes[i], nodes[i]])
        # nodes_new.append(nodes[i])
    nodes_new = nodes_new[1:-1]

    cmap_custom = LinearSegmentedColormap.from_list('custom', list(zip(nodes_new, colors_new)))
    cmap_custom.name = cmap_name

    if show_plot == True:
        fig_cmap, ax_cmap = plt.subplots(figsize=(12, 2))
        ax_cmap.imshow(np.outer(np.ones(100), np.arange(0, 1, 0.001)), cmap=cmap_custom, origin='lower', extent=[0, 1, 0, 0.1])
        ax_cmap.set_title('Your custom segmented colormap')
        ax_cmap.set_xticks(ticks=np.array(nodes))
        ax_cmap.set_xticklabels(['{:.2f}'.format(node) for node in nodes], rotation=60)
        ax_cmap.set_yticks([])
        fig_cmap.tight_layout()
    return cmap_custom


def make_cell_image(image_dimensions, num_voronoi_points=400, gauss_smoothing_sigma=6, threshold_percentile=70, minimum_cell_area=25,
                    show_plots=False, include_perimeter_regions=False):
    """
    Produce a thresholded cell image using Voronoi diagrams and Gaussian smoothing. 
    Note that this function has a number of parameters that (currently) do not scale with image size. Default values are chosen such that they work well for a 1920x1080 image, 
    but deviations from that image size may require some tweaking of the parameters to get a good result.

    Parameters
    ----------
    image_dimensions : list or tuple of int
        [width, height] in pixels for the output image.
    num_voronoi_points : int
        Number of scatterpoints used to generate the Voronoi diagram. Higher number = more cells, in general.
    gauss_smoothing_sigma : float
        Standard deviation for Gaussian smoothing (in pixels). A larger number means rounder cells. 
    threshold_percentile : float
        Percentile for thresholding the smoothed image. Thickness of the "webbing" between cells. Low number = thicker webbing
    minimum_cell_area : int
        Minimum area (in pixels) for cells to keep. 
    show_plots : bool, optional
        If True, display intermediate plots (default is False).
    include_perimeter_regions : bool, optional
        If True, include regions that fall over the border (default is False).

    Returns
    -------
    image_final : np.ndarray
        The final thresholded cell image.
    """
    print('...Producing a thresholded cell image')
    vor = make_voronoi(num_voronoi_points, *image_dimensions)

    # Convert the voronoi diagram ridges into (x,y) points
    x_new, y_new = voronoi_to_points(vor, 1)

    # Remove points that fall outside of the region of interest
    good_indices = np.where((x_new > 0) & (x_new < image_dimensions[0]) & (y_new > 0) & (y_new < image_dimensions[1]))[0]
    x_new = x_new[good_indices]
    y_new = y_new[good_indices]

    if show_plots == True:
        fig, ax = plt.subplots(figsize=(8, 6))
        fig = voronoi_plot_2d(vor, ax=ax, show_vertices=False)
        ax.set_xlim(0, image_dimensions[0])
        ax.set_ylim(0, image_dimensions[1])
        ax.set_aspect('equal')
        ax.scatter(x_new, y_new, s=20)
        ax.set_title('Original Voronoi diagram')
        fig.tight_layout()

    # Take the (x,y) points and histogram them. 
    image = np.histogram2d(x_new, y_new, bins=image_dimensions, range=[[0, image_dimensions[0]], [0, image_dimensions[1]]])[0].T
    if show_plots == True:
        fig2, ax2 = plt.subplots(2, 3, figsize=(18, 7), sharey=True)
        ax2[0, 0].imshow(image, origin='lower')
        ax2[0, 0].set_title('Histogrammed Voronoi webbing')

    # Apply some gaussian smoothing to the image
    image_proc = np.log10(image + .0001)
    image_proc = gaussian_filter(image_proc, gauss_smoothing_sigma)
    if show_plots == True:
        ax2[0, 1].imshow(image_proc, origin='lower')
        ax2[0, 1].set_title('Gaussian smoothed')
        ax2[0, 1].set_aspect('equal')

    # Threshold the image
    image_threshold = np.zeros(image_proc.shape, dtype=np.uint8)
    image_threshold[image_proc < np.percentile(image_proc, threshold_percentile)] = 1
    if show_plots == True:    
        ax2[0, 2].imshow(image_threshold, origin='lower')
        ax2[0, 2].set_title('Threshold applied')
        ax2[0, 2].set_aspect('equal')

    if include_perimeter_regions == False:
        # Remove the regions that fall over the border
        image_final = remove_perimeter_regions(image_threshold)
        if show_plots == True:    
            ax2[1, 0].imshow(image_final, origin='lower')
            ax2[1, 0].set_title('Perimeter regions removed')
    else:
        image_final = image_threshold.copy()

    # Identify regions that are smaller in size than some threshold
    image_final = remove_small_regions(image_final, minimum_cell_area)
    if show_plots == True:    
        ax2[1, 1].imshow(image_final, origin='lower')
        ax2[1, 1].set_title('Small cells removed')
        fig2.tight_layout()

    # Normalize the final image to the range [0,1]
    image_final = (image_final - np.min(image_final)) / (np.max(image_final) - np.min(image_final))

    return image_final

# Identify the centroids of each cell


def calculate_cell_centroids(thresholded_cell_image):
    """
    Calculate the centroids of each cell in a thresholded cell image.

    Parameters
    ----------
    thresholded_cell_image : np.ndarray
        Binary image of thresholded cells.

    Returns
    -------
    centroids : np.ndarray
        Array of (x, y) centroid coordinates for each cell.
    """
    print('...Calculating cell centroids')

    # Define a grid of (x,y) coordinates to represent the pixel locations in the image. Necessary for making a contour plot later.
    x, y = [np.arange(thresholded_cell_image.shape[1]), np.arange(thresholded_cell_image.shape[0])]
    x, y = np.meshgrid(x, y)

    # Determine the (x,y) of the various closed contours in a thresholded_cell_image
    list_of_contours, hierarchy = cv2.findContours(thresholded_cell_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1:]
    x_centroids = []
    y_centroids = []
    for contour in list_of_contours:
        temp_thresholded_cell_image = np.zeros(thresholded_cell_image.shape, dtype=np.uint8)
        temp_thresholded_cell_image = cv2.drawContours(temp_thresholded_cell_image, [contour], contourIdx=-1, color=1, thickness=-1)
        x_centroid = np.sum(temp_thresholded_cell_image * x) / np.sum(temp_thresholded_cell_image)
        y_centroid = np.sum(temp_thresholded_cell_image * y) / np.sum(temp_thresholded_cell_image)
        # ax.scatter(x_centroid,y_centroid,color='red')
        x_centroids.append(x_centroid)
        y_centroids.append(y_centroid)
    return np.transpose(np.array([x_centroids, y_centroids]))


def remove_cells_outside_circular_region(thresholded_cell_image, center, radius):
    """
    Remove cells in a thresholded cell image that fall outside a defined circular region.

    Parameters
    ----------
    thresholded_cell_image : np.ndarray
        Binary image of thresholded cells.
    center : list or tuple of float
        (x, y) coordinates of the circle center.
    radius : float
        Radius of the circle.

    Returns
    -------
    new_thresholded_cell_image : np.ndarray
        Image with cells outside the circle removed.
    """
    print('...Removing cells that fall outside the defined region')

    # Define a grid of (x,y) coordinates to represent the pixel locations in the image. Necessary for making a contour plot later.
    x, y = [np.arange(thresholded_cell_image.shape[1]), np.arange(thresholded_cell_image.shape[0])]
    x, y = np.meshgrid(x, y)

    # Determine the (x,y) of the various closed contours in a thresholded_cell_image
    list_of_contours, hierarchy = cv2.findContours(thresholded_cell_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1:]
    new_thresholded_cell_image = np.zeros(thresholded_cell_image.shape, dtype=np.uint8)
    # cell_centroids = calculate_cell_centroids(cell_field)
    # x_centroids,y_centroids=[cell_centroids[:,0],cell_centroids[:,1]]
    for contour in list_of_contours:
        temp_thresholded_cell_image = np.zeros(thresholded_cell_image.shape, dtype=np.uint8)
        temp_thresholded_cell_image = cv2.drawContours(temp_thresholded_cell_image, [contour], contourIdx=-1, color=1, thickness=-1)
        x_centroid = np.sum(temp_thresholded_cell_image * x) / np.sum(temp_thresholded_cell_image)
        y_centroid = np.sum(temp_thresholded_cell_image * y) / np.sum(temp_thresholded_cell_image)
        distance_from_circle_center = np.sqrt((x_centroid - center[0])**2 + (y_centroid - center[1])**2)
        if distance_from_circle_center < radius:
            new_thresholded_cell_image += temp_thresholded_cell_image
    return new_thresholded_cell_image
