import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
import os
import time

def generate_paint_pour_images(image_dimensions, num_images=3, display_image=True, save_image=True, show_intermediate_plots=False, make_surface_plot=False, add_cells=False, display_colormap=False, cmap_name='any', output_directory='D:/Google Drive/Python Projects/Paint Pouring/Pictures/temp/'):
    """
    Generate and optionally display or save paint pour images using fractal noise and custom colormaps.

    Args:
        image_dimensions (list): [width, height] in pixels for the output image.
        num_images (int): Number of images to generate.
        display_image (bool): Whether to display the image interactively.
        save_image (bool): Whether to save the image as a PNG file.
        show_intermediate_plots (bool): Show intermediate diagnostic plots.
        make_surface_plot (bool): Show a 3D surface plot of the image.
        add_cells (bool): Overlay Voronoi cell structure on the image.
        display_colormap (bool): Display the chosen colormap.
        cmap_name (str): Colormap to use ('any', 'custom', or a matplotlib colormap name).
        output_directory (str): Directory to save output images.
    """
    start_time = time.time() # Start the timer to measure elapsed time
    # If the image to be displayed is too large (larger than your monitor's resolution, force interactive mode off. Otherwise some strange things can happen.
    if (image_dimensions[0] > 1920) or (image_dimensions[1] > 1080) or not display_image:
        plt.ioff()
    else:
        plt.ion()

    # Generate some arrays to hold the x and y coordinates for the image grid
    # x, y = [np.arange(image_dimensions[0]), np.arange(image_dimensions[1])]
    # x, y = np.meshgrid(x, y)

    for i in range(num_images):
        plt.close('all')
        print('Currently making image ', i+1, ' of ', num_images)
        # Call the new function in the loop
        generate_paint_pour_image(
            image_dimensions=image_dimensions,
            display_image=display_image,
            save_image=save_image,
            show_intermediate_plots=show_intermediate_plots,
            make_surface_plot=make_surface_plot,
            add_cells=add_cells,
            display_colormap=display_colormap,
            cmap_name=cmap_name,
            output_directory=output_directory)

    end_time = time.time()
    elapsed_time = round(end_time - start_time, 2)
    print('Elapsed Time: ' + str(elapsed_time) + ' seconds')

def generate_paint_pour_image(
        image_dimensions,
        display_image=True,
        save_image=True,
        show_intermediate_plots=False,
        make_surface_plot=False,
        add_cells=False,
        display_colormap=False,
        cmap_name='any',
        output_directory='D:/Google Drive/Python Projects/Paint Pouring/Pictures/temp/', 
        seed=None
    ):
    """
    Generate a single paint pour image using fractal noise, custom colormaps, and optional Voronoi cell overlays.

    Args:
    image_dimensions (list): [width, height] in pixels for the output image.
    display_image (bool): Whether to display the image interactively.
    save_image (bool): Whether to save the image as a PNG file.
    show_intermediate_plots (bool): Show intermediate diagnostic plots.
    make_surface_plot (bool): Show a 3D surface plot of the image.
    add_cells (bool): Overlay Voronoi cell structure on the image.
    display_colormap (bool): Display the chosen colormap.
    cmap_name (str): Colormap to use ('any', 'custom', or a matplotlib colormap name).
    output_directory (str): Directory to save output images.

    Returns:
    None
    """
    if seed is None:
        # Generate a random seed if not provided
        # This ensures reproducibility if the same seed is used
        seed = np.random.randint(1, int(1e8))
        np.random.seed(seed)

    octave_powers = [
        1,
        np.round(np.random.uniform(0.1, 0.5), 1),
        np.round(np.random.uniform(0.0, 0.1), 2),
        np.random.choice([0.0, 0.01, 0.02, 0.08], p=[0.55, 0.15, 0.15, 0.15])
    ]
    stretch_value = np.random.randint(-2, 3)
    noise_field, vector_info = fractal_noise(image_dimensions, octave_powers, stretch_value, 
                                             show_fractal_noise_plot=show_intermediate_plots, show_perlin_noise_plots=show_intermediate_plots)
    noise_field = (noise_field - np.min(noise_field)) / (np.max(noise_field) - np.min(noise_field))

    if add_cells:
        include_perimeter_regions = False
        gauss_smoothing_sigma = 6
        threshold_percentile = 70
        cell_field = make_cell_image(
            image_dimensions,
            num_voronoi_points=800,
            gauss_smoothing_sigma=gauss_smoothing_sigma,
            threshold_percentile=threshold_percentile,
            minimum_region_area=20,
            show_plots=show_intermediate_plots,
            include_perimeter_regions=include_perimeter_regions
        )
        area_with_cells_x, area_with_cells_y = [image_dimensions[0] / 2, image_dimensions[1] / 2]
        area_with_cells_radius = 200
        if show_intermediate_plots:
            fig, ax = plt.subplots(1)
            ax.imshow(cell_field, origin='lower')
            area_with_cells = plt.Circle(
                (area_with_cells_x, area_with_cells_y),
                area_with_cells_radius,
                fill=None,
                edgecolor='r',
                linewidth=3
            )
            ax.add_patch(area_with_cells)
            fig.tight_layout()
        cell_field = remove_cells_outside_circular_region(
            cell_field,
            [area_with_cells_x, area_with_cells_y],
            area_with_cells_radius
        )
        if show_intermediate_plots:
            fig, ax = plt.subplots(1)
            ax.imshow(cell_field, origin='lower', vmin=0, vmax=1)
            area_with_cells = plt.Circle(
                (area_with_cells_x, area_with_cells_y),
                area_with_cells_radius,
                fill=None,
                edgecolor='r',
                linewidth=3
            )
            ax.add_patch(area_with_cells)
            fig.tight_layout()
        ind = np.where(cell_field == 1)
        noise_field[ind] = 1.01

    rescaling_exponent = 10 ** np.random.uniform(0.1, 3)
    noise_field = log_rescaler(noise_field, exponent=rescaling_exponent)

    num_levels = np.random.choice([30, 40, 50])

    if cmap_name == 'custom':
        cmap_base = make_custom_colormap(
            colors=['#33192F', '#803D75', '#CF2808', '#FEE16E', '#6AA886', '#5CE5FB', '#1A1941'],
            show_plot=display_colormap
        )
    elif cmap_name == 'any':
        cmap_base = pick_random_colormap(show_plot=False)
    else:
        cmap_base = plt.cm.get_cmap(cmap_name)
    if display_colormap:
        plot_colormap(cmap_base, title='Your specified base colormap, ' + cmap_base.name)

    colors = np.random.randint(low=0, high=256, size=num_levels)
    nodes = np.sort(np.random.uniform(low=0, high=1, size=len(colors) - 1))
    cmap = make_custom_segmented_colormap(
        colors=cmap_base(colors),
        nodes=[0] + list(nodes) + [1],
        show_plot=display_colormap,
        cmap_name=cmap_base.name
    )
    cmap.set_over(cmap_base(np.random.uniform(low=0, high=1)))

    if (image_dimensions[0] > 1920) or (image_dimensions[1] > 1080):
        plt.ioff()
    fig, ax = plt.subplots(1, figsize=(image_dimensions[0] / 120, image_dimensions[1] / 120))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    fig.add_axes(ax)
    ax.imshow(noise_field, cmap=cmap, origin='lower', vmin=0, vmax=1)
    if (image_dimensions[0] > 1920) or (image_dimensions[1] > 1080):
        plt.ion()

    if save_image:
        filename = (
            cmap.name + '_' + str(num_levels) + 'levels_' +
            '_'.join(['{:.2f}'.format(i) for i in octave_powers[1:]]) +
            '_stretch' + str(stretch_value) +
            '_exponent' + '{:.0f}'.format(rescaling_exponent)
        )
        if add_cells:
            filename += '_gausssmooth' + str(gauss_smoothing_sigma) + '_threshold' + str(threshold_percentile)
        filename += '_seed' + str(seed)
        output_directory_temp = output_directory
        if not os.path.exists(output_directory_temp):
            os.makedirs(output_directory_temp)
        fig.savefig(output_directory_temp + filename + '.png', dpi=120)

    if make_surface_plot:
        y, x = np.meshgrid(np.arange(image_dimensions[0]), np.arange(image_dimensions[1]))
        fig1, ax1 = plt.subplots(figsize=(8, 6), subplot_kw={"projection": "3d"})
        surf = ax1.plot_surface(x, y, noise_field, cmap=cmap, linewidth=0, antialiased=False, rcount=50, ccount=50)
        fig1.tight_layout()
        print('\n\n')
        if add_cells:
            print('Gauss smoothing sigma = ', gauss_smoothing_sigma)
            print('Threshold percentile = ', threshold_percentile)
            print('Rescaling exponent =', rescaling_exponent)
    if display_image is True:
        plt.show()

def log_rescaler(y,exponent):
    if exponent != 1:
        y_rescaled = np.log10(exponent*y+1)/np.log10(exponent)
        y_rescaled -= np.min(y_rescaled[y_rescaled != -np.inf])
        y_rescaled /= y_rescaled.max()
        if (exponent < 1) and (exponent > 0):
            y_rescaled = abs(y_rescaled-1)
    else:
        y_rescaled = y.copy()
    return y_rescaled  

#Rescale an array from 0-1 with a power law, just like you would do in DS9
def power_rescaler(y,exponent):
    if exponent < 1:
        print('WARNING, exponent must be >=1')
        os.sys.exit()
    y_rescaled = (exponent**y-1)/exponent**y
    y_rescaled -= np.min(y_rescaled[y_rescaled != -np.inf])
    y_rescaled /= y_rescaled.max()
    return y_rescaled  

#Define a function for interpolating between two points, which we do a lot here. This is a convenient one because it doesn't have "kinks" at the endpoints like a linear interpolation function would.
#https://en.wikipedia.org/wiki/Smoothstep
# @njit(parallel=True,fastmath=True)   #Like magic, the @njit bit makes the below function run faster by converting it into machine code.
def smootherstep_function(x):
    return 6*x**5-15*x**4+10*x**3

#You should really read the Wikipedia page on Perlin Noise before trying to dissect this function. (https://en.wikipedia.org/wiki/Perlin_noise#Algorithm_detail)
#Seriously, this part is complicated and took a lot of iterations, linear algebra, and troubleshooting. I would just take it at face value.
#Maybe watch my earlier "How It's Done" video, too...
def perlin_field(image_dimensions,octave,stretch,make_tileable=False, show_plots=False):
    
    #Break the image field up into a grid of NxM "cells". Where N,M are defined by the current "octave" and "stretch" parameters.
    #The stretch parameter will add more cells in the x or y direction, depending on whether it is positive or negative and it cause the appearnce to be "stretched" like it was rendered for a different resolution and then stretched to fit the desired one.
    #Stretch must be an integer and adds either columns (positive) or rows (negative).
    #The 0th octave will ALWAYS be a 2x2 grid.
    if octave == 0:
        grid_dimensions = [2,2]
    elif stretch < 0:
        grid_dimensions = [abs(stretch)+2**(octave),2**(octave)]
    elif stretch > 0:
        grid_dimensions = [2**(octave),stretch+2**(octave)]
    else:
        grid_dimensions = [2**(octave),2**(octave)]
        
    #print('Dimensions of vector grid: ',grid_dimensions)
    #Build some arrays that store information about the gradient vectors at each grid node (or "vertices")
    if octave == 0:     #If we're doing the 0th octave, the vectors get positioned OUTSIDE the corners of the image. Since the 1st octave still produces too hilly of a surface unto itself.
        vector_coords_x = np.linspace(-0.5*image_dimensions[0],1.5*image_dimensions[0],grid_dimensions[0],dtype=np.float32)    #X coordinates of vertices (Note the units here are not in pixels, but in "# of vertices from the left". I call this the "cell coordinate system").
        vector_coords_y = np.linspace(-0.5*image_dimensions[1],1.5*image_dimensions[1],grid_dimensions[0],dtype=np.float32)    #Y coordinates of vertices (Note the units here are not in pixels, but in "# of vertices from the bottom")
    else:               #If we're doing any other order, the vectors are positioned so the upper leftmost/rightmost one is at the upper left/right pixel of the image, and so on.
        vector_coords_x = np.linspace(0,image_dimensions[0],grid_dimensions[0],dtype=np.float32)    #X coordinates of vertices (Note the units here are not in pixels, but in "# of vertices from the left". I call this the "cell coordinate system").
        vector_coords_y = np.linspace(0,image_dimensions[1],grid_dimensions[1],dtype=np.float32)    #Y coordinates of vertices (Note the units here are not in pixels, but in "# of vertices from the bottom")
    
    #What DIRECTION is each vector pointing?
    vector_dir_x = (np.random.random((grid_dimensions[1],grid_dimensions[0]))*2-1).astype('float32')    #What direction is the vector pointing at this vertex? (x-coordinate)
    vector_dir_y = (np.random.random((grid_dimensions[1],grid_dimensions[0]))*2-1).astype('float32')    #What direction is the vector pointing at this vertex? (y-coordinate)
    
    #Normalize the vector at each grid node by finding the magnitude and dividing each component by that value.
    vector_mag = np.sqrt(vector_dir_x**2 + vector_dir_y**2)
    vector_dir_x = vector_dir_x/vector_mag
    vector_dir_y = vector_dir_y/vector_mag

    #Make the vectors "loop". i.e. the vectors at the right edge = the vectors at the left edge, and so on.
    #Helps for Perlin flow fields. Not necessary for paint pours. 
    if make_tileable:
        vector_dir_x[-1,:] = vector_dir_x[0,:]
        vector_dir_x[:,-1] = vector_dir_x[:,0]
        vector_dir_y[-1,:] = vector_dir_y[0,:]
        vector_dir_y[:,-1] = vector_dir_y[:,0]
    
    #Create many sets of coordinates that represent the (x,y) coordinates of the points WITHIN each "cell" where we will calculate a dot product with the vectors at the four corners of that cell.
    #Note that these values are in the cell coordinate system and are equal in length to the image dimensions. If your grid is 3 cells wide by 4 cells tall, then the pixel at the top right of the image will have the coordinate (3,4)
    if octave == 0:     #Because I defined the 0th octave differently than the rest, the coordinates are defined slightly differently in this case.
        grid_points_x = np.linspace(0,grid_dimensions[0]-1,image_dimensions[0]*2,endpoint=False)
        grid_points_y = np.linspace(0,grid_dimensions[1]-1,image_dimensions[1]*2,endpoint=False)   
        
        #Define some empty arrays that we'll store stuff in, shortly.
        dot_products = np.zeros((image_dimensions[1]*2,image_dimensions[0]*2,4)).astype('float32')      #Where we'll store the results of the dot products of each grid point with each of the four nearest grid nodes.
        weights = np.zeros((image_dimensions[1]*2,image_dimensions[0]*2,4)).astype('float32')           #Where we'll store the weighting values that we use for interpolating results of the [4]x[Image_width]x[Image_height] dot product array
        image = np.zeros((image_dimensions[1]*2,image_dimensions[0]*2)).astype('float32')               #Where we'll store the final image, which we fill in one "cell" at a time.
        
    else:
        grid_points_x = np.linspace(0,grid_dimensions[0]-1,image_dimensions[0],endpoint=False)
        grid_points_y = np.linspace(0,grid_dimensions[1]-1,image_dimensions[1],endpoint=False)
        
        #Define some empty arrays that we'll store stuff in, shortly.
        dot_products = np.zeros((image_dimensions[1],image_dimensions[0],4)).astype('float32')      #Where we'll store the results of the dot products of each grid point with each of the four nearest grid nodes.
        weights = np.zeros((image_dimensions[1],image_dimensions[0],4)).astype('float32')           #Where we'll store the weighting values that we use for interpolating results of the [4]x[Image_width]x[Image_height] dot product array
        image = np.zeros((image_dimensions[1],image_dimensions[0])).astype('float32')               #Where we'll store the final image, which we fill in one "cell" at a time.

    #How many individual cells does our grid consist of? For the 0th and 1st octave, the answer is 1.
    num_cells_x = grid_dimensions[0]-1
    num_cells_y = grid_dimensions[1]-1
    
    #Now we're gonna do a bunch of calculations for the pairs of (x,y) coordinate pairs that fall within each cell, one cell at a time.
    for i in range(num_cells_y):
        for j in range(num_cells_x):
            #Locate the grid points whose x,y coordinates place it in the current cell. These are referred to as the "offset vectors" on Wikipedia. As in "offset from the node of interest". A "node" is "one of the cell corners".
            #So if our node of interest is the bottom left corner of a cell, the "offset vector" for a point in the top right corner of the cell will be [~1,~1]
            cell_coords_x = grid_points_x[np.where((grid_points_x >= j) & (grid_points_x < (j+1)))]-j       #This is an array of grid points with coordinates that fall in cell (i,j). Think of these coordinate points as "position within the current cell". These coordinate values range from [0,1)
            cell_coords_y = grid_points_y[np.where((grid_points_y >= i) & (grid_points_y < (i+1)))]-i
            
            a,b = np.meshgrid(cell_coords_x,cell_coords_y)  #turn those cell coordinates into a meshgrid of points.
            coordinates = np.array([a.ravel(),b.ravel()])   #Reshape those big 'ol 2D arrays into something that's easier for humans to work with.
            
            #Count some things
            num_points_in_cell = len(coordinates[0])        #number of unique points there are in this cell total
            num_points_in_cell_x = len(cell_coords_x)       #Number of unique x-coordinates there are for this cell
            num_points_in_cell_y = len(cell_coords_y)       #Number of unique y-coordinates there are for this cell
            
            #Calculate the x,y position of this cell within the overall image, because we're going to fill in the "image" variable one chunk at a time.
            x_low = len(np.where(grid_points_x < j)[0])
            x_high = x_low+num_points_in_cell_x
            y_low = len(np.where(grid_points_y < i)[0])
            y_high = y_low+num_points_in_cell_y
            
            #Dot product each grid point's offset vector with the gradient vector in the bottom left of the cell.
            #This is the first of four dot products that we will calculate for each cell.
            vector_temp = np.array([np.repeat(vector_dir_x[i,j],num_points_in_cell),np.repeat(vector_dir_y[i,j],num_points_in_cell)])   #Generate a 2xN array where the corner vector's components are repeated N times. This is done so we can calculate all the grid points in this cell simultaneously.
            result_temp = np.reshape(np.sum(coordinates*vector_temp,axis=0),(num_points_in_cell_y,num_points_in_cell_x))                #Perform the dot product of each individual grid point in that cell with that gradient vector.
            dot_products[y_low:y_high,x_low:x_high,2] = result_temp             #write the result to the appropriate region of the dot product result array
            #Calculate the weight of each offset vector proximity from the bottom left corner vector
            weights[y_low:y_high,x_low:x_high,2] = np.reshape(smootherstep_function(1-coordinates[0])*smootherstep_function(1-coordinates[1]),(num_points_in_cell_y,num_points_in_cell_x))  
            
            #Then bottom right
            coordinates_temp = np.array([coordinates[0]-1,coordinates[1]])      #recalculate the relative position of each point in the cell based on the bottom right corner
            vector_temp = np.array([np.repeat(vector_dir_x[i,j+1],num_points_in_cell),np.repeat(vector_dir_y[i,j+1],num_points_in_cell)])
            result_temp = np.reshape(np.sum(coordinates_temp*vector_temp,axis=0),(num_points_in_cell_y,num_points_in_cell_x))
            dot_products[y_low:y_high,x_low:x_high,3] = result_temp
            weights[y_low:y_high,x_low:x_high,3] = np.reshape(smootherstep_function(coordinates[0])*smootherstep_function(1-coordinates[1]),(num_points_in_cell_y,num_points_in_cell_x))
    
            #Then top right
            coordinates_temp = np.array([coordinates[0]-1,coordinates[1]-1])
            vector_temp = np.array([np.repeat(vector_dir_x[i+1,j+1],num_points_in_cell),np.repeat(vector_dir_y[i+1,j+1],num_points_in_cell)])
            result_temp = np.reshape(np.sum(coordinates_temp*vector_temp,axis=0),(num_points_in_cell_y,num_points_in_cell_x))
            dot_products[y_low:y_high,x_low:x_high,0] = result_temp
            weights[y_low:y_high,x_low:x_high,0] = np.reshape(smootherstep_function(coordinates[0])*smootherstep_function(coordinates[1]),(num_points_in_cell_y,num_points_in_cell_x))
    
            #Then top left
            coordinates_temp = np.array([coordinates[0],coordinates[1]-1])
            vector_temp = np.array([np.repeat(vector_dir_x[i+1,j],num_points_in_cell),np.repeat(vector_dir_y[i+1,j],num_points_in_cell)])
            result_temp = np.reshape(np.sum(coordinates_temp*vector_temp,axis=0),(num_points_in_cell_y,num_points_in_cell_x))
            dot_products[y_low:y_high,x_low:x_high,1] = result_temp
            weights[y_low:y_high,x_low:x_high,1] = np.reshape(smootherstep_function(1-coordinates[0])*smootherstep_function(coordinates[1]),(num_points_in_cell_y,num_points_in_cell_x))
            
    #Calculate the Perlin noise image by calculating the weighted average of all the individual slices of the dot product array
    image = weights[:,:,0]*dot_products[:,:,0]+weights[:,:,1]*dot_products[:,:,1]+weights[:,:,2]*dot_products[:,:,2]+weights[:,:,3]*dot_products[:,:,3]
    
    #For the 0th octave case, we have to trim the center [image_dimensions] pixels out, because the 0th octave noise is twice as large in both the x- and y-directions.
    if octave == 0:
        image = image[int(image_dimensions[1]/2):int(image_dimensions[1]+image_dimensions[1]/2),int(image_dimensions[0]/2):int(image_dimensions[0]+image_dimensions[0]/2)]

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
        fig.tight_layout()
        plt.show()

    #Save some handy diagnostic info about the random vectors used to create this particular Perlin noise image
    vector_info = [vector_coords_x, vector_coords_y, vector_dir_x, vector_dir_y]
    
    return image.astype('float32'), vector_info

#Fractal noise is just multiple layers (octaves) of Perlin noise added on top of one another.
#This function works by calling the Perlin_field function multiple times, each time with a different octave, and adding the result to a master image array
#Typically each octave gets multiplied by 1/(2^octave) before adding it to the master image array. 
#However, here we replace 1/(2^octave) with "relative_power" which gets randomly chosen from an octave-dependent range of values.
def fractal_noise(image_dimensions, relative_powers, stretch, show_perlin_noise_plots=False, show_fractal_noise_plot=False):
    print('...Making fractal noise layer')
    num_octaves = len(relative_powers)
    image = np.zeros((image_dimensions[1], image_dimensions[0]))  # Define an empty array where we'll build the final image
    vectors = None

    # Calculate multiple Perlin noise fields. Each one is twice as dense as the last.
    for i in range(num_octaves):
        if relative_powers[i] > 0:  # No point in expending the computer's time to calculate a Perlin noise field if the relative power is 0.
            perlin_image, vectors = perlin_field(image_dimensions, i, stretch, show_plots=show_perlin_noise_plots)
            image += relative_powers[i] * perlin_image  # Add that Perlin noise field to the total, with geometrically decreasing weighting.

    # Normalize the image to the range [0,1]
    image -= np.min(image)
    if np.max(image) > 0:
        image /= np.max(image)

    if show_fractal_noise_plot:
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(image, origin='lower')
        ax.set_title(f'Fractal Noise\nRelative Powers={np.array(relative_powers)}, Stretch={stretch}')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax)
        fig.tight_layout()
        plt.show()

    return image.astype('float32'), vectors

def make_voronoi(npoints,width,height):
    print('...Making Voronoi object')
    x = np.random.uniform(0,width,npoints)
    y = np.random.uniform(0,height,npoints)
    points = np.array(list(zip(x,y)))
    return Voronoi(points)

def voronoi_to_points(voronoi,spacing):
    print('...Converting Voronoi object to discrete points')
    #Converts the lines of a Voronoi object to a series of (x,y) points with a spacing = "spacing".
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
            if N>0:
                t = np.linspace(0,1,N+1)
                x = x0 * (1 - t) + x1 * t
                y = y0 * (1 - t) + y1 * t
                ridge_points = np.vstack([ridge_points, np.array([x,y]).T])
    
    #Trim down the arrays to only include the unique points (some endpoints get counted twice)
    #print('Number of points (pre-trimming)',len(ridge_points[:,0]))
    _,unique_indices = np.unique(ridge_points[:,0],return_index=True)
    ridge_points = ridge_points[unique_indices]
    #print('Number of points (post-trimming)',len(ridge_points[:,0]))

    return ridge_points[:, 0], ridge_points[:, 1]

def remove_perimeter_regions(thresholded_image):
    print('...Removing cells that fall partially outside image')
    image_with_border = cv2.copyMakeBorder(thresholded_image,1,1,1,1,cv2.BORDER_CONSTANT,value=1)
    flooded_image = cv2.floodFill(image_with_border,None,(0,0),0)[1]
    h,w = flooded_image.shape
    trimmed_image = flooded_image[1:h-1,1:w-1]
    return trimmed_image

def get_contour_pixel_areas(image,list_of_contours):
    print('...Calculating area of cells')
    #Determine the areas of the various closed contours in an image
    height,width = image.shape
    areas = []
    for contour in list_of_contours:
        temp_image = np.zeros(image.shape,dtype=np.uint8)
        temp_image = cv2.drawContours(temp_image,[contour],contourIdx=-1,color=1,thickness=-1)
        contour_area = np.count_nonzero(temp_image)
        areas.append(contour_area)
    return np.array(areas)

def remove_small_regions(image,size_threshold):
    print('...Removing cells smaller than '+str(int(size_threshold))+' pixels in area')
    contours,hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1:]
    contour_areas = get_contour_pixel_areas(image, contours)
    small_regions = np.where(contour_areas < size_threshold)[0]
    # small_regions = small_regions[np.where(contour_areas < size_threshold)[0]]
    
    #Remove regions that are smaller than some threshold
    # if len(small_regions > 0):
    temp_image = np.zeros(image.shape,dtype=np.uint8)
    for i,region in enumerate(small_regions):
        temp_image = cv2.drawContours(temp_image,contours,contourIdx=region,color=1,thickness=-1)
    final_image = cv2.bitwise_xor(temp_image,image)  
    return final_image

def pick_random_colormap(print_choice=False,show_plot=False):
    print('...Picking random colormap')
    #Some colormaps are just bad looking for this kind of art, IMO. I list them here so I can make sure to avoid them during the random-picking process later.
    bad_cmaps = ['flag','Accent','gist_stern','Paired','Dark2','Set1','Set2','Set3','tab10','tab20','tab20c','tab20b','binary','Pastel1','Pastel2','gist_yarg','gist_gray','brg','CMRmap','gist_ncar','gist_rainbow','hsv','terrain','gnuplot2','nipy_spectral','prism']
    non_reversed_colormaps = [x for x in plt.colormaps() if '_r' not in x]      #Generate a list of all colormaps that don't contain "_r" in their name, indicating they are just a reversed version of another colormap. "Grays" and "Grays_r" look fundamentally the same for this type of art.

    #Pick a random colormap
    cmap = plt.cm.get_cmap(np.random.choice(non_reversed_colormaps))

    #Re-pick the colormap if it randomly chose a Certified Ugly (TM) one the first time. Keep picking new colormaps until it picks one that isn't ugly.
    while any(cmap.name in s for s in bad_cmaps):
        # print('Randomly chose an ugly colormap! Choosing again...')
        cmap = plt.cm.get_cmap(np.random.choice(non_reversed_colormaps))
    if print_choice==True:
        print('...Chosen colormap: ',cmap.name)
    if show_plot == True:
        plot_colormap(cmap,title=f'Your randomly-chosen base colormap, {cmap.name}')
    return cmap

def plot_colormap(cmap,title='Your colormap'):
    fig_cmap,ax_cmap = plt.subplots(figsize=(12,2))
    ax_cmap.imshow(np.outer(np.ones(100),np.arange(0,1,0.001)),cmap=cmap,origin='lower',extent=[0,1,0,0.1])
    ax_cmap.set_title(title)
    ax_cmap.set_yticks([])
    fig_cmap.tight_layout()

def make_custom_colormap(colors=None,nodes=None,show_plot=False):
    print('...Making a custom continuous colormap')
    #Colors = a list of hex codes or RGB tuples for colors you want your colormap to be composed of
    #Nodes = a numpy array of values between 0 and 1 that indicate which "position" of the colormap you want each color to be tied to
    #       -The first and last value must be 0 and 1, respectively.
    #       -For example, if nodes = [0,0.5,1], your colormap will start at color[0], hit color[1] at the middle value, and reach color[2] at the max value
    if colors is None:
        print('WARNING: No input colors specified. Picking some default values....')
        colors=['#33192F','#803D75','#CF2808','#FEE16E','#6AA886','#5CE5FB','#1A1941']
    if nodes is None:
        print('WARNING: No input nodes specified. Picking an evenly-spaced array....')
        nodes = np.linspace(0,1.0,len(colors))  
    # nodes = np.concatenate((np.linspace(0,0.6,len(colors)-1),np.array([1.0])))
    cmap_custom = LinearSegmentedColormap.from_list('custom', list(zip(nodes, colors)))
    
    if show_plot == True:
        fig_cmap,ax_cmap = plt.subplots(figsize=(12,2))
        ax_cmap.imshow(np.outer(np.ones(100),np.arange(0,1,0.001)),cmap=cmap_custom,origin='lower',extent=[0,1,0,0.1])
        ax_cmap.set_title('Your custom colormap')
        ax_cmap.set_xticks(ticks=np.array(nodes))
        ax_cmap.set_xticklabels(['{:.2f}'.format(node) for node in nodes],rotation=60)
        ax_cmap.set_yticks([])
        fig_cmap.tight_layout()
    return cmap_custom

def make_custom_segmented_colormap(colors=None,nodes=None,show_plot=False,cmap_name = 'custom'):
    print('...Making a custom segmented colormap')
    #Colors = a list of tuples for colors you want your colormap to be composed of
    #Nodes = a numpy array of values between 0 and 1 that indicate which "position" of the colormap you want each color to be tied to
    #       -The first and last value must be 0 and 1, respectively.
    #       -For example, if nodes = [0,0.5,1], your colormap will start at color[0], hit color[1] at the middle value, and reach color[2] at the max value
    if colors is None:
        print('WARNING: No input colors specified. Picking some default values....')
        colors=[(1,0,0,1),(0,1,0,1),(0,0,1,1)]
    if nodes is None:
        print('WARNING: No input nodes specified. Picking an evenly-spaced array....')
        nodes = np.linspace(0,1.0,len(colors)+1)  
        
    #Because we're making a segmented colormap, we must duplicate each color in the colors array, as well as the inner noes of the nodes array
    colors = [tuple(color[:-1]) for color in colors]
    colors_new = []
    nodes_new = []
    for i in range(len(colors)):
        colors_new.append(colors[i])
        colors_new.append(colors[i])
    for i in range(len(nodes)):
        nodes_new.append(nodes[i])
        nodes_new.append(nodes[i])
    nodes_new = nodes_new[1:-1]

    cmap_custom = LinearSegmentedColormap.from_list('custom', list(zip(nodes_new, colors_new)))
    cmap_custom.name = cmap_name
    
    if show_plot == True:
        fig_cmap,ax_cmap = plt.subplots(figsize=(12,2))
        ax_cmap.imshow(np.outer(np.ones(100),np.arange(0,1,0.001)),cmap=cmap_custom,origin='lower',extent=[0,1,0,0.1])
        ax_cmap.set_title('Your custom segmented colormap')
        ax_cmap.set_xticks(ticks=np.array(nodes))
        ax_cmap.set_xticklabels(['{:.2f}'.format(node) for node in nodes],rotation=60)
        ax_cmap.set_yticks([])
        fig_cmap.tight_layout()
    return cmap_custom

def make_cell_image(image_dimensions,num_voronoi_points,gauss_smoothing_sigma,threshold_percentile,minimum_region_area,
                    show_plots=False,include_perimeter_regions=False):
    print('...Producing a thresholded cell image')
    #num_voronoi_points = number of scatterpoints used to generate the Voronoi diagram. Higher number = more cells, in general
    #gauss_smoothing_sigma = In pixels, how "round" the corners of the cells are
    #threshold_percentile = thickness of the webbing between cells. Low number = thicker webbing
    #minimum_region_area = any cells with an area smaller than this (in pixels) will be removed
    vor = make_voronoi(num_voronoi_points,*image_dimensions)

    # print('Making cell image!')
    #Convert the voronoi diagram ridges into (x,y) points
    x_new,y_new = voronoi_to_points(vor,1)
    # print('Done converting into (x,y) points')
    
    #Remove points that fall outside of the region of interest
    good_indices = np.where((x_new > 0) & (x_new < image_dimensions[0]) & (y_new > 0) & (y_new < image_dimensions[1]))[0]
    x_new = x_new[good_indices]
    y_new = y_new[good_indices]

    if show_plots == True:
        fig,ax = plt.subplots(figsize=(8,6))
        fig = voronoi_plot_2d(vor,ax=ax,show_vertices=False)
        ax.set_xlim(0,image_dimensions[0])
        ax.set_ylim(0,image_dimensions[1])
        ax.set_aspect('equal')
        ax.scatter(x_new,y_new,s=20)
        fig.tight_layout()
    
    #Take the (x,y) points and histogram them.
    image = np.histogram2d(x_new,y_new,bins=image_dimensions,range=[[0,image_dimensions[0]],[0,image_dimensions[1]]])[0].T
    if show_plots == True:
        fig2,ax2 = plt.subplots(1,5,figsize=(18,5),sharey=True)
        ax2[0].imshow(image,origin='lower')
        ax2[0].set_title('Original Voronoi')
    # print('Done histgramming')

    #Apply some gaussian smoothing to the image
    image_proc = np.log10(image+.0001)
    image_proc = gaussian_filter(image_proc,gauss_smoothing_sigma)
    if show_plots == True:
        ax2[1].imshow(image_proc,origin='lower')
        ax2[1].set_title('imagemed + smoothed')
        ax2[1].set_aspect('equal')
    
    #Threshold the image
    image_threshold = np.zeros(image_proc.shape,dtype=np.uint8)
    image_threshold[image_proc < np.percentile(image_proc,threshold_percentile)] = 1
    if show_plots == True:    
        ax2[2].imshow(image_threshold,origin='lower')
        ax2[2].set_title('Threshold applied')
        ax2[2].set_aspect('equal')
    # print('Done thresholding')
    
    if include_perimeter_regions == False:
        #Remove the regions that fall over the border
        image_final = remove_perimeter_regions(image_threshold)
        if show_plots == True:    
            ax2[3].imshow(image_final,origin='lower')
            ax2[3].set_title('Border regions removed')
        # print('Done removing outer regions')
    else:
        image_final = image_threshold.copy()
        
    #Identify regions that are smaller in size than some threshold
    image_final = remove_small_regions(image_final, minimum_region_area)
    if show_plots == True:    
        ax2[4].imshow(image_final,origin='lower')
        ax2[4].set_title('Small regions removed')
        fig2.tight_layout()
    # print('Done removing small regions')

    return image_final

#Identify the centroids of each cell
def calculate_cell_centroids(thresholded_cell_image):
    print('...Calculating cell centroids')
    
    #Define a grid of (x,y) coordinates to represent the pixel locations in the image. Necessary for making a contour plot later.
    x,y = [np.arange(thresholded_cell_image.shape[1]),np.arange(thresholded_cell_image.shape[0])]
    x,y = np.meshgrid(x,y)

    #Determine the (x,y) of the various closed contours in a thresholded_cell_image
    list_of_contours,hierarchy = cv2.findContours(thresholded_cell_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1:]
    x_centroids = []
    y_centroids = []
    for contour in list_of_contours:
        temp_thresholded_cell_image = np.zeros(thresholded_cell_image.shape,dtype=np.uint8)
        temp_thresholded_cell_image = cv2.drawContours(temp_thresholded_cell_image,[contour],contourIdx=-1,color=1,thickness=-1)
        x_centroid = np.sum(temp_thresholded_cell_image*x)/np.sum(temp_thresholded_cell_image)
        y_centroid = np.sum(temp_thresholded_cell_image*y)/np.sum(temp_thresholded_cell_image)
        # ax.scatter(x_centroid,y_centroid,color='red')
        x_centroids.append(x_centroid)
        y_centroids.append(y_centroid)
    return np.transpose(np.array([x_centroids,y_centroids]))


def remove_cells_outside_circular_region(thresholded_cell_image,center,radius):
    print('...Removing cells that fall outside the defined region')
    
    #Define a grid of (x,y) coordinates to represent the pixel locations in the image. Necessary for making a contour plot later.
    x,y = [np.arange(thresholded_cell_image.shape[1]),np.arange(thresholded_cell_image.shape[0])]
    x,y = np.meshgrid(x,y)
    
    #Determine the (x,y) of the various closed contours in a thresholded_cell_image
    list_of_contours,hierarchy = cv2.findContours(thresholded_cell_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1:]
    new_thresholded_cell_image = np.zeros(thresholded_cell_image.shape,dtype=np.uint8)
    # cell_centroids = calculate_cell_centroids(cell_field)
    # x_centroids,y_centroids=[cell_centroids[:,0],cell_centroids[:,1]]
    for contour in list_of_contours:
        temp_thresholded_cell_image = np.zeros(thresholded_cell_image.shape,dtype=np.uint8)
        temp_thresholded_cell_image = cv2.drawContours(temp_thresholded_cell_image,[contour],contourIdx=-1,color=1,thickness=-1)
        x_centroid = np.sum(temp_thresholded_cell_image*x)/np.sum(temp_thresholded_cell_image)
        y_centroid = np.sum(temp_thresholded_cell_image*y)/np.sum(temp_thresholded_cell_image)
        distance_from_circle_center = np.sqrt((x_centroid-center[0])**2+(y_centroid-center[1])**2)
        if distance_from_circle_center < radius:
            new_thresholded_cell_image += temp_thresholded_cell_image
    return new_thresholded_cell_image