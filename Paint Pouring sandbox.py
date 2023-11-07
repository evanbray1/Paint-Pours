import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.ndimage import gaussian_filter
#import matplotlib.cm as cm
import numpy as np
import time
import os
import cv2
plt.close('all')
start_time = time.time()
# np.random.seed(7)
#THIS IS A TEMPORARY EDIT

#########USER-DEFINED VARIABLES#########
# image_dimensions = [500,500]    #[Width,Height] in pixels
image_dimensions = [1200,800]
# image_dimensions = [1920,1080]
# image_dimensions = [2000,1600]
#image_dimensions = [1600,2000]
# image_dimensions = [3000,2400]

save_image = False                  #Do you want to save a .png copy of your image?
num_images = 1                    #How many images do you want to produce?
make_surface_plot = False           #Helpful for diagnostic purposes in case you want to see a low-res surface plot of your image
add_cells = True

cmap_name = 'any'                 #Which colormap do you want to use for your images? Use "any" to pick one at random, 'custom' to use a custom one from the block below, or pick one from this list: https://matplotlib.org/stable/tutorials/colors/colormaps.html
output_directory = '_temp wallpapers/'   #The relative directory where the output images will be saved
# output_directory = '8x10s to print/'   #The relative directory where the output images will be saved

########################################


#Want to make a custom colormap? Do it here.
if cmap_name == 'custom':
    colors=['#33192F','#803D75','#CF2808','#FEE16E','#6AA886','#5CE5FB','#1A1941']
    nodes = np.linspace(0,1.0,len(colors))
    # nodes = np.concatenate((np.linspace(0,0.6,len(colors)-1),np.array([1.0])))
    cmap_custom = LinearSegmentedColormap.from_list('custom', list(zip(nodes, colors)))
    
    plt.close('all')
    fig_cmap,ax_cmap = plt.subplots(figsize=(8,2))
    ax_cmap.imshow(np.outer(np.ones(100),np.arange(0,1,0.001)),cmap=cmap_custom,origin='lower')
    ax_cmap.set_title('Your custom colormap')
    ax_cmap.axis('off')
    fig_cmap.tight_layout()

#You should really read the Wikipedia page on Perlin Noise before trying to dissect this function. (https://en.wikipedia.org/wiki/Perlin_noise#Algorithm_detail)
#Seriously, this part is complicated and took a lot of iterations, linear algebra, and troubleshooting. I would just take it at face value.
#Maybe watch my earlier "How It's Done" video, too...
def perlin_field(image_dimensions,octave,stretch):
    
    #Break the image field up into a grid of NxM "cells". Where N,M are defined by the current "octave" and "stretch" parameters
    #Setting stretch=2 adds two columns. So now you have a grid of [2,4] cells when octave=1. This roughly 
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
    #Help for Perlin flow fields. Not necessary for paint pour stuff.
    # vector_dir_x[-1,:] = vector_dir_x[0,:]
    # vector_dir_x[:,-1] = vector_dir_x[:,0]
    # vector_dir_y[-1,:] = vector_dir_y[0,:]
    # vector_dir_y[:,-1] = vector_dir_y[:,0]
    
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
            
            #Dot product each grid point's offset vector with the gradient vector in the bottom left
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
        # image = image[int(image_dimensions[0]/2):int(image_dimensions[0]+image_dimensions[0]/2),int(image_dimensions[1]/2):int(image_dimensions[1]+image_dimensions[1]/2)]
        image = image[int(image_dimensions[1]/2):int(image_dimensions[1]+image_dimensions[1]/2),int(image_dimensions[0]/2):int(image_dimensions[0]+image_dimensions[0]/2)]

    #Save some handy diagnostic info about the random vectors used to create this particular Perlin noise image
    vector_info = [vector_coords_x, vector_coords_y, vector_dir_x, vector_dir_y]
    
    return image.astype('float32'), vector_info

#Fractal noise is just multiple layers (octaves) of Perlin noise added on top of one another.
#This function works by calling the Perlin_field function multiple times, each time with a different octave, and adding the result to a master image array
#Typically each octave gets multiplied by 1/(2^octave) before adding it to the master image array. 
#However, here we replace 1/(2^octave) with "relative_power" which gets randomly chosen from an octave-dependent range of values.
def fractal_noise(image_dimensions,relative_powers,stretch):
    num_octaves = len(relative_powers)
    image = np.zeros((image_dimensions[1],image_dimensions[0]))     #Define an empty array where we'll build the final image
    
    #Calculate multiple Perlin noise fields. Each one is twice as dense as the last.
    for i in range(0,num_octaves):
        if relative_powers[i]>0:    #No point in expending the comput
            perlin_image, vectors = perlin_field(image_dimensions,i,stretch)    #Calculate a Perlin noise field
            image += relative_powers[i]*perlin_image                                  #Add that Perlin noise field to the total, with geometrically decreasing weighting.
            # image += 1/(2**i)*perlin_image
        
    return image.astype('float32'), vectors

def make_voronoi(npoints,width,height):
    x = np.random.uniform(0,width,npoints)
    y = np.random.uniform(0,height,npoints)
    points = np.array(list(zip(x,y)))
    return Voronoi(points)

def voronoi_to_points(voronoi,spacing):
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

def remove_outer_regions(thresholded_image):
    image_with_border = cv2.copyMakeBorder(thresholded_image,1,1,1,1,cv2.BORDER_CONSTANT,value=1)
    flooded_image = cv2.floodFill(image_with_border,None,(0,0),0)[1]
    h,w = flooded_image.shape
    trimmed_image = flooded_image[1:h-1,1:w-1]
    return trimmed_image

def get_contour_pixel_areas(image,list_of_contours):
    height,width = image.shape
    areas = []
    for contour in list_of_contours:
        temp_image = np.zeros(image.shape,dtype=np.uint8)
        temp_image = cv2.drawContours(temp_image,[contour],contourIdx=-1,color=1,thickness=-1)
        contour_area = np.count_nonzero(temp_image)
        areas.append(contour_area)
    return np.array(areas)

def remove_small_regions(image,size_threshold):
    image,contours,hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
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

def make_cell_image(image_dimensions,num_voronoi_points,gauss_smoothing_sigma,threshold_percentile,minimum_region_area,show_plots=False):
    vor = make_voronoi(num_voronoi_points,*image_dimensions)
    #num_voronoi_points = number of scatterpoints used to generate the Voronoi diagram. Higher number = more cells, in general
    #gauss_smoothing_sigma = how "round" the corners of the cells are
    #threshold_percentile = thickness of the webbing between cells
    #minimum_region_area = any cells with an area smaller than this (in pixels) will be removed
    
    
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
    
    #Remove the regions that fall over the border
    image_outer_removed = remove_outer_regions(image_threshold)
    if show_plots == True:    
        ax2[3].imshow(image_outer_removed,origin='lower')
        ax2[3].set_title('Border regions removed')
    # print('Done removing outer regions')
    
    #Identify regions that are smaller in size than some threshold
    image_small_removed = remove_small_regions(image_outer_removed, minimum_region_area)
    if show_plots == True:    
        ax2[4].imshow(image_small_removed,origin='lower')
        ax2[4].set_title('Small regions removed')
        fig2.tight_layout()
    # print('Done removing small regions')

    return image_small_removed

#Define a grid of (x,y) coordinates to represent the pixel locations in the image. Necessary for making a contour plot later.
x,y = [np.arange(image_dimensions[0]),np.arange(image_dimensions[1])]
x,y = np.meshgrid(x,y)

#Some colormaps are just bad looking for this art, IMO. I list them here so I can make sure to avoid them during the random-picking process later.
bad_cmaps = ['Accent','Paired','Dark2','Set1','Set2','Set3','tab10','tab20','tab20c','tab20b','binary','Pastel1','Pastel2','gist_yarg','gist_gray','brg','CMRmap','gist_ncar','gist_rainbow','hsv','terrain','gnuplot2','nipy_spectral','prism']
non_reversed_colormaps = [x for x in plt.colormaps() if '_r' not in x]      #Generate a list of all colormaps that don't contain "_r" in their name, indicating they are just a reversed version of another colormap. "Grays" and "Grays_r" look fundamentally the same for this type of art.
#%%

for i in range(num_images):
    print('Currently making image ',i+1,' of ',num_images)
    plt.close('all')
    
    octave_powers = [1,np.round(np.random.uniform(0.1,0.5),1),np.round(np.random.uniform(0.0,0.1),2),np.random.choice([0.0,0.01,0.02,0.08],p=[0.55,0.15,0.15,0.15])] #Recommend the following [1,(0.1-0.5),(0.0-0.1),(0.0-0.1)]. The 0th octave should always keep a power of 1, for convenience purposes.
    stretch_value = np.random.randint(-5,6)   #Changes how strongly the contour lines tend to be oriented horizontally vs. vertically. 0 = no preference. 5 = strong preference for horizontal. -5 = strong preference for vertical.
    octave_powers = [1, 0.02, 0.0, 0.0]
    #Actually calculate the Fractal noise image!
    noise_field, vector_info = fractal_noise(image_dimensions, octave_powers,stretch_value)
    
    #Normalize the image to the range [0,1]
    noise_field = (noise_field-np.min(noise_field))/(np.max(noise_field)-np.min(noise_field))    
    
    if add_cells == True:
        #To do this, we create a separate fractal noise layer that we multiply by an array of weights, and add this result to the final "cell-free" image. 
        # octave_powers_cell_weights = [1,np.round(np.random.uniform(0.1,0.5),1)]
        # octave_powers_cell_weights = [1,0.5]

        # cell_weight_array,vector_info_weights = fractal_noise(image_dimensions, octave_powers_cell_weights,0)
        # cell_weight_array = (cell_weight_array-np.min(cell_weight_array))/(np.max(cell_weight_array)-np.min(cell_weight_array))
    
        # #Create the noise layer that represents contribution from the cells
        # octave_powers_cells = [0,0,0,0,np.random.choice([0.01,0.02,0.08])]
        # cell_field,vector_info_cells = fractal_noise(image_dimensions, octave_powers_cells,0)
        # cell_field = cell_field**2
        # cell_field = (cell_field-np.min(cell_field))/(np.max(cell_field)-np.min(cell_field))        
    
        # cell_field *= 0.2*cell_weight_array
        # cell_field = np.random.uniform(0.1,0.5)*make_cell_image(image_dimensions, 200, 15, 70, 10)
        cell_field = 0.5*make_cell_image(image_dimensions, num_voronoi_points=200, gauss_smoothing_sigma=15, threshold_percentile=70, 
                                         minimum_region_area=10)

        # plt.imshow(cell_field)
        #############PICK UP HERE****
        
    noise_field += cell_field
    
    #Pick the number of levels in your contour map, and the Z-values they correspond to
    num_levels = np.random.choice([7,10,13,17,20,25,30,40,50])
    levels = np.sort(np.random.uniform(low=noise_field.min(),high=noise_field.max(),size=num_levels))
    
    #Pick the colormap to be used for this image and record its name
    if cmap_name == 'custom':
        cmap_name_temp = cmap_custom.name
        cmap = cmap_custom
    else:
        if cmap_name == 'any':
            cmap_name_temp = np.random.choice(non_reversed_colormaps)
        else:
            cmap_name_temp = cmap_name
        cmap = plt.cm.get_cmap(cmap_name_temp)    #Retrieve the colormap

    #Re-pick the colormap if it randomly chose a Certified Ugly (TM) one the first time. Keep picking new colormaps until it picks one that isn't ugly.
    if (cmap_name not in bad_cmaps) & (cmap_name != 'custom'):
        while any(cmap.name in s for s in bad_cmaps):
            print('Randomly chose an ugly colormap! Choosing again...')
            cmap = plt.cm.get_cmap(np.random.choice(non_reversed_colormaps))
            
    colors = np.random.randint(low=0,high=256,size=num_levels)  #Pick "num_levels" random colors from the chosen colormap. 
    cmap = ListedColormap([cmap(i) for i in colors])    #Re-make the colormap using our chosen colors
    
    #TEMPORARY PLOTTING STUFF
    fig,ax = plt.subplots(1,figsize=(image_dimensions[0]/120, image_dimensions[1]/120))# ax = plt.Axes(fig, [0., 0., 1., 1.])           #make it so the plot takes up the ENTIRE figure
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    fig.add_axes(ax)
    ax.set_xlim(0,image_dimensions[0])     #Set the x- and y-bounds of the plotting area.
    ax.set_ylim(0,image_dimensions[1])
    ax.imshow(noise_field,cmap=cmap_name_temp)
    fig.tight_layout()
    # os.sys.exit()
    
    
    #### PLOTTING TIME ####
    my_dpi=120      #Don't fuck with this. Idk why but 120 always works, regardless of monitor.
    # fig,ax = plt.subplots(1,figsize=(image_dimensions[0]/my_dpi, image_dimensions[1]/my_dpi))# ax = plt.Axes(fig, [0., 0., 1., 1.])           #make it so the plot takes up the ENTIRE figure
    # ax = plt.Axes(fig, [0., 0., 1., 1.])
    # fig.add_axes(ax)                  
    # contour = ax.contourf(x,y,noise_field,cmap=cmap,levels=levels,extend='both')
    
    # ax.set_xlim(0,image_dimensions[0])     #Set the x- and y-bounds of the plotting area.
    # ax.set_ylim(0,image_dimensions[1])
    # fig.tight_layout()
    
    if save_image == True:
        if cmap_name == 'any':    #Change the filename depending on whether you opted to use a random colormap or not.
            filename = cmap_name_temp+'_'+str(num_levels)+'levels_'+'_'.join(['{:.2f}'.format(i) for i in octave_powers[1:]])+'_stretch'+str(stretch_value)+\
                '.png'
        else:
            filename = str(num_levels)+'levels_'+'_'.join(['{:.2f}'.format(i) for i in octave_powers[1:]])+'_stretch'+str(stretch_value)+\
                '_'+cmap_name_temp+'.png'
                    
        #Save the images in the desired output_directory
        output_directory_temp = output_directory#+cmap_name_temp+'/'
        if os.path.exists(output_directory_temp) == False:    #Does the specified directory already exist?
            os.makedirs(output_directory_temp)                    #Create the directory if necessary.
        fig.savefig(output_directory_temp+filename,dpi=my_dpi)    #save the displayed image
    
    #Sometimes it can be helpful to view the image as an interactable surface plot instead of a contour plot.
    if make_surface_plot == True:
        y, x = np.meshgrid(np.arange(image_dimensions[0]),np.arange(image_dimensions[1]))
        fig1, ax1 = plt.subplots(figsize=(8,6),subplot_kw={"projection": "3d"})
        surf = ax1.plot_surface(x,y,noise_field, cmap=cmap,linewidth=0, antialiased=False,rcount=50,ccount=50)
        fig1.tight_layout()

#Diagnostic stuff for troubleshooting colormaps
# cmap = plt.cm.get_cmap('viridis')
# x = np.linspace(0,256,1000)
# x = np.arange(256)
# y = np.zeros(len(x))
# color = cmap(x)
# plt.scatter(x,y,c=[cmap(i) for i in x])

end_time=time.time()
elapsed_time = round(end_time - start_time,2)   #calculate the amount of time that has elapsed since program start, and print it
print('Elapsed Time: '+str(elapsed_time)+' seconds')