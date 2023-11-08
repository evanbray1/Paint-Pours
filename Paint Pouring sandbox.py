import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
from scipy.spatial import Voronoi, voronoi_plot_2d
#import matplotlib.cm as cm
import numpy as np
import time
import os
import cv2
import paint_pour_tools
plt.close('all')
start_time = time.time()
# np.random.seed(7)

#########USER-DEFINED VARIABLES#########
# image_dimensions = [500,500]    #[Width,Height] in pixels
# image_dimensions = [1200,800]
image_dimensions = [1*1920,1*1080]
# image_dimensions = [2000,1600]
#image_dimensions = [1600,2000]
# image_dimensions = [3000,2400]

save_image = True                  #Do you want to save a .png copy of your image?
num_images = 1                    #How many images do you want to produce?
make_surface_plot = False           #Helpful for diagnostic purposes in case you want to see a low-res surface plot of your image
add_cells = True

cmap_name = 'any'                 #Which colormap do you want to use for your images? Use "any" to pick one at random, 'custom' to use a custom one from the block below, or pick one from this list: https://matplotlib.org/stable/tutorials/colors/colormaps.html
output_directory = 'Pictures/_test/'   #The relative directory where the output images will be saved
# output_directory = '8x10s to print/'   #The relative directory where the output images will be saved

########################################

def make_cell_image(image_dimensions,num_voronoi_points,gauss_smoothing_sigma,threshold_percentile,minimum_region_area,show_plots=False):
    #num_voronoi_points = number of scatterpoints used to generate the Voronoi diagram. Higher number = more cells, in general
    #gauss_smoothing_sigma = In pixels, how "round" the corners of the cells are
    #threshold_percentile = thickness of the webbing between cells
    #minimum_region_area = any cells with an area smaller than this (in pixels) will be removed
    vor = paint_pour_tools.make_voronoi(num_voronoi_points,*image_dimensions)

    
    
    # print('Making cell image!')
    #Convert the voronoi diagram ridges into (x,y) points
    x_new,y_new = paint_pour_tools.voronoi_to_points(vor,1)
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
    image_outer_removed = paint_pour_tools.remove_perimeter_regions(image_threshold)
    if show_plots == True:    
        ax2[3].imshow(image_outer_removed,origin='lower')
        ax2[3].set_title('Border regions removed')
    # print('Done removing outer regions')
    
    #Identify regions that are smaller in size than some threshold
    image_small_removed = paint_pour_tools.remove_small_regions(image_outer_removed, minimum_region_area)
    if show_plots == True:    
        ax2[4].imshow(image_small_removed,origin='lower')
        ax2[4].set_title('Small regions removed')
        fig2.tight_layout()
    # print('Done removing small regions')

    return image_small_removed

#Define a grid of (x,y) coordinates to represent the pixel locations in the image. Necessary for making a contour plot later.
x,y = [np.arange(image_dimensions[0]),np.arange(image_dimensions[1])]
x,y = np.meshgrid(x,y)
    
for i in range(num_images):
    print('Currently making image ',i+1,' of ',num_images)
    plt.close('all')
    
    octave_powers = [1,np.round(np.random.uniform(0.1,0.5),1),np.round(np.random.uniform(0.0,0.1),2),np.random.choice([0.0,0.01,0.02,0.08],p=[0.55,0.15,0.15,0.15])] #Recommend the following [1,(0.1-0.5),(0.0-0.1),(0.0-0.1)]. The 0th octave should always keep a power of 1, for convenience purposes.
    stretch_value = np.random.randint(-5,6)   #Changes how strongly the contour lines tend to be oriented horizontally vs. vertically. 0 = no preference. 5 = strong preference for horizontal. -5 = strong preference for vertical.
    octave_powers = [1, 0.02, 0.0, 0.0]
    #Actually calculate the Fractal noise image!
    noise_field, vector_info = paint_pour_tools.fractal_noise(image_dimensions, octave_powers,stretch_value)
    
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
                                         minimum_region_area=10,show_plots=False)

        # plt.imshow(cell_field)
        #############PICK UP HERE****
        
        noise_field += cell_field
    
    #Pick the number of levels in your contour map, and the Z-values they correspond to
    num_levels = np.random.choice([7,10,13,17,20,25,30,40,50])
    levels = np.sort(np.random.uniform(low=noise_field.min(),high=noise_field.max(),size=num_levels))
    
    #Pick the colormap to be used for this image and record its name
    if cmap_name == 'custom':
        cmap = paint_pour_tools.make_custom_colormap(colors=['#33192F','#803D75','#CF2808','#FEE16E','#6AA886','#5CE5FB','#1A1941'],show_plot=True)
    elif cmap_name == 'any':
        cmap = paint_pour_tools.pick_random_colormap()
            
    # colors = np.random.randint(low=0,high=256,size=num_levels)  #Pick "num_levels" random colors from the chosen colormap. 
    # cmap = ListedColormap([cmap(i) for i in colors],name=cmap.name)    #Re-make the colormap using our chosen colors
    
    #TEMPORARY PLOTTING STUFF
    fig,ax = plt.subplots(1,figsize=(image_dimensions[0]/120, image_dimensions[1]/120))
    ax = plt.Axes(fig, [0., 0., 1., 1.])           #make it so the plot takes up the ENTIRE figure
    fig.add_axes(ax)
    ax.set_xlim(0,image_dimensions[0])     #Set the x- and y-bounds of the plotting area.
    ax.set_ylim(0,image_dimensions[1])
    ax.imshow(noise_field,cmap=cmap)
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
            filename = cmap.name+'_'+str(num_levels)+'levels_'+'_'.join(['{:.2f}'.format(i) for i in octave_powers[1:]])+'_stretch'+str(stretch_value)+\
                '.png'
        else:
            filename = str(num_levels)+'levels_'+'_'.join(['{:.2f}'.format(i) for i in octave_powers[1:]])+'_stretch'+str(stretch_value)+\
                '_'+cmap.name+'.png'
                    
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