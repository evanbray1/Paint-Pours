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
image_dimensions = [2*1920,2*1080]
# image_dimensions = [2000,1600]
#image_dimensions = [1600,2000]
# image_dimensions = [3000,2400]

save_image = True                  #Do you want to save a .png copy of your image?
num_images = 30                    #How many images do you want to produce?
make_surface_plot = False           #Helpful for diagnostic purposes in case you want to see a low-res surface plot of your image
add_cells = True

cmap_name = 'any'                 #Which colormap do you want to use for your images? Use "any" to pick one at random, 'custom' to use a custom one from the block below, or pick one from this list: https://matplotlib.org/stable/tutorials/colors/colormaps.html
output_directory = 'Pictures/4K Wallpapers/'   #The relative directory where the output images will be saved
# output_directory = '8x10s to print/'   #The relative directory where the output images will be saved

########################################

#Define a grid of (x,y) coordinates to represent the pixel locations in the image. Necessary for making a contour plot later.
x,y = [np.arange(image_dimensions[0]),np.arange(image_dimensions[1])]
x,y = np.meshgrid(x,y)
    
for i in range(num_images):
    print('Currently making image ',i+1,' of ',num_images)
    plt.close('all')
    
    # octave_powers = [1,np.round(np.random.uniform(0.1,0.5),1),np.round(np.random.uniform(0.0,0.1),2),np.random.choice([0.0,0.01,0.02,0.08],p=[0.55,0.15,0.15,0.15])] #Recommend the following [1,(0.1-0.5),(0.0-0.1),(0.0-0.1)]. The 0th octave should always keep a power of 1, for convenience purposes.
    stretch_value = np.random.randint(-5,6)   #Changes how strongly the contour lines tend to be oriented horizontally vs. vertically. 0 = no preference. 5 = strong preference for horizontal. -5 = strong preference for vertical.
    octave_powers = [1, 0.02, 0.05, 0.0]
    #Actually calculate the Fractal noise image!
    noise_field, vector_info = paint_pour_tools.fractal_noise(image_dimensions, octave_powers,stretch_value)
    
    #Normalize the image to the range [0,1]
    noise_field = (noise_field-np.min(noise_field))/(np.max(noise_field)-np.min(noise_field))    
    
    if add_cells == True:
        include_perimeter_regions = np.random.choice([True,False])
        gauss_smoothing_sigma = np.random.choice([5,15,25])
        threshold_percentile = np.random.choice([60,70,80])
        cell_field = 0.5*paint_pour_tools.make_cell_image(image_dimensions, num_voronoi_points=200, gauss_smoothing_sigma=gauss_smoothing_sigma, threshold_percentile=threshold_percentile, 
                                         minimum_region_area=20,show_plots=False,include_perimeter_regions=include_perimeter_regions)
        
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
    
    my_dpi=120      #Don't fuck with this. Idk why but 120 always works, regardless of monitor.
    
    if save_image == True:
        # filename = cmap.name+'_'+str(num_levels)+'levels_'+'_'.join(['{:.2f}'.format(i) for i in octave_powers[1:]])+'_stretch'+str(stretch_value)+\
        #     '_gausssmooth'+str(gauss_smoothing_sigma)+'_threshold'+str(threshold_percentile)+'.png'
        filename = cmap.name+'_'+'_stretch'+str(stretch_value)+\
            '_gausssmooth'+str(gauss_smoothing_sigma)+'_threshold'+str(threshold_percentile)+'.png'
                    
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