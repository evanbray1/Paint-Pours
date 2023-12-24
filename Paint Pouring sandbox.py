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
import paint_pour_tools
plt.close('all')
start_time = time.time()
np.random.seed(7)

#TODO, fix the "your chosen colormap" plot so it's not uniformly divided.

# x = np.linspace(0,1,1000)
# y = x.copy()
# y -= np.min(y[y != -np.inf])
# y /= y.max()
# y_log = np.log10(exponent*y+1)/np.log10(exponent)
# y_log -= np.min(y_log[y_log != -np.inf])
# y_log /= y_log.max()



# fig,ax = plt.subplots(1)
# ax.plot(x,y,label='Linear')
# for exponent in np.logspace(-1,5,7):
#     ax.plot(x,log_rescaler(y, exponent),label='log base '+str(np.round(exponent,1)))
# for exponent in np.logspace(0,5,6):
#     ax.plot(x,power_rescaler(y, exponent),':',label='power base '+str(np.round(exponent,1)))
# ax.legend(loc='best')
# ax.set_xlim(0,1)
# ax.set_ylim(0,1)
# fig.tight_layout()


#To-do, pick a more optimum set of gauss_smoothing and threshold parameters. 
#Refer to images saved in Pictures folder.

#########USER-DEFINED VARIABLES#########
# image_dimensions = [500,500]    #[Width,Height] in pixels
# image_dimensions = [1200,800]
image_dimensions = [1*1920,1*1080]
# image_dimensions = [2000,1600]
# image_dimensions = [2000,2000]
# image_dimensions = [3000,2400]

display_image = True               #Do you want to display the image on the screen? NOTE, automatically set to false when image_dimensions > [1920,1080], otherwise there's some rendering problems.
save_image = True                 #Do you want to save a .png copy of your image?
num_images = 1                    #How many images do you want to produce?
show_intermediate_plots = True  #Do you want to show some intermediate results to help with troubleshooting?
make_surface_plot = True           #Helpful for diagnostic purposes in case you want to see a low-res surface plot of your image
add_cells = False
display_colormap = True         #Do you want to display your chosen colormap in a separate window?

cmap_name = 'any'                 #Which colormap do you want to use for your images? Use "any" to pick one at random, 'custom' to use a custom one from the block below, or pick one from this list: https://matplotlib.org/stable/tutorials/colors/colormaps.html
output_directory = 'Pictures/_test/'   #The relative directory where the output images will be saved

cmap_name = 'any'                 #Which colormap do you want to use for your images? Use "any" to pick one at random, 'custom' to use a custom one from the block below, or pick one from this list: https://matplotlib.org/stable/tutorials/colors/colormaps.html
output_directory = 'C:/Users/jkemb/My Drive/Python Projects/Paint Pouring/Pictures/_temp/'   #The relative directory where the output images will be saved

########################################

#Define a grid of (x,y) coordinates to represent the pixel locations in the image. Necessary for making a contour plot later.
x,y = [np.arange(image_dimensions[0]),np.arange(image_dimensions[1])]
x,y = np.meshgrid(x,y)
    
for i in range(num_images):
    print('Currently making image ',i+1,' of ',num_images)
    plt.close('all')
    
    # octave_powers = [1,np.round(np.random.uniform(0.1,0.5),1),np.round(np.random.uniform(0.0,0.1),2),np.random.choice([0.0,0.01,0.02,0.08],p=[0.55,0.15,0.15,0.15])] #Recommend the following [1,(0.1-0.5),(0.0-0.1),(0.0-0.1)]. The 0th octave should always keep a power of 1, for convenience purposes.
    stretch_value = np.random.randint(-5,6)   #Changes how strongly the contour lines tend to be oriented horizontally vs. vertically. 0 = no preference. 5 = strong preference for horizontal. -5 = strong preference for vertical.
    octave_powers = [0, 0.0, 0.1, 0.0]
    #Actually calculate the Fractal noise image!
    noise_field, vector_info = paint_pour_tools.fractal_noise(image_dimensions, octave_powers,stretch_value)
    
    #Normalize the image to the range [0,1]
    noise_field = (noise_field-np.min(noise_field))/(np.max(noise_field)-np.min(noise_field))    
    
    if add_cells == True:
        include_perimeter_regions = False#np.random.choice([True,False])
        gauss_smoothing_sigma = 15#np.random.choice([5,15,25])
        threshold_percentile = 70#np.random.choice([60,70,80])
        cell_field = paint_pour_tools.make_cell_image(image_dimensions, num_voronoi_points=100, gauss_smoothing_sigma=gauss_smoothing_sigma, threshold_percentile=threshold_percentile, 
                                         minimum_region_area=20,show_plots=show_intermediate_plots,include_perimeter_regions=include_perimeter_regions)
        
        # gauss_smoothing_options = np.arange(.001,stop=25,step=1)
        # print('Beginning to loop!')
        # for gauss_smoothing_sigma in gauss_smoothing_options:
        # cell_field = paint_pour_tools.make_cell_image(image_dimensions, num_voronoi_points=100, gauss_smoothing_sigma=gauss_smoothing_sigma, threshold_percentile=threshold_percentile, 
        #                                  minimum_region_area=20,show_plots=False,include_perimeter_regions=include_perimeter_regions)
            
            # plt.close('all')
            # plt.ioff() 
            # plt.imshow(cell_field)
            # plt.tight_layout()
            # plt.savefig('Pictures/test image_'+str(np.round(gauss_smoothing_sigma,1))+'.png',dpi=150)
            # plt.ion()
        
        #Define an area within which you want cells to appear
        area_with_cells_x,area_with_cells_y = [image_dimensions[0]/2,image_dimensions[1]/2]
        area_with_cells_radius = 200
        if show_intermediate_plots == True:
            fig,ax = plt.subplots(1)
            ax.imshow(cell_field,origin='lower')
            area_with_cells = plt.Circle((area_with_cells_x,area_with_cells_y),area_with_cells_radius,fill=None,edgecolor='r',linewidth=3)
            ax.add_patch(area_with_cells)
            fig.tight_layout()
        
        #Identify the centroids of each cell
        def calculate_cell_centroids(thresholded_cell_image):
            print('...Calculating cell centroids')
            #Determine the (x,y) of the various closed contours in a thresholded_cell_image
            list_of_contours,hierarchy = cv2.findContours(cell_field, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
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
        cell_centroids = calculate_cell_centroids(cell_field)

    

        def remove_cells_outside_circular_region(thresholded_cell_image,center,radius):
            print('...Removing cells that fall outside the defined region')
            #Determine the (x,y) of the various closed contours in a thresholded_cell_image
            list_of_contours,hierarchy = cv2.findContours(cell_field, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            new_thresholded_cell_image = np.zeros(thresholded_cell_image.shape,dtype=np.uint8)
            cell_centroids = calculate_cell_centroids(cell_field)
            x_centroids,y_centroids=[cell_centroids[:,0],cell_centroids[:,1]]
            for contour in list_of_contours:
                temp_thresholded_cell_image = np.zeros(thresholded_cell_image.shape,dtype=np.uint8)
                temp_thresholded_cell_image = cv2.drawContours(temp_thresholded_cell_image,[contour],contourIdx=-1,color=1,thickness=-1)
                x_centroid = np.sum(temp_thresholded_cell_image*x)/np.sum(temp_thresholded_cell_image)
                y_centroid = np.sum(temp_thresholded_cell_image*y)/np.sum(temp_thresholded_cell_image)
                distance_from_circle_center = np.sqrt((x_centroid-center[0])**2+(y_centroid-center[1])**2)
                if distance_from_circle_center < radius:
                    new_thresholded_cell_image += temp_thresholded_cell_image
            return new_thresholded_cell_image
        
        cell_field = remove_cells_outside_circular_region(cell_field, [area_with_cells_x,area_with_cells_y], area_with_cells_radius)
        if show_intermediate_plots == True:
            fig,ax = plt.subplots(1)
            ax.imshow(cell_field,origin='lower')
            area_with_cells = plt.Circle((area_with_cells_x,area_with_cells_y),area_with_cells_radius,fill=None,edgecolor='r',linewidth=3)
            ax.add_patch(area_with_cells)
            fig.tight_layout()
            
        # os.sys.exit()
        
        
        noise_field += cell_field
    
    #If desired, re-map the values of noise_field. Perhaps with a log or power stretch.
    rescaling_exponent = 1#10**np.random.uniform(0.1,3)
    noise_field = paint_pour_tools.log_rescaler(noise_field,exponent=rescaling_exponent)
    
    #Pick the number of levels in your contour map, and the Z-values they correspond to
    num_levels = 16#np.random.choice([7,10,13,17,20,25,30,40,50])
    levels = np.sort(np.random.uniform(low=noise_field.min(),high=noise_field.max(),size=num_levels))
    
    #Pick the colormap to be used for this image
    if cmap_name == 'custom':
        cmap = paint_pour_tools.make_custom_colormap(colors=['#33192F','#803D75','#CF2808','#FEE16E','#6AA886','#5CE5FB','#1A1941'],show_plot=True)
    elif cmap_name == 'any':
        cmap = paint_pour_tools.pick_random_colormap()
        
    #Pick discrete colors from the colormap and shuffle them around to make a new version.
    colors = np.random.randint(low=0,high=256,size=num_levels)  #Pick "num_levels" random colors from the chosen colormap. 
    cmap = ListedColormap([cmap(i) for i in colors],name=cmap.name)    #Re-make the colormap using our chosen colors
    
    if display_colormap == True:
        # fig_cmap,ax_cmap = plt.subplots(figsize=(8,2))
        # ax_cmap.imshow(np.outer(np.ones(100),np.arange(0,1,0.001)),cmap=cmap,origin='lower')
        # ax_cmap.set_title('Your chosen colormap')
        # ax_cmap.axis('off')
        # fig_cmap.tight_layout()
        
        fig = plt.figure()
        ax = fig.add_axes([0.05, 0.80, 0.9, 0.1])
        
        cb = colorbar.ColorbarBase(ax, orientation='horizontal',cmap=cmap)
    
    #Plotting time
    if display_image == True:
        #Temporarily disable displaying the image if the image is larger than the screen, otherwise we'll get weird graphical bugs
        if (image_dimensions[0] > 1920) or (image_dimensions[1] > 1080): 
            plt.ioff()
        fig,ax = plt.subplots(1,figsize=(image_dimensions[0]/120, image_dimensions[1]/120))
        ax = plt.Axes(fig, [0., 0., 1., 1.])           #make it so the plot takes up the ENTIRE figure
        fig.add_axes(ax)
        # contour = ax.contourf(x,y,noise_field,cmap=cmap,levels=levels,extend='both')
        # ax.set_xlim(0,image_dimensions[0])     #Set the x- and y-bounds of the plotting area.
        # ax.set_ylim(0,image_dimensions[1])
        ax.imshow(noise_field,cmap=cmap)
        # fig.tight_layout()
        if (image_dimensions[0] > 1920) or (image_dimensions[1] > 1080):    #Re-enable interactive mode, in case it was turned off earlier.
            plt.ion()
    
    if save_image == True:
        filename = cmap.name+'_'+str(num_levels)+'levels_'+'_'.join(['{:.2f}'.format(i) for i in octave_powers[1:]])+\
            '_stretch'+str(stretch_value)+'_exponent'+'{:.0f}'.format(rescaling_exponent)

        if add_cells == True:
            filename += '_gausssmooth'+str(gauss_smoothing_sigma)+'_threshold'+str(threshold_percentile)+'.png'

                    
        #Save the images in the desired output_directory
        output_directory_temp = output_directory#+cmap_name_temp+'/'
        if os.path.exists(output_directory_temp) == False:    #Does the specified directory already exist?
            os.makedirs(output_directory_temp)                    #Create the directory if necessary.
        fig.savefig(output_directory_temp+filename+'.png',dpi=120)    #save the displayed image

    
    #Sometimes it can be helpful to view the image as an interactable surface plot instead of a contour plot.
    if make_surface_plot == True:
        y, x = np.meshgrid(np.arange(image_dimensions[0]),np.arange(image_dimensions[1]))
        fig1, ax1 = plt.subplots(figsize=(8,6),subplot_kw={"projection": "3d"})
        surf = ax1.plot_surface(x,y,noise_field, cmap=cmap,linewidth=0, antialiased=False,rcount=50,ccount=50)
        fig1.tight_layout()

end_time=time.time()
elapsed_time = round(end_time - start_time,2)   #calculate the amount of time that has elapsed since program start, and print it
print('Elapsed Time: '+str(elapsed_time)+' seconds')