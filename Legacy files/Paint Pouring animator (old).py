#%%
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from scipy import stats
import colorsys
import matplotlib.colors as mcolors
import matplotlib.animation as animation
from numba import njit
import pandas as pd
import matplotlib.tri as tri
from scipy.spatial import Delaunay
from scipy.spatial import Voronoi, voronoi_plot_2d

start_time = time.time()
plt.close('all')
#TO DO
#Add a filter that deletes a point if the closest N% of points aren't the same type

# num_frames = 50
simulation_duration = 10 #in seconds
timestep = 0.02
# droplets_per_point = 5
starting_coords = [0.1,0.5]

save_animation = 0
show_plot = 0

pouring_blueprint = pd.DataFrame([],columns=['Start Time (s)','Duration (s)','Pour Rate (particles/s)','Starting_x','Starting_y',
                                             'Direction (deg)','Velocity (units/s)','Color'])
pouring_blueprint.loc[0] = [0,1.0,500,0.1,0.5,0,0.8,'b']    #The first element of the first row should always be 0!
pouring_blueprint.loc[1] = [2,1.0,500,0.1,0.1,90,0.8,'r']    #The first element of the first row should always be 0!
pouring_blueprint.loc[2] = [3.5,1.0,500,0.0,0.1,45,1.2,'g']    #The first element of the first row should always be 0!

paint_colors = ['b', 'r', 'g']
custom_cmap = mcolors.ListedColormap(paint_colors)

# colors = ['blue','red','goldenrod']
# nodes = [0,0.5,1]
# custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom", list(zip(nodes, colors)))

class MyClass():
    def __init__(self):
        self.linewidth = 1.5
        self.v_max = 0.1#1.0
        self.mass = 1
        self.dt = timestep
        # self.charge = 5E-4
        # self.static_friction_force = 0.03 #0.06 is a good value
        self.interaction_radius = 0.1 #0.03 is a good value
        self.current_time = 0
        self.current_pour = 0
        self.animation_complete = 0
        
        self.viscosity = 1
        self.charge = 5E-3/self.viscosity
        self.static_friction_force = .1*self.viscosity #0.06 is a good value
        self.sigma = 0.05
        
        
        #For dense runs
        # self.charge = 1E-4
        # self.static_friction_force = 0.03 #0.06 is a good value
        # self.interaction_radius = .04 #0.03 is a good value
        
        self.fig,self.ax = plt.subplots(1,1,figsize=(7,6))               #Change the figure size here
        self.scatter1 = self.ax.scatter([], [],c=[],s=15.0,vmin=0,vmax=len(paint_colors)-1,cmap=custom_cmap,label='particle positions')
        self.ax.set_aspect('equal')
        print('=====INIT FUNCTION RAN=====')

    def plot_init(self):
        self.line1, = self.ax.plot([0,0], [1,0], lw=self.linewidth, color='black')  #[xi,xf],[yi,yf]
        self.line2, = self.ax.plot([0,1], [1,1], lw=self.linewidth, color='black')
        self.line3, = self.ax.plot([1,1], [1,0], lw=self.linewidth, color='black')
        self.line4, = self.ax.plot([0,1], [0,0], lw=self.linewidth, color='black')
        self.fontsize = 14
        self.ax.set_xlim(-0.1,1.1)
        self.ax.set_ylim(-0.1,1.1)
        self.ax.set_title('Paint Pour Sandbox',fontsize=self.fontsize)
        self.ax.tick_params(axis='both',labelsize=self.fontsize/1.3)
        self.fig.tight_layout()

        #Initialize variables
        self.x = np.array([])
        self.y = np.array([])
        self.color = np.array([])
        self.v_x = np.array([])
        self.v_y = np.array([])
        self.v_mag = np.array([])
        self.update_scatterplot()
        return self.scatter1

        # return self.line
    
    #Function that is called every time the x/y coordinate arrays are updated
    def update_scatterplot(self):
        # print(x)
        coords_temp= np.array([self.x,self.y]).T
        self.scatter1.set_offsets(coords_temp)
        # self.scatter1.set_array(np.array(len(coords_temp)*list(np.linspace(0,1,10))))
        # self.scatter1.set_array(coords_temp[:,0])
        # self.scatter1.set_array(np.random.uniform(0,1,len(coords_temp)))
        # self.scatter1.set_array(np.array(len(coords_temp[:,0])*[1]))
        # print(len(coords_temp[:,0]))
        # print(len(np.array(len(coords_temp[:,0])*custom_cmap(self.current_pour+1))))
        # self.color = np.array(len(coords_temp[:,0])*[self.current_pour])
        # print(len(self.x))
        # print(len(self.color))
        self.scatter1.set_array(self.color)

        return self.scatter1


    #Actions to perform with each iteration
    def ani_update(self, i):
        if self.animation_complete == 0:
    
            print('====Current Time: ',np.round(self.current_time,2))
            print('Current pour # is: ',self.current_pour)
    
            #Drop more points onto the board
            
            if self.current_time<(pouring_blueprint['Start Time (s)'][self.current_pour]+pouring_blueprint['Duration (s)'][self.current_pour]):
                # print(round(pouring_blueprint['Pour Rate (particles/s)'][self.current_pour]*self.dt))
    
                points_to_add = round(pouring_blueprint['Pour Rate (particles/s)'][self.current_pour]*self.dt)
    
                # print('Adding ',points_to_add, 'Points!')
                # self.x = np.append(self.x,np.random.uniform(0,0.02*0.1/self.dt,points_to_add) 
                #     + pouring_blueprint['Starting_x'][self.current_pour]+np.cos(np.deg2rad(pouring_blueprint['Direction (deg)'][self.current_pour]))
                #     *(self.current_time - pouring_blueprint['Start Time (s)'][self.current_pour])*pouring_blueprint['Velocity (units/s)'][self.current_pour])
                # self.y = np.append(self.y,np.random.uniform(0,0.02*0.1/self.dt,points_to_add) 
                #    + pouring_blueprint['Starting_y'][self.current_pour]+np.sin(np.deg2rad(pouring_blueprint['Direction (deg)'][self.current_pour]))
                #    *(self.current_time - pouring_blueprint['Start Time (s)'][self.current_pour])*pouring_blueprint['Velocity (units/s)'][self.current_pour])
   
                self.x = np.append(self.x,np.random.normal(0,0.0001/self.dt,points_to_add) 
                    + pouring_blueprint['Starting_x'][self.current_pour]+np.cos(np.deg2rad(pouring_blueprint['Direction (deg)'][self.current_pour]))
                    *(self.current_time - pouring_blueprint['Start Time (s)'][self.current_pour])*pouring_blueprint['Velocity (units/s)'][self.current_pour])
                self.y = np.append(self.y,np.random.normal(0,0.0001/self.dt,points_to_add) 
                   + pouring_blueprint['Starting_y'][self.current_pour]+np.sin(np.deg2rad(pouring_blueprint['Direction (deg)'][self.current_pour]))
                   *(self.current_time - pouring_blueprint['Start Time (s)'][self.current_pour])*pouring_blueprint['Velocity (units/s)'][self.current_pour])
   
                # print(len(self.x))
                # print(len(self.y))
                self.color = np.append(self.color,np.array([paint_pour.current_pour for i in range(points_to_add)]))
                self.v_x = np.append(self.v_x,np.zeros(points_to_add))
                self.v_y = np.append(self.v_y,np.zeros(points_to_add))                
    
            #calculate net force on each particle, one particle at a time
            self.num_particles_temp = len(self.v_x)
            self.F_total_x = np.zeros(self.num_particles_temp)
            self.F_total_y = np.zeros(self.num_particles_temp)
    
            #Calculate the forces on each particle from all the other particles
            for particle in range(0,self.num_particles_temp):
                dx_from_others = self.x - self.x[particle]
                dy_from_others = self.y - self.y[particle]
                r_from_others = np.sqrt(dx_from_others**2 + dy_from_others**2)
                r_from_others[np.where(r_from_others == 0)[0]] = np.inf    #set the distance from itself = infinity
                self.within_proximity = np.where(r_from_others < self.interaction_radius)[0]
                
                #Using Coulomb's law
                # _F_total = -self.charge/r_from_others[self.within_proximity]**2
                # print(_F_total[0])
                # F_from_others_x = -self.charge*dx_from_others[self.within_proximity]/r_from_others[self.within_proximity]**2
                # F_from_others_y = -self.charge*dy_from_others[self.within_proximity]/r_from_others[self.within_proximity]**2
                
                #Using modified Lennard-Jones
                # _F_total = -self.charge*(-1/r_from_others[self.within_proximity]+1/r_from_others[self.within_proximity]**2)
                
                #Using the Lennard-Jones Potential
                _F_total =  self.sigma**6*(r_from_others[self.within_proximity]**6-2*self.sigma**6)/r_from_others[self.within_proximity]**13
                # _F_total =  self.charge*((6*self.sigma**6/r_from_others[self.within_proximity]**7-12*self.sigma**12/r_from_others[self.within_proximity]**13))
                # print(_F_total[0])

                F_from_others_x = _F_total*dx_from_others[self.within_proximity]/r_from_others[self.within_proximity]
                F_from_others_y = _F_total*dy_from_others[self.within_proximity]/r_from_others[self.within_proximity]
                
                # _close_particles = np.where(r_from_others[self.within_proximity] < self.sigma*1.122)[0]
                # F_from_others_x[_close_particles] = np.log10(F_from_others_x[_close_particles])
                # F_from_others_y[_close_particles] = np.log10(F_from_others_y[_close_particles])

                # F_from_others_x = dx_from_others[self.within_proximity]/r_from_others[self.within_proximity]**2
                # F_from_others_y = dy_from_others[self.within_proximity]/r_from_others[self.within_proximity]**2
                
                self.F_total_x[particle],self.F_total_y[particle] = np.array([np.sum(F_from_others_x),np.sum(F_from_others_y)])
    
            # print('Net Forces on particle 0: ',self.F_total_x[0],self.F_total_y[0])
            #Update the velocities of each particle (TEMPORARY. Velocities will be updated again after taking into account friction forces)
            self.v_x += self.F_total_x/self.mass*self.dt
            self.v_y += self.F_total_y/self.mass*self.dt
            self.v_mag = np.sqrt(self.v_x**2+self.v_y**2)
            # print('Velocity of particle 0: ',self.v_x[0],self.v_y[0])
    
            #Calculate the friction force on each particle and add it to the running total
            nonzero_velocities = np.where(self.v_mag > 0)[0]    #We will only update the values where v_mag > 0, otherwise we get divide-by-zero errors
            self.F_friction_x = np.zeros(self.num_particles_temp)
            self.F_friction_y = np.zeros(self.num_particles_temp)
            self.F_friction_x[nonzero_velocities] = -1*self.static_friction_force*np.sign(self.v_x[nonzero_velocities])*abs(self.v_x[nonzero_velocities]/self.v_mag[nonzero_velocities])
            self.F_friction_y[nonzero_velocities] = -1*self.static_friction_force*np.sign(self.v_y[nonzero_velocities])*abs(self.v_y[nonzero_velocities]/self.v_mag[nonzero_velocities])

            #Calculate the net force on each particle by adding in the friction forces
            self.F_total_x[nonzero_velocities] += self.F_friction_x[nonzero_velocities]
            self.F_total_y[nonzero_velocities] += self.F_friction_y[nonzero_velocities]
            
            #Update the velocities of each particle
            self.v_x += self.F_total_x/self.mass*self.dt
            self.v_y += self.F_total_y/self.mass*self.dt
            self.v_mag = np.sqrt(self.v_x**2+self.v_y**2)
            # print(self.v_x)

            # print('(pre) Velocity of particle 0: ',self.v_x[0],self.v_y[0])
    
            #Artificially slow down any particles that would be traveling faster than v_max
            self.speeding_particles = np.where(abs(self.v_mag) > self.v_max)[0]
            # print(self.v_mag[0])
            if len(self.speeding_particles) > 0:
                print('***Detected ',len(self.speeding_particles),' speeding particles!!')
                self.v_x[self.speeding_particles] = self.v_max*self.v_x[self.speeding_particles]/self.v_mag[self.speeding_particles]
                self.v_y[self.speeding_particles] = self.v_max*self.v_y[self.speeding_particles]/self.v_mag[self.speeding_particles]
                self.v_mag[self.speeding_particles] = np.sqrt(self.v_x[self.speeding_particles]**2+self.v_y[self.speeding_particles]**2)
            # print('(post) Velocity of particle 0: ',self.v_x[0],self.v_y[0])
            # print(self.v_mag[0])

            #Update the coordinates of each particle
            # print(self.v_x)
            self.x += self.v_x*self.dt
            self.y += self.v_y*self.dt
            # print(self.x)

            # print('Coordinates of particle 0: ',self.x[0],self.y[0])
            # print('Number of current particles: ',self.num_particles_temp)
            
            #Are any values equal to NaN?
            bad_values = np.where(np.isnan(self.y) == True)[0]
            if len(bad_values) > 0:
                print('NaN values are ',bad_values)
                self.animation_complete = 1
    
            #Delete the particles that fell off the canvas
            # self.particles_to_delete = np.where((self.x < 0) | (self.x > 1) | (self.y < 0) | (self.y > 1))[0]
            self.particles_to_delete = np.where((self.x < -0.05) | (self.x > 1.05) | (self.y < -.05) | (self.y > 1.05))[0]
            if len(self.particles_to_delete) > 0:
                # print('Deleting the following particles: ',self.particles_to_delete)

                # print(len(self.particles_to_delete),' particles fell off the canvas!')
                self.x = np.delete(self.x,self.particles_to_delete)         
                self.y = np.delete(self.y,self.particles_to_delete)  
                self.v_x = np.delete(self.v_x,self.particles_to_delete)         
                self.v_y = np.delete(self.v_y,self.particles_to_delete) 
                self.color = np.delete(self.color,self.particles_to_delete) 
                self.num_particles_temp = len(self.x)
    
            #Delete the particles that are surrounded by partiles of another type
            self.particles_to_delete = np.array([])
            for particle in range(0,self.num_particles_temp):
                dx_from_others = self.x - self.x[particle]
                dy_from_others = self.y - self.y[particle]
                r_from_others = np.sqrt(dx_from_others**2 + dy_from_others**2)
                r_from_others[np.where(r_from_others == 0)[0]] = np.inf    #set the distance from itself = infinity
                closest_particles = np.argsort(r_from_others)[0:10]
                number_of_like_particles = len(np.where(self.color[closest_particles] == self.color[particle])[0])
                if number_of_like_particles/len(closest_particles) < .1:
                    self.particles_to_delete = np.append(self.particles_to_delete,particle)
                    
            # print(len(self.particles_to_delete),' particles had no adjacent like colors!')
            if len(self.particles_to_delete) > 0:
                # print('Deleting the following particles: ',self.particles_to_delete)
                self.x = np.delete(self.x,self.particles_to_delete)         
                self.y = np.delete(self.y,self.particles_to_delete)  
                self.v_x = np.delete(self.v_x,self.particles_to_delete)         
                self.v_y = np.delete(self.v_y,self.particles_to_delete) 
                self.color = np.delete(self.color,self.particles_to_delete) 
                self.num_particles_temp = len(self.x)
    
                
            #Check if the current pour is complete. If so, increment self.current_pour
            if self.current_pour < (len(pouring_blueprint)-1):
                if self.current_time>=(pouring_blueprint['Start Time (s)'][self.current_pour]+pouring_blueprint['Duration (s)'][self.current_pour]):
                    self.current_pour+=1
            # print('temp')
    
            #Update the scatterplot
            self.current_time += self.dt
            if show_plot == True:
                self.update_scatterplot()
                self.ax.set_title('Paint pour sandbox: '+str(np.round(self.current_time,1))+' seconds')
            return self.scatter1


    def animate(self):
        self.anim = animation.FuncAnimation(self.fig, self.ani_update, init_func=self.plot_init, 
                    frames=round(simulation_duration/self.dt), interval=33, blit=False,repeat=False)
        # plt.show()
        if save_animation == 1:
            self.anim.save('D:/Google Drive/Python Projects/Paint Pouring/testing.mp4', fps=1/self.dt, extra_args=['-vcodec', 'libx264'],dpi=100)
            self.animation_complete = 1



# @njit    
def particle_forces(xcoords,ycoords,particle):
    print('(inside func) particle number is ',particle)
    dx_from_others = xcoords - xcoords[particle]
    dy_from_others = ycoords - ycoords[particle]
    r_from_others = np.sqrt(dx_from_others**2 + dy_from_others**2)
    r_from_others[np.where(r_from_others == 0)[0]] = np.inf    #set the distance from itself = infinity
    F_from_others_x = -1*dx_from_others/r_from_others**2
    F_from_others_y = -1*dy_from_others/r_from_others**2
    result = np.array([np.sum(F_from_others_x),np.sum(F_from_others_y)])
    return result

# @njit
def update_velocities(v_x_temp,v_y_temp,F_total,mass):
    v_x_temp += F_total[0]/mass
    v_y_temp += F_total[1]/mass
    return v_x_temp,v_y_temp

# def testing_func(thing_to_print):
#     print(thing_to_print)
#     return thing_to_print

# animation_done = False

# def main():
paint_pour = MyClass()
paint_pour.animate()
    # return paint_pour
end_time=time.time()
elapsed_time = round(end_time - start_time,2)   #calculate the amount of time that has elapsed since program start, and print it
print('Elapsed Time: '+str(elapsed_time)+' seconds')

# paint_pour = main()
animation_done = True
#%%
plt.close('all')
fig,ax = plt.subplots(1,figsize=(6,5))
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.scatter(paint_pour.x,paint_pour.y,c=paint_pour.color)
#%%
# plt.close('all')
fig,ax = plt.subplots(1,figsize=(6,5))
ax.set_xlim(0,1)
ax.set_ylim(0,1)

for i,color_num in enumerate(np.unique(paint_pour.color)):
    _ind = np.where(paint_pour.color == color_num)
    x,y,z = paint_pour.x[_ind],paint_pour.y[_ind],paint_pour.color[_ind]
    ax.tricontourf(x,y,z,colors=pouring_blueprint['Color'][i],alpha=1.0)

#%%

# x = paint_pour.x
# y = paint_pour.y
# # z = np.random.uniform(0,1,len(x))
# z = np.ones(len(x))

# triang = tri.Triangulation(x,y)

# plt.tricontourf(x,y,z,colors='blue')

if animation_done == True:
    # plt.close('all')
    fig,ax = plt.subplots(1,figsize=(6,5))
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    color_array = []
    for i in range(len(paint_colors)):
        ind = np.where(paint_pour.color == i)[0]
        temp = len(ind)*[paint_colors[i]]
        color_array.extend(temp)
        # ax.tricontourf(paint_pour.x[ind],paint_pour.y[ind],paint_pour.color[ind],cmap=custom_cmap,levels=1,alpha=0.9)
        
        # if len(ind) > 0:
        #     vor = Voronoi(np.array([paint_pour.x[ind],paint_pour.y[ind]]).T)    
            
        #     # colorize
        #     for point in vor.point_region:
        #         if -1 not in vor.regions[point]:
        #             polygon = [vor.vertices[i] for i in vor.regions[point]]
                    
        #             # current_color = paint_colors[int(paint_pour.color[np.where(paint_pour.x == vor.points[point-1][0])[0][0]])]
                  
        #             ax.fill(*zip(*polygon),paint_colors[i])
        
    # ax.scatter(paint_pour.x,paint_pour.y,c=color_array)
    # tricontour = ax.tricontourf(paint_pour.x,paint_pour.y,paint_pour.color,cmap=custom_cmap,levels=np.arange(-0.5, len(paint_colors)),alpha=0.3)
    # ax.tricontourf(paint_pour.x,paint_pour.y,paint_pour.color)
    # cbar = fig.colorbar(tricontour,ticks=[0,1,2])
    # cbar.ax.set_xticklabels([0,1,2])
    ax.set_aspect('equal')
    ax.set_xlim(0,1.0)
    ax.set_ylim(0,1.0)
    fig.tight_layout()
    
    vor = Voronoi(np.array([paint_pour.x,paint_pour.y]).T)    
    voronoi_plot_2d(vor,show_vertices=False,ax=ax,point_size=0)
    # colorize
    for point in vor.point_region:
        if point != vor.point_region.max():
            if -1 not in vor.regions[vor.point_region[point]]:
                polygon = [vor.vertices[i] for i in vor.regions[vor.point_region[point]]]
                
                current_color = paint_colors[int(paint_pour.color[np.where(paint_pour.x == vor.points[point][0])[0][0]])]
              
                ax.fill(*zip(*polygon),current_color)
                
    # for i,region in enumerate(vor.regions):
    #     if not -1 in region:
    #         polygon = [vor.vertices[i] for i in region]
    #         # print('i = ',i)
    #         # print('point number is ',vor.)
    #         # print('y-coordinate is ')
    #         current_point = vor.point_region[i]
    #         current_color = paint_colors[int(paint_pour.color[np.where(paint_pour.x == vor.points[current_point][0])[0][0]])]
    #         ax.fill(*zip(*polygon),current_color)


#%%
r = np.linspace(0.01,0.1,100000)
sigma = 0.05
force = 6*sigma**6/r** - 12*sigma**12/r**7
fig,ax = plt.subplots(1,figsize=(6,5))
ax.plot(r,np.log10(force))
# ax.set_ylim(-1,1)
