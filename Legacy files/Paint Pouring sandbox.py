import matplotlib.pyplot as plt
import numpy as np
import time
import os
from scipy import stats
import colorsys
import matplotlib.colors as mcolors
import matplotlib.animation as animation
from numba import njit

#TO DO
#Add "maximum influence radius" between particles so they are only repulsed by particles within a certain distance
#Delete particles that fall off the canvas

plt.close('all')

num_frames = 10
droplets_per_point = 5
pour_speed = 0.01 #normalized distance to move per timestep
starting_coords = [0.1,0.5]
dt = 0.2
# drop_step = 0.05
save_animation = 0
damping_factor = 0.5
static_friction_force = 1000

#initialize the plot
plt.close('all')
fig,ax = plt.subplots(1,1,figsize=(7,6))               #Change the figure size here
ax.set_aspect('equal')

# coords_x = [drop_points_x[0]]
# coords_y = [drop_points_y[0]]
# coords = np.array([drop_points_x[0],drop_points_y[0]])
coords = np.empty((droplets_per_point,2))
coords[:,0] = np.random.uniform(0,0.01,droplets_per_point) + starting_coords[0]
coords[:,1] = np.random.uniform(0,0.01,droplets_per_point) + starting_coords[1]
v_x = np.zeros(droplets_per_point)
v_y = np.zeros(droplets_per_point)
v_mag = np.zeros(droplets_per_point)

v_max = 0.05


# intialize the line objects
scatter1 = ax.scatter([], [],s=15.0,cmap='coolwarm',label='particle positions')
linewidth = 1.5
line1, = ax.plot([0,0], [1,0], lw=linewidth, color='black')  #[xi,xf],[yi,yf]
line2, = ax.plot([0,1], [1,1], lw=linewidth, color='black')
line3, = ax.plot([1,1], [1,0], lw=linewidth, color='black')
line4, = ax.plot([0,1], [0,0], lw=linewidth, color='black')

#norm = plt.Normalize(min(v_total),max(v_total))
lines = [scatter1,line1,line2,line3,line4]



#Set limits and labels for the axes
fontsize = 14
ax.set_xlim(-0.1,1.1)
ax.set_ylim(-0.1,1.1)
ax.set_title('Charged particles',fontsize=fontsize)
ax.tick_params(axis='both',labelsize=fontsize/1.3)
# cbar = fig.colorbar(scatter1, format='%.0e',extend='max',ticks=[0,5e-4])
# cbar.set_ticklabels(['Slow', 'Fast'])  # horizontal colorbar
# cbar.set_label('Particle Velocity (arbitrary units)', fontsize=fontsize)
plt.tight_layout()

@njit    
def particle_forces2(coords,npoints,particle):
    dx_from_others = coords[:,0] - coords[particle,0]
    dy_from_others = coords[:,1] - coords[particle,1]
    r_from_others = np.sqrt(dx_from_others**2 + dy_from_others**2)
    r_from_others[np.where(r_from_others == 0)[0]] = np.inf    #set the distance from itself = infinity
    F_from_others_x = -1*dx_from_others/r_from_others**2
    F_from_others_y = -1*dy_from_others/r_from_others**2
    result = np.array([np.sum(F_from_others_x),np.sum(F_from_others_y)])
    return result

@njit
def update_velocities(v_x_temp,v_y_temp,F_total,mass):
    v_x_temp += F_total[0]/mass
    v_y_temp += F_total[1]/mass
    return v_x_temp,v_y_temp

def animate(i):
    global coords
    global v_x
    global v_y
    global v_mag
    
    if i > 0:
        #Math that gets recalculated each iteration
        print('Iteration: '+str(i))

        #calculate net force on each particle, one particle at a time
        num_particles_temp = len(coords)
        F_total_x = np.zeros(num_particles_temp)
        F_total_y = np.zeros(num_particles_temp)
        for particle in range(0,num_particles_temp):
            interparticle_forces = particle_forces2(coords,num_particles_temp ,particle) 
            if particle == 1:
                print('net interparticle forces: ',interparticle_forces)
                
            if v_mag[particle] > 0:
                F_friction_x = static_friction_force*np.sign(v_x[particle])*abs(v_x[particle]/v_mag[particle])
                F_friction_y = static_friction_force*np.sign(v_y[particle])*abs(v_y[particle]/v_mag[particle])

                if particle == 1:
                    print('net friction force (in x): ',F_friction_x)
            
            
            #TODO, sort out how to add friction forces without changing the sign of the velocity
            if v_x[particle] < 0
            
            F_total_x[particle] = interparticle_forces[0]# + F_friction_x
            F_total_y[particle] = interparticle_forces[1]# + F_friction_y
        
        #Calculate friction forces for moving particles


        # moving_particles = np.where(v_mag > 0)[0]
        # print(F_total_x[0:3])
        # print(v_mag)
        # F_total_x[moving_particles] -= static_friction_force*np.sign(v_x[moving_particles])*abs(v_x[moving_particles]/v_mag[moving_particles])
        # F_total_y[moving_particles] += static_friction_force*np.sign(v_y[moving_particles])*v_y[moving_particles]/v_mag[moving_particles]
        # print(F_total_x[0:3])
        # print('force_x after friction: ',F_total_x[0])

            # F_total_y[particle] -= static_friction_force*np.sign(v_y[particle])*v_y[particle]/v_mag[particle]
        # print(len(v_x))
        # print(len(v_mag))

        #Put all the forces together into one array
        F_total = np.array([F_total_x,F_total_y])
        # F_mag = np.sqrt(F_total_x**2+F_total_y**2)
        
        #Subtract off friction
        # F_mag
        
        
        #Subtract off friction forces
        # F_total[np.where(F_total > 0)] -= static_friction_force
        # F_total[np.where(F_total < 0)] += static_friction_force
        # stopped_particles = np.where(abs(F_total) < static_friction_force)
        # print(stopped_particles)
        # F_total[stopped_particles] = 0
        # F_total
        # print(F_total)
        #update particle velocities
        # print(v_x[0])
        v_x,v_y = update_velocities(v_x,v_y,F_total,1)
        # print(len(v_x))
        
        
        # print('Force_x on 0 is: ',F_total_x[0])
        # print('velocity_x of 0 is: ',v_x[0])

        v_mag = np.sqrt(v_x**2+v_y**2)
        v_x = np.where(abs(v_mag) > v_max,v_max*v_x/v_mag,v_x)*(1-damping_factor)
        v_y = np.where(abs(v_mag) > v_max,v_max*v_y/v_mag,v_y)*(1-damping_factor)
        v_mag = np.sqrt(v_x**2+v_y**2)
        # print(v_x[0])

        # v_x = v[0]
        # v_y = v[1]
        
        #update coordinate of each particle
        coords[:,0] += v_x*dt
        coords[:,1] += v_y*dt
        # print(coords[0,:])
        
        if i < num_frames/2:
            #Add coordinates for the next round of droplets
            x = np.random.uniform(0,0.02,droplets_per_point)+starting_coords[0]+pour_speed*i
            y = np.random.uniform(0,0.02,droplets_per_point)+starting_coords[1]
            coords = np.append(coords,np.array([x,y]).T,axis=0)
            
            #Append some zeros to the velocity arrays
            v_x = np.append(v_x,np.zeros(droplets_per_point))
            v_y = np.append(v_y,np.zeros(droplets_per_point))
            v_mag = np.append(v_mag,np.zeros(droplets_per_point))


    lines[0].set_offsets(coords)

    return lines

#Actually do the animation
anim = animation.FuncAnimation(fig, animate,repeat=False,frames=num_frames, interval=16.6667*10, blit=False)

if save_animation == 1:
    anim.save('testing.mp4', fps=30, extra_args=['-vcodec', 'libx264'],dpi=100)
