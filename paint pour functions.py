from numba import njit

#Define a function for interpolating between two points, which we do a lot here. This is a convenient one because it doesn't have "kinks" at the endpoints like a linear interpolation function would.
#https://en.wikipedia.org/wiki/Smoothstep
@njit(parallel=True,fastmath=True)   #Like magic, the @njit bit makes the below function run faster by converting it into machine code.
def smootherstep_function(x):
    return 6*x**5-15*x**4+10*x**3
