from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import math

def plotFresnelReflectance(n_t, n_i):
    
    max_val  = 91 
    degrees  = np.arange(max_val)
    par_ref  = np.empty(max_val, dtype=np.float64)
    ort_ref  = np.empty(max_val, dtype=np.float64)

    for i in range(max_val):
        par_ref[i], ort_ref[i] = unpolarizedReflectance(n_t, n_i, i)
 
    brewster = np.degrees(np.arctan(n_i / n_t))
    critical = np.degrees(np.arcsin(n_i / n_t))

    # Start plotting
    plt.xlim([0,90])
    plt.xlabel('Angle of incidence (degrees)')
    plt.ylabel('Reflection coefficient (%)')
    plt.plot(degrees, par_ref, c='r', label='parallel reflectance')
    plt.plot(degrees, ort_ref, c='b', label='orthogonal reflectance')
    
    # One or the other
    plotBrewster(brewster)
   # plotCritical(critical)
    plt.legend(loc='upper left')

    plt.show()
  
def plotBrewster(brewster):
    plt.axvline(brewster, color='black',
                label='Brewster angle = ' + '{0:.2f}'.format(brewster),
                linestyle='dashed')
    plt.text(brewster + 1 , 0.5,'Brewster angle',rotation=90, fontsize=15)

def plotCritical(critical): 
    plt.axvline(critical, color='green',
                label='Critical angel = ' + '{0:.2f}'.format(critical),
                linestyle='dashed')
    plt.text(critical + 1 , 0.5,'Critical angle',rotation=90, fontsize=15)

def unpolarizedReflectance(n_t, n_i, angle_i):
    theta_t = np.arcsin((n_t/n_i)*np.sin(np.radians(angle_i)))
    if math.isnan(theta_t):
        print(angle_i)
    
    theta_i = np.radians(angle_i)
    ref_par = abs( (n_t*np.cos(theta_i) - n_i*np.cos(theta_t)) /
                   (n_t*np.cos(theta_i) + n_i*np.cos(theta_t)) )**2

    ref_ort = abs( (n_i*np.cos(theta_i) - n_t*np.cos(theta_t)) /
                   (n_i*np.cos(theta_i) + n_t*np.cos(theta_t)) )**2
    return ref_par, ref_ort

def schlick(n_t, n_i):
    
    max_val = 91
    degrees = np.arange(max_val)
    f_r_approx = np.empty(max_val, dtype=np.float64) 

    R0, _ = unpolarizedReflectance(n_t, n_i, 0) # ignore second value because they are identical
    normal = np.array([1,0])
    for i in range(max_val):
        half_vector_angle = (90 - i)/2
        h = createRotatedVector(np.radians(half_vector_angle), normal)
        v = createRotatedVector(np.radians(i), normal)
        cos_theta = np.dot(h,v)
        fresnel_reflectance = R0 + (1 - R0)*(1 - cos_theta)**5
        f_r_approx[i] = fresnel_reflectance

    # Start plotting
    plt.xlim([0,90])
    plt.xlabel('Angle of incidence (degrees)')
    plt.ylabel('Reflection coefficient (%)')
    plt.plot(degrees, f_r_approx, c='g', label='Schlick approximation')
    plt.legend(loc='upper left')
    plt.show()

def createRotatedVector(a, normal):
    rotation_matrix = np.array( [ [np.cos(a), np.sin(a)],[-np.sin(a), np.cos(a)]] )
    return (normal[0] * np.cos(a) + normal[1] * np.sin(a), -normal[0] * np.sin(a) + normal[1]*np.cos(a))




# Part 1
#plotFresnelReflectance(1.4, 1.0)

# Part 2
#plotFresnelReflectance(1.0, 1.4)

# Part 3 Slick approximation
#schlick(1.0, 1.4)
