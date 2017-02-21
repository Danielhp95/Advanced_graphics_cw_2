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
    critical = np.degrees(np.arccos(n_t / n_i))
    # Start plotting
    plt.xlim([0,90])
    plt.xlabel('Angle of incidence (degrees)')
    plt.ylabel('Reflection coefficient (%)')
    plt.plot(degrees, par_ref, c='r', label='parallel reflectance')
    plt.plot(degrees, ort_ref, c='b', label='orthogonal reflectance')
    brewster_label = 'Brewster angel =' + str(brewster)
    plt.axvline(brewster, color='black',
                label='Brewster angel = ' + '{0:.2f}'.format(brewster),
                linestyle='dashed')
    plt.axvline(critical, color='green',
                label='Critical angel = ' + '{0:.2f}'.format(critical),
                linestyle='dashed')
    #plt.legend(loc='upper left')
    plt.legend(loc='upper right', bbox_to_anchor=(0.5, 0.90),
          ncol=3, fancybox=True, shadow=True)
    plt.show()
    

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
    
# Part 1
plotFresnelReflectance(1.4, 1.0)

# Part 2
#plotFresnelReflectance(1.0, 1.4)

# Part 3 Slick approximation
