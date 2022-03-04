
"""
Script containing functions for propagating an optical field.

Created on Wed Mar 03 09:14:39 2022

@author: Fredrik Skärberg // fredrik.skarberg@physics.gu.se

"""

import numpy as np
from scipy import ndimage

def precalc_Tz (k, zv, K, C):
    """
    Computes the T_z matrix used for propagating an optical field.

    Input:
        k : Wavevector
        zv : List of z propagation steps
        K : Transformation matrix
        C : Circular disk
    Output:
        z-propagator matrix

    """

    return [C*np.fft.fftshift(np.exp(k * 1j*z*(K-1))) for z in zv]

def precalc(field):
    """
    Precalculate some constants for propagating field for faster computations.

    Input:
        field : Complex valued optical field.
    Output:
        K : Transformation matrix
        C : Circular disk

    """
    
    k = 2* np.pi / 0.633 # Wavevector
    
    yr, xr = field.real.shape

    x = 2 * np.pi/(.114) * np.arange(-(xr/2-1/2), (xr/2 + 1/2), 1)/xr
    y = 2 * np.pi/(.114) * np.arange(-(yr/2-1/2), (yr/2 + 1/2), 1)/yr

    KXk, KYk = np.meshgrid(x, y)
    K = np.real(np.sqrt(np.array(1 -(KXk/k)**2 - (KYk/k)**2 , dtype = np.complex64)))
    
    #Create a circular disk here.
    C = np.fft.fftshift(((KXk/k)**2 + (KYk/k)**2 < 1)*1.0)

    return x, y, K, C 

def refocus_field_z(field, z_prop):
    """
    Function for refocusing field.

    Input:
        field : Complex valued optical field.
        z_prop : float or integer of amount of z propagation.

    Output: 
         Refocused field

    """

    k = 2 * np.pi / 0.633 # Wavevector
    _, _ , K, C = precalc(field)
    
    z = [z_prop]
    
    #Get matrix for propagation.
    Tz = precalc_Tz(k, z, K, C)    
    
    #Fourier transform of field
    f1 = np.fft.fft2(field)
    
    #Propagate f1 by z_prop
    refocused = np.array(np.fft.ifft2(Tz*f1), dtype = np.complex64)
    
    return np.squeeze(refocused)


def refocus_field(field, steps=51, interval = [-10, 10]):
    """
    Function for refocusing field.

    Input:
        field : Complex valued optical field.
        Steps : N progation steps, equally ditributed in interval.
        Interval : Refocusing interval

    Output: 
        A stack of propagated_fields

    """
    
    k = 2 * np.pi / 0.633
    _, _, K, C = precalc(field)
    
    zv = np.linspace(interval[0], interval[1], steps)
    
    #Get matrix for propagation.
    Tz = precalc_Tz(k, zv, K, C)    
    
    #Fourier transform
    f1 = np.fft.fft2(field)
    
    #Stack of different refocused images.
    refocused =  np.array([np.fft.ifft2(Tz[i]*f1) for i in range(steps)], dtype = np.complex64)
    
    return refocused
    
def adjescent_pixels(image, n_rows):
    """
    Compute the sum of absolute value between pixels in image.

    Input:
        Image : Complex valued image of optical field. (must not be complex, works anyways...)
        n_rows : How many rows to consider
    """
    abssum = 0
    for i in range(n_rows-1):
        abssum += np.sum(np.abs(image[i, :].real - image[i+1, :].real))
        
    return float(abssum / n_rows)  
    
def find_focus_field(field, steps=51, interval = [-10, 10], m = 'abs', use_max_real = False, bbox = []):
    """
    Find focus of optical field.

    Input:
        field : Complex valued optical field.
        Steps : N progation steps, equally ditributed in interval.
        Interval : Refocusing interval
        m : Evaluation criterion, 'abs', 'sobel', or 'adjescent'
        use_max_real : simply take the max value of the real part of the optical field and extract an roi around that point.
        bbox : evaluate the focus in the region of the bounding box.

    Output: 
        z_prop : float optimal z propagation

    """

    zv = np.linspace(interval[0], interval[1], steps)
    
    #Predefined bbox
    if len(bbox)==4 and use_max_real == False:
        a = refocus_field(field[bbox[0]:bbox[1], bbox[2]:bbox[3]], 
                          steps=steps, 
                          interval = interval)
        
    elif use_max_real==True:
        idx_max = np.unravel_index(np.argmax(field.real, axis=None), field.real.shape)
        padsize = 256
        
        #Use an roi where max is found
        if idx_max[0]-padsize > 0 and idx_max[0]+padsize<field.shape[0] and idx_max[1]-padsize > 0 and idx_max[1]+padsize<field.shape[1]:
            field = field[idx_max[0]-padsize : idx_max[0]+padsize, idx_max[1]-padsize : idx_max[1]+padsize]
        
        #Use center...
        else:
            field = field[int(field.shape[0]/2 - padsize) : int(field.shape[0]/2 + padsize), int(field.shape[1]/2 - padsize) : int(field.shape[1]/2 + padsize)]     
        
        a = refocus_field(field, 
                          steps=steps, 
                          interval = interval)  
        
    else:
        a = refocus_field(field, 
                          steps=steps, 
                          interval = interval)   
    
    #standard deviation of sobelfiltered image. Other criterions can be used aswell
    if m == 'sobel':
        criterion = [np.std(ndimage.sobel(im.real)) + np.std(ndimage.sobel(im.imag))  for im in a]
    elif m == 'abs':
        criterion = [np.std(np.abs(im))  for im in a]
    elif m == 'adjescent':
        criterion = [adjescent_pixels(im, n_rows = 1024)  for im in a]
        
    #idx of max 
    idx = np.argmax(criterion)
    
    #How much propagation in z
    z_focus = zv[idx]
    
    return z_focus
