
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

def precalc(field, wavelength=0.532):
    """
    Precalculate some constants for propagating field for faster computations.

    Input:
        field : Complex valued optical field.
        wavelength : in um
    Output:
        K : Transformation matrix
        C : Circular disk

    """
    
    k = 2* np.pi / wavelength # Wavevector
    
    yr, xr = field.real.shape

    x = 2 * np.pi/(.114) * np.arange(-(xr/2-1/2), (xr/2 + 1/2), 1)/xr
    y = 2 * np.pi/(.114) * np.arange(-(yr/2-1/2), (yr/2 + 1/2), 1)/yr

    KXk, KYk = np.meshgrid(x, y)
    K = np.real(np.sqrt(np.array(1 -(KXk/k)**2 - (KYk/k)**2 , dtype = np.complex64)))
    
    #Create a circular disk here.
    C = np.fft.fftshift(((KXk/k)**2 + (KYk/k)**2 < 1)*1.0)

    return x, y, K, C 

def refocus_field_z(field, z_prop, padding = 0, wavelength = 0.532):
    """
    Function for refocusing field.

    Input:
        field : Complex valued optical field.
        z_prop : float or integer of amount of z propagation.
        padding : Pad image by this amount in all directions
    Output: 
         Refocused field

    """

    if not np.iscomplexobj(field):
        field = field[..., 0] + 1j*field[..., 1]
        
    if padding > 0:
        field = np.pad(field, ((padding, padding), (padding, padding)), mode = 'reflect')

    k = 2 * np.pi / wavelength # Wavevector
    _, _ , K, C = precalc(field, wavelength)
    
    z = [z_prop]
    
    #Get matrix for propagation.
    Tz = precalc_Tz(k, z, K, C)    
    
    #Fourier transform of field
    f1 = np.fft.fft2(field)
    
    #Propagate f1 by z_prop
    refocused = np.array(np.fft.ifft2(Tz*f1), dtype = np.complex64)
    
    if padding > 0:
        refocused = refocused[:, padding:-padding, padding:-padding]

    return np.squeeze(refocused)


def refocus_field(field, steps=51, interval = [-10, 10], padding = 0, wavelength = 0.532):
    """
    Function for refocusing field.

    Input:
        field : Complex valued optical field.
        Steps : N progation steps, equally ditributed in interval.
        Interval : Refocusing interval
        padding : Pad image by this amount in all directions

    Output: 
        A stack of propagated_fields

    """
    if not np.iscomplexobj(field):
        field = field[..., 0] + 1j*field[..., 1]

    if padding > 0:
        field = np.pad(field, ((padding, padding), (padding, padding)), mode = 'reflect')

    k = 2 * np.pi / wavelength
    _, _, K, C = precalc(field)
    
    zv = np.linspace(interval[0], interval[1], steps)
    
    #Get matrix for propagation.
    Tz = precalc_Tz(k, zv, K, C)    
    
    #Fourier transform
    f1 = np.fft.fft2(field)
    
    #Stack of different refocused images.
    refocused =  np.array([np.fft.ifft2(Tz[i]*f1) for i in range(steps)], dtype = np.complex64)
    
    if padding > 0:
        refocused = refocused[:, padding:-padding, padding:-padding]

    return refocused
    
def find_focus_field(field, steps=51, interval = [-10, 10], m = 'fft', padding=0, use_max_real = False, bbox = []):
    """
    Find focus of optical field.

    Input:
        field : Complex valued optical field.
        Steps : N progation steps, equally ditributed in interval.
        Interval : Refocusing interval
        m : Evaluation criterion, 'abs', 'sobel', or 'adjescent', 'tamura'
        padding : Pad image by this amount in all directions
        use_max_real : simply take the max value of the real part of the optical field and extract an roi around that point.
        bbox : evaluate the focus in the region of the bounding box.

    Output: 
        z_prop : float optimal z propagation

    """

    if not np.iscomplexobj(field):
        field = field[..., 0] + 1j*field[..., 1]

    #Interval to propagate within
    zv = np.linspace(interval[0], interval[1], steps)
    
    #Predefined bbox
    if len(bbox)==4 and use_max_real == False:
        a = refocus_field(field[bbox[0]:bbox[1], bbox[2]:bbox[3]], 
                          steps=steps, 
                          interval = interval,
                          padding=padding)
        
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
                          interval = interval,
                          padding=padding)  
        
    else:
        a = refocus_field(field, 
                          steps=steps, 
                          interval = interval,
                          padding=padding)   
    
    #Some ways of finding criterions.
    if m == 'fft':
        criterion = [-np.std((np.fft.fft2(im)).imag) for im in field]    
    elif m == 'abs':
        criterion = [np.std(np.abs(im)) for im in field]
    elif m == 'maxabs':
        criterion = [np.max(im.real) + np.max(im.imag) for im in field]
    elif m == 'sobel':
        criterion = [-(np.std(ndimage.sobel(im.real)) + np.std(ndimage.sobel(im.imag))) for im in field]
    elif m == 'adjescent':
        n_rows = int(0.5 * field.shape[0])
        criterion = [adjescent_pixels(im.imag, n_rows) for im in field]
    elif m == 'tamura':
        ### OBS quite slow for big images...
        criterion = [Tamura_coefficient(SoG(im)) for im in field]
    elif m =='classic':
        abssqim = [np.abs(im)**2 for im in field]
        criterion = [np.sum(np.abs(ab - np.median(ab))**2) for ab in abssqim]

    #idx of max 
    idx = np.argmax(criterion)
    
    #How much propagation in z
    z_focus = zv[idx]
    
    return z_focus, criterion

def find_focus_field_stack(field,  m = 'fft', padmin = 6):
    """
    Find focus of optical field and return the "most" focused image in stack.

    Input:
        field : Complex valued optical field as a stack.
        m : Evaluation criterion, 'abs', 'sobel', or 'adjescent', 'tamura'
        use_max_real : simply take the max value of the real part of the optical field and extract an roi around that point.
    Output: 
        z_prop : Return focused image

    """
    F = field.copy()

    #Make ROI smaller
    if padmin < int(field.shape[1] / 2):
        field = field[:, padmin:-padmin, padmin:-padmin]

    #Make it complex if not complex
    if not np.iscomplexobj(field):
        field = field[..., 0] + 1j*field[..., 1]

    #Calculate criterions.
    if m == 'fft':
        criterion = [-np.std((np.fft.fft2(im)).imag) for im in field]  
    elif m == 'abs':
        criterion = [np.std(np.abs(im)) for im in field]
    elif m == 'maxabs':
        criterion = [np.max(im.real) + np.max(im.imag) for im in field]
    elif m == 'sobel':
        criterion = [-(np.std(ndimage.sobel(im.real)) + np.std(ndimage.sobel(im.imag))) for im in field]
    elif m == 'adjescent':
        n_rows = int(0.5 * field.shape[0])
        criterion = [adjescent_pixels(im, n_rows) for im in field]
    elif m == 'tamura':
        ### OBS quite slow for big images...
        criterion = [Tamura_coefficient(SoG(im)) for im in field]
    elif m =='classic':
        abssqim = [np.abs(im)**2 for im in field]
        criterion = [np.sum(np.abs(ab - np.median(ab))**2) for ab in abssqim]

    #idx of max 
    idx = np.argmax(criterion)
    
    #Return the max.
    focused_field = F[idx]
    
    return focused_field

def Tamura_coefficient(vec):

    """
    Compute the Tamura coefficient.
    """
    return np.sqrt(np.std(vec) / np.mean(vec))

def SoG(field):

    """
    Sparsity of the Gradient (SoG)

    Use this for a small region, otherwise it is slow.

    Input:
        Complex valued optical field.
    Output:
        Sparsity of the gradient, to be used to calculate in focus.
    """
    grad_x = np.abs(field[1:, 1:] - field[1:, :-1])**2
    grad_y = np.abs(field[1:, 1:] - field[:-1, 1:])**2

    res = np.sqrt(grad_x + grad_y)

    #res = []
    #for i in range(1, row):
    #    for j in range(1, col):

     #       res.append(
     #           np.sqrt(
     #               np.abs(field[i, j] - field[i, j-1])**2 + \
     #               np.abs(field[i, j] - field[i-1, j])**2
     #               )
     #       )
    
    return res

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