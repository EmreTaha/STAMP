import numpy as np

from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares


def apply_flatten(img, lyr, level):
    
    image=np.transpose(img, axes=[2, 0, 1])# Shape of image: slc, ht, cols(across A-scans)

    layers=np.transpose(lyr)
    layers=np.expand_dims(layers, axis=1)# Shape of layers: slc by 1 by A-scans
        
    shape = image.shape
        
    dy = np.tile(layers.astype(np.int64) - level,(1,image.shape[1],1))# layers=512 by 128
    
    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
        
    indices = np.reshape(x, (-1, 1)), np.reshape(y + dy, (-1, 1)), np.reshape(z, (-1, 1))
    
    out=map_coordinates(image, indices, order=0, mode='constant').reshape(shape)
    out=np.transpose(out, axes=[1, 2, 0])
    
    return out, dy

###Arunava's code
def paraBolEqn(data,e,f,g,h,i,j):
    x,y = data
    #return -(e*(x**2)  +f*(x*y) + g*(y**2) + h*x + i*y) + j
    #return -(((x-b)/a)**2+((y-d)/c)**2)+e
    return (e*(x**2)  +f*(x*y) + g*(y**2) + h*x + i*y) + j
def fun(x, t, y):
    return paraBolEqn(t,x[0],x[1],x[2],x[3], x[4], x[5])-y
def fit_paraboloid(lyr):
    # Z-> the height along each A-scan
    # X->A-scan/columns
    # Y-> B-scans/Slice nos.
    doex=np.arange(0,lyr.shape[0])
    doex=np.expand_dims(doex,axis=1)
    doex=np.tile(doex, (1,lyr.shape[1]))
    doey=np.arange(0, lyr.shape[1])
    doey=np.expand_dims(doey, axis=0)
    doey=np.tile(doey, (lyr.shape[0],1))
    # values inside lyr is : Z, size is Ascans by slice
    doex=doex.flatten()
    doey=doey.flatten()
    doez=lyr.flatten()
    # mean shift
    doex=doex-np.mean(doex)
    doey=doey-np.mean(doey)
    z_mn=np.mean(doez)
    doez=doez-z_mn
    x0 = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    res_robust = least_squares(fun, x0, loss='soft_l1', f_scale=1.0, args=((doex,doey), doez))
    #res_robust = least_squares(fun, x0, loss='cauchy', f_scale=1.0, args=((doex,doey), doez))
    popt=res_robust.x
    z_fit=paraBolEqn((doex,doey),popt[0],popt[1],popt[2],popt[3], popt[4], popt[5])
    ######## A second fitting to remove outliers
    err=(z_fit-doez)*(z_fit-doez)
    idx=np.where(err<np.median(err))
    res_robust = least_squares(fun, x0, loss='soft_l1', f_scale=1.0, args=((doex[idx],doey[idx]), doez[idx]))
    popt=res_robust.x
    z_fit=paraBolEqn((doex,doey),popt[0],popt[1],popt[2],popt[3], popt[4], popt[5])
    ###
    z_fit=z_fit+z_mn
    lyr_fit=z_fit.reshape(lyr.shape)
    return lyr_fit


def tile_tensors(images,hrow,hcol,img_shape):
    '''Gets an array of Gray image tensors and return them in a single image grid'''
    images = (images.reshape(hrow, hcol, img_shape, img_shape)
              .swapaxes(1,2)
              .reshape(1,hrow*img_shape, hcol*img_shape))
    return images