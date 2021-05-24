
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 15:49:37 2020

@author: Mohamed Nazmi Idros 
"""

from skimage import io, feature
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
#%% Elippse Function 

def func_ell (p,a,b,c,d,f,g):
    """
    Compute the unknown the variable in the elippse
    Param: 
        p = an array [x,y]
        a,b,c,d,f,g = unknown variable in the elippse
        
    return:
        y**2
    """
    x,y = p
    return -a*(x**2)/c-(2*b)*x*y/c-(2*d)*x/c-(2*f)*y/c - (g/c)
    
def ell_fit (xx,yy):
    """
    Compute the elippse fitting through curve_fit function 
    (ref:   #https://mathworld.wolfram.com/Ellipse.html)
    
    Params: 
        xx, yy : an array of x and y cooordinate of the droplet 
        
    return: 
        ell_fitting: Elippse fitting 
        xc, yc = center of the elippse 
        a4 = major semi-axis length
        b4 = minor semi-axis length 
        z = the unknown variable in elippse fitting 
    """
    
    ymax = np.max(yy)
    z,j = curve_fit (func_ell,[xx,yy], yy**2) 
    a,b,c,d,f,g = z
    xc = ((c*d)- (b*f))/((b**2)-(a*c))
    yc = ((a*f)-(b*d))/((b**2)-(a*c))
    
    
    a1 = 2*((a*(f**2))+(c*(d**2))+(g*(b**2))-(2*b*d*f)-(a*c*g))
    a2 = ((b**2)-(a*c))
    a3 = np.sqrt(((a-c)**2)+4*(b**2))-(a+c)
    a4 = np.sqrt(a1/(a2*a3))
       
    b3 = -np.sqrt(((a-c)**2)+4*(b**2))-(a+c)
    b4 = np.sqrt(a1/(a2*b3))

    Xpoints = np.arange(xc-a4,xc+a4,0.1)
    
    if yc < ymax: 
        Ypoints = np.sqrt((1-(((Xpoints-xc)**2)/(a4**2)))*(b4**2)) + yc
    else:
        Ypoints = -np.sqrt((1-(((Xpoints-xc)**2)/(a4**2)))*(b4**2)) + yc
    
    ell_fitting = np.transpose([Xpoints,Ypoints])

    return ell_fitting, xc, yc , z, a4, b4

def ode_ell (x,y,a,b,c,d,f):
    """
    Compute the slope of a coordinate in an elippse 
    Params:
        x,y = the coordinate of desire slop 
        a,b,c,d,f = unknown variable in elippse fitting 
        
    return: 
        slop of at the coordinate provided 
    """
    return -((a*x)+(b*y)+d)/((c*y)+(b*x)+f)

#%% Circle Function 
def func (p,k,l,c):
    """
    Compute the unknown the variable in the circle
    Param: 
        p = an array [x,y]
        a,b,c,d,f,g = unknown variable in the circle
        
    return:
        y**2
    """
    x,y = p
    return k*x+l*y + c 

def circle_fit (xx,yy):
    """
    Compute the circle fitting through curve_fit function 
    
    Params: 
        xx, yy : an array of x and y cooordinate of the droplet 
        
    return: 
        ell_fitting: circle fitting 
        xc, yc = center of the circle
        r = radius of the circle  
    """
    
    ymax=np.max(yy)
    
    z,j=curve_fit(func,[xx,yy],xx**2 + yy**2)
    xc,yc = z[:2]/2
    r = np.sqrt(z[2]+xc**2 + yc**2)
    
    points = np.arange(xc-r,xc+r,0.1)
    v = (r**2)-(np.power((points - xc), 2))
    
    if yc < ymax :
        vv = np.sqrt(v[v>0])+yc
    else:
        vv = -np.sqrt(v[v>0])+yc
    
    index = np.where(v>0) 
    
    
    circle_fitting = np.transpose([points[index],vv])
    
    return circle_fitting, xc, yc, r

def ode(x,r,xc):    
    """
    Compute the slope of a coordinate in an circle 
    Params:
        x = the coordinate of desire slop 
        r = radius of the slope 
        xc = x coordinate of the center 
        
    return: 
        slop of at the coordinate provided 
    """
    return -(r**2 - (x - xc)**2)**(-0.5)*(-1.0*x + 1.0*xc)
    #%% 
def CA_analysis (name): 
    """
    The main function which will import image, convert image based on edges, 
    classified the droplet and baseline data, perform circle and elippse fitting and 
    calculate the contact angle based on circle and elippse fitting 
    
    Params:
        name = image file name 
        
    Return: 
        al_c = The contact angle for the droplet from left side obtain thorugh circle fitting 
        ar_c = The contact angle for the droplet from right side obtain thorugh circle fitting
        av_c = The avarage contact angle for the droplet from both sides obtain thorugh circle fitting
        al_e = The contact angle for the droplet from left side obtain thorugh elippse fitting
        ar_e = The contact angle for the droplet from right side obtain thorugh elippse fitting 
        av_e = The average contact angle for the droplet from both sides obtain thorugh elippse fitting 
    """
    
    #Importing image
    im=io.imread(name, as_gray=True)
    imp=feature.canny(im, sigma=2.8, low_threshold = 0, high_threshold = 0.44)
    impp=1-imp
    Y,X= np.where(impp == 0)
    
    
    #Assume the droplet region 
    Xmin = np.argmin(X)
    Y0 = Y[Xmin]
    Xmax = np.argmax(X)
    Yf = Y[Xmax]
    
    index = np.where((Y < np.min([Y0,Yf])*0.99))
    x , y = [X[index],Y[index]]
    
    droplet = np.transpose(np.array(np.vstack([x,y])))
    
    #removing the additional circle if there are two circle in the data 
    cir_fit, xc, yc, r  = circle_fit(x,y)
    points = np.array([(x, y) for x, y in droplet if (x - xc ) ** 2 + (y - yc ) ** 2 >= r ** 2])
      
    #Baseline Fiting
    indexB = np.where((Y > np.min([Y0,Yf])))
    xb , yb = [X[indexB],Y[indexB]]
    p, V = np.polyfit(xb,yb, 1, cov=True)
    poly1d_fn = np.poly1d(p) 
    
    
    #Circle Fiting 
    cir_fit, xc, yc, r  = circle_fit(points[:,0], points[:,1])
    
    #Intersection 
    poly1d_fn = np.poly1d(p)
    lin_fit = poly1d_fn(cir_fit[:,0])
    idx = np.argwhere(np.diff(np.sign(cir_fit[:,1]-lin_fit))).flatten()
    intersection = cir_fit[idx[0]]
    intersection = np.vstack([intersection, cir_fit[idx[-1]]])
    
    #Circle Contact angle 
    x0 = intersection[0,0]
    x1 = intersection[1,0]
    
    ymax = np.max(points[:,1])
    if yc < ymax :
        scirclel = ode(x0, r,xc)
        m = p[0]
        al_c = 180-(-np.arctan(scirclel)*(180/np.pi) + np.arctan(m)*180/np.pi)
    
        scircler = ode(x1, r,xc)
        m = p[0]
        ar_c=  180+(-np.arctan(scircler)*(180/np.pi) + np.arctan(m)*180/np.pi)
    else:
        scirclel = ode(x0, r,xc)
        m = p[0]
        al_c = -np.arctan(scirclel)*(180/np.pi)-np.arctan(m)*180/np.pi
    
        scircler = ode(x1, r,xc)
        m = p[0]
        ar_c= np.arctan(scircler)*(180/np.pi)
        
    av_c = (ar_c+al_c)/2
    
    plt.figure()
    plt.imshow(im,cmap=plt.cm.gray, origin= 'upper')
    plt.plot(cir_fit[:,0], cir_fit[:,1])
    plt.plot(cir_fit[:,0], poly1d_fn(cir_fit[:,0]) )
    plt.scatter(intersection[:,0],intersection[:,1])
    plt.xlim([0,len(im[0,:])])
    plt.ylim([len(im[:,0]), 0])
    plt.savefig('processdata\\' + name.replace('rawdata\\','circlefitted').replace('bmp','jpg').replace('tif','jpg'))
    
    
    #Elippse Fiting 
    fit_ell, xc , yc, z, a4, b4 = ell_fit(points[:,0], points[:,1])
    a,b,c,d,f,g = z
  
    
    nan = np.argwhere(np.isnan(fit_ell))
    if len (nan) != 0:
        fit_ell = np.delete(fit_ell,nan,0)
    #Intersection
    
    lin_fit = poly1d_fn(fit_ell[:,0])
    idx = np.argwhere(np.diff(np.sign(np.round(fit_ell[:,1]-lin_fit)))).flatten()
    intersection = fit_ell[idx[0]]
    intersection = np.vstack([intersection, fit_ell[idx[-1]]])
   
    #Elippse contact angle 
    x0 = intersection[0,0]
    y0 = intersection[0,1]
    x1 = intersection[1,0]
    y1 = intersection[1,1]
    
    ymax = np.max(y)
    
    if yc < ymax:
        scirclel = ode_ell(x0,y0,a,b,c,d,f)
        m = p[0]
        al_e = 180 - (np.arctan(scirclel)*(180/np.pi) + np.arctan(m)*180/np.pi)
    
        scircler = ode_ell(x1,y1,a,b,c,d,f)
        m = p[0]
        ar_e=  180 + (np.arctan(scircler)*(180/np.pi) - np.arctan(m)*180/np.pi)
    else:
        scirclel = ode_ell(x0,y0,a,b,c,d,f)
        m = p[0]
        al_e = -np.arctan(scirclel)*(180/np.pi)+np.arctan(m)*180/np.pi
    
        scircler = ode_ell(x1,y1,a,b,c,d,f)
        m = p[0]
        ar_e= np.arctan(scircler)*(180/np.pi)
        
    av_e = (ar_e+al_e)/2
    
    plt.figure()
    plt.imshow(im,cmap=plt.cm.gray, origin= 'upper')
    plt.plot(fit_ell[:,0],fit_ell[:,1], 'm')
    plt.plot(fit_ell[:,0], poly1d_fn(fit_ell[:,0]), '-r')
    plt.scatter(intersection[:,0],intersection[:,1])
    plt.xlim([0,len(im[0,:])])
    plt.ylim([len(im[:,0]), 0])
    plt.savefig('processdata\\' + name.replace('rawdata\\','elippsefitted').replace('bmp','jpg').replace('tif','jpg'))
    
    return al_c,ar_c, av_c, al_e, ar_e, av_e
