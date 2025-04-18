import numpy as np
import pylab as plt
import scipy.integrate as itg
import scipy.stats as st
import numpy.fft as ft
import numpy.random as rnd
import scipy.optimize as op
import scipy.special as sp
from astropy.io import fits

def Load_Lightcurve(fileroute,tbin,\
                        time_col=None,flux_col=None,error_col=None,extention=1,element=None):
    '''
    Loads a data lightcurve as a 'Lightcurve' object, assuming a text file
    with three columns. Can deal with headers and blank lines.
    
    inputs:
        fileroute (string)      
            - fileroute to txt file containing lightcurve data
        tbin (int)              
            - The binning interval of the lightcurve
        time_col (str, optional)
            - Time column name in fits file (if using fits file)
        flux_col (str, optional)
            - Flux column name in fits file (if using fits file)
        error_col (str, optional)
            - Flux error column name in fits file (if using fits file, and want
              errors)
        extention (int, optional)
            - Index of fits extension to use (default is 1)
        element (int,optional)
            - index of element of flux column to use, if mulitple (default None)
    outputs:
        lc (lightcurve)         
            - output lightcurve object
    '''
    two = False    
            
    if time_col or flux_col:
        if (time_col != None and flux_col != None):
            try:
                fitsFile = fits.open(fileroute)
                data = fitsFile[1].data

                time = data.field(time_col)

                if element:
                    flux = data.field(flux_col)[:,element]
                else:
                    flux = data.field(flux_col)

                if error_col != None:
                    if element:
                        error = data.field(error_col)[:,element]
                    else:
                        error = data.field(error_col)
                else:
                    two = True

            except IOError:
                print("*** LOAD FAILED ***")
                print("Input file not found")
                return
            except KeyError:
                print("*** LOAD FAILED ***")
                print("One or more fits columns not found in fits file")
                return
        else:            
            print("*** LOAD FAILED ***")
            print("time_col and flux_col must be defined to load data from fits.")
            return
    else:
        try:
            f = open(fileroute,'r')

            time,flux,error = [],[],[]
            success = False
            read = 0    
            
            for line in f:
                columns = line.split()
                    
                if len(columns) == 3:
                    t,f,e = columns

                    try:
                        time.append(float(t))
                        flux.append(float(f))
                        error.append(float(e))
                        print("one")
                        read += 1
                        success = True
                    except ValueError:
                        continue
                elif len(columns) == 2:
                    t,f = columns

                    try:
                        time.append(float(t))
                        flux.append(float(f))
                        print("two")
                        read += 1
                        success = True
                        two = True
                    except ValueError:
                        continue
            if success:
                print("Read {} lines of data".format(read))
            else:
                print("*** LOAD FAILED ***") #ERROR SOMEWHERE IN HERE
                print("Input file must be a text file with 3 columns (time,flux,error)")
                return

        except IOError:
            print("*** LOAD FAILED ***")
            print("Input file not found")
            return
       
    if two:
        lc = Lightcurve(np.array(time), np.array(flux),tbin)
    else:
        lc = Lightcurve(np.array(time), np.array(flux),tbin, np.array(error))
    return lc