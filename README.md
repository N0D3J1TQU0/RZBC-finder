# RZBC finder

#Red sequence-redshift Binned method Correlation finder

#IMPORTS USED - install via pip if not available

import os

import astropy.stats as st

import matplotlib.pyplot as plt

from astropy.table import Table, vstack, join, setdiff, unique


import astropy.units as u

import numpy as np

import math

from scipy.stats import chisquare#, binned_statistic_2d

from scipy.optimize import curve_fit

from scipy import signal

from astropy.coordinates import SkyCoord

from lmfit import Model

import warnings

from astropy.cosmology import FlatLambdaCDM

cosmo_boc = FlatLambdaCDM(H0=68.3,Om0=0.299) #Bocquet et al. 2015 cosmology

cosmo_ero = FlatLambdaCDM(H0=70.0,Om0=0.3) #eROSITA cosmology

=========================================================

#To run get inside SPT-ECS folder and prompt "run ../photo_z_finder08.py"

#This will process the .csv catalogs for each cluster stated in the target table "tab"
![alt text](https://github.com/N0D3J1TQU0/RZBC-finder/blob/main/SPT-ECS/fullplot.jpg)

# Example cluster J1131-1955
![alt text](https://github.com/N0D3J1TQU0/RZBC-finder/blob/main/SPT-ECS/J1131-1955/J1131-1955_z1_chi2.jpg)
![alt text](https://github.com/N0D3J1TQU0/RZBC-finder/blob/main/SPT-ECS/J1131-1955/J1131-1955_z1_cmd.jpg)
![alt text](https://github.com/N0D3J1TQU0/RZBC-finder/blob/main/SPT-ECS/J1131-1955/gr_overdensity.jpg)

