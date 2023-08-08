# Red-z-bfinder

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

To run get inside SPT-ECS folder and prompt "run ../photo_z_finder08.py"
