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

#plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['xtick.major.size'] = 7
plt.rcParams['xtick.minor.size'] = 3
plt.rcParams['ytick.major.size'] = 7
plt.rcParams['ytick.minor.size'] = 3

#IF RUNNING FIRST TIME CHECK ALL LINES WITH THE "???" STRING
###############PRELOADS###########
#bcgz20 =  Table.read("bayliss_bcg_radec.latex",format="latex")
#wr = Table.read("wrongies.csv",format="csv")              #list of cluster i want to check spectroscopy   ???
#spectab = Table.read("megacrate.csv",format="csv")        #spectroscopic catalogue of galaxies   ???
cmr = Table.read("../cmr_z_DES_griz_zf3_0p4decay_chab_new",format="ascii")   #Ezgal models for griz (DECaLS) ??? (check path!)
cmr = cmr[1:]   #remove some nan values in the model
#cmr = cmr[:290]
cmr = cmr[:120]   #working only up to z=1
##############INPUTS##############
#----------Input table set
filename = "sptecs_subsample.csv"
tab = Table.read(filename,format=filename.split(".")[1])      #load input table
nameind = 0                                        #point cluster name column index  (use tab.colnames.index("COLNAME") to find the index)
raind = 1                                          #cluster ra center
decind = 2                                         #cluster dec center
r200ind = 6    #in arcmin                          #cluster r200
#bcgra = "BCG_R.A."   #theoretical BCGs ra and dec columns    ??? (comment if no BCG to compare)
#bcgdec = "BCG_Decl."
z_theo = "z"  #theoretical z column ??? (comment if no BCG to compare and check fullplot at the end)
z_theo_name = "bleem_z"  #what is theoretical z? (label for plots)
#---------parameters
s = 3.0                              #sigmas for sigmaclipping
max_sig = 0.25                       #max sigma limit for RS overdensity (currently not using)
r2cut = 0.5#0.75                     #how many r200 for RS identification
init_bands = ["gr","ri","iz"]#,"rz"]   #bands to work
pluscut = 3                          #(mstar + pluscut) faint cut 
minuscut = -2                        #(mstar + minuscut) bright cut
mingals = 50#10                      #minimum number of galaxies (if not it triggers r2cut+=0.25)
cosmo = cosmo_boc
xmi = 15                             #CMD limits
xma = 27
ymi = -0.52
yma = 2
plots = True                         #do plots?
save = True                         #save results?   if False, check the "kal" table and save manually
resume = False                       #stack results to last pztab (check tab excepts)
bricky = False                        #is data bricked?
##################################
#------------tab_excepts----------------#  ???
#tab = Table(tab[list(tab[tab.colnames[nameind]]).index("J1257-2926")])         #use this line for individual cluster run ??? (comment for full table)
#tab = tab[np.logical_and(tab[tab.colnames[nameind]]!="SPT-CLJ0205-5829",tab[tab.colnames[nameind]]!="SPT-CLJ2040-4451")]         #use this line to remove some clusters 
#tab = tab[tab[tab.colnames[nameind]]!="em01_106144_020_ML00009_009_c947"]     #no band info
#tab = tab[tab[tab.colnames[nameind]]!="SPT-CLJ2100-5708"]
#tab = tab[tab[tab.colnames[nameind]]!="J1000-3016"]     #no complete band info
#tab = tab[tab[tab.colnames[nameind]]!="J1238-2854"]     #no complete band info
#tab = tab[tab[tab.colnames[nameind]]!="J1325-2014"]     #no complete band info
#tab = tab[tab[tab.colnames[nameind]]!="J1239-2915"]     #no complete band info
#tab = tab[tab[tab.colnames[nameind]]!="J1332-2017"]     #no complete band info
#----resume excepts
#expit = "J1251-2230"
#tab = tab[list(tab[tab.colnames[nameind]]).index(expit):]    #here put the name of the cluster that yield error, then use resume=True and run to resume from that cluster
#pztab = Table.read("temp_"+expit+".csv",format="csv")
#----------------------------------------
init_cluster = list(tab[tab.colnames[nameind]])
init_RA = np.array(tab[tab.colnames[raind]])
init_DEC = np.array(tab[tab.colnames[decind]])
init_r200 = tab[tab.colnames[r200ind]]*u.arcmin
#-------------------DECaLS cat init
if bricky:
    init_brick_lst = []                          #???#here i will load the name of available bircks directly form a file produced by get_bricks.py 
    for i in range(len(tab)):
        binf = Table.read(tab[tab.colnames[nameind]][i]+"/bricks_info.txt",format="ascii")
        init_brick_lst += [list(binf["LIST"])]   #if you have full catalogs instead of brick catalogs, modify this part and anything related to init_brick_lst
else:
    init_brick_lst = np.zeros(len(init_RA))
cols = ["ls_id","ra","dec","type","fitbits","flux_g","flux_r","flux_i","flux_z","flux_ivar_g","flux_ivar_r","flux_ivar_i","flux_ivar_z","mw_transmission_g","mw_transmission_r","mw_transmission_i","mw_transmission_z","galdepth_g","galdepth_r","galdepth_i","galdepth_z"]#,"nobs_g","nobs_r","nobs_i","nobs_z"]
#the script expect a "cluster_name"/"cluster_name".csv file containing decals data
##################################

###############DEFINITIONS################

def col_evo_weights(z):    #??? color band weights!!
    bands = ["gr","ri","iz"]#,"rz"]
    #--H17 weight
    #if z<0.35:
    #    w = [2,2,0,0,0]
    #if z>=0.35 and z<0.75:
    #    w = [0,0,2,0,2]
    #if z>=0.75:
    #    w = [0,0,0,2,2]
    #--myweight
    if z==-1.0:
        w = [0,0,0]#,0]
    if z>=0.0 and z<0.35:
        w = [2,0,0]#,0]
    if z>=0.35 and z<0.75:
        w = [0,2,0]#,0]
    if z>=0.75 and z<1.1:
        w = [0,0,2]#,0]
    if z>=1.1:
        w = [0,0,2]#,0]
    #--simunclusters
    #w = [2]

    while len(w)<len(init_bands):    #uwoparche para giles
        w += [0]

    if len(w)>len(init_bands):
        print("CRITICAL MISMATCH: check col_evo_weights def!!")
        print(init_bands)
        print(w)
        w = 0
    return w

def linear(x, A, B):
    return (A*x) + B
lmodel = Model(linear)

def gauss(x, amp, cen, wid):
    "1-d gaussian: gaussian(x, amp, cen, wid)"
    return (amp/(np.sqrt(2*np.pi)*wid)) * np.exp(-(x-cen)**2 /(2*wid**2))
gmodel = Model(gauss)

def model_dist(m,b,xlst,ylst):
    if m==0.0:          #uwoparche para evitar nans cuando fit de RS es plana
        m = -1e-9
    lst = []
    for i in range(len(xlst)):
        x0 = xlst[i]
        y0 = ylst[i]
        with warnings.catch_warnings():  # Ignore warnings
            warnings.simplefilter('ignore')
            Mneg = -1/m
            Bneg = y0 - (Mneg*x0)
            x1 = (b-Bneg)/(Mneg-m)
            y1 = (m*x1)+b
            if y0>y1:
                lst += [np.sqrt(((x0-x1)**2) + ((y0-y1)**2))]
            else:
                lst += [(-1)*np.sqrt(((x0-x1)**2) + ((y0-y1)**2))]
    return lst

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return (average, np.sqrt(variance))

#def radial_filter(x_lst,rad):      #hennig17
#    Rs = rad/6
#    c = rad/Rs
#    xx = c*np.array(x_lst)
#    res = []
#    for i in range(len(xx)):
#        x = xx[i]
#        if x>1:
#            y = (2/np.sqrt((x**2)-1))*np.arctan(np.sqrt((x-1)/(x+1))) + np.log(x/2)
#        if x<1:
#            y = (2/np.sqrt(1-(x**2)))*np.arctanh(np.sqrt((1-x)/(x+1))) + np.log(x/2)
#        if x==1:
#            y =  1 + np.log(x/2)
#        res += [(1/((x**2)-1))*y]
#    return res

def radial_filter(x_lst,rad):    #klein19
    Rs = rad/6
    xx = np.array(x_lst)/Rs
    res = []
    for i in range(len(xx)):
        x = xx[i]
        if x>1:
            y = 1 - (2/np.sqrt((x**2)-1))*np.arctan(np.sqrt((x-1)/(x+1)))
        if x<1: 
            y = 1 - (2/np.sqrt(1-(x**2)))*np.arctanh(np.sqrt((1-x)/(x+1)))
        else:
            y = 1
        if np.array(x_lst)[i]<0.1*rad:
            x = 0.1*rad/Rs
            y = 1 - (2/np.sqrt(1-(x**2)))*np.arctanh(np.sqrt((1-x)/(x+1)))
        res += [(1/((x**2)-1))*y]
    return res

#def radial_filter(x_lst,rad):    #mine
#    xx = np.array(x_lst)/rad
#    res = []
#    for i in range(len(xx)):
#        x = xx[i]
#        y = np.exp(4*(1-x))
#        res += [y]
#    return res

def extrapolator(scmr):
    hres_cmr = Table(scmr[0])         #also extrapolate more models for extra resolution
    hres_cmr.remove_row(0)
    for i in range(len(scmr)):
        if i == len(scmr)-1:
            break
        lst = []
        for col in scmr.colnames:
            if col=="col1":
                deci = 3
            else:
                deci = 4
            step = np.around((scmr[col][i+1] - scmr[col][i])/2,deci)
            if np.sign(step)==1.0:
                lst += [np.linspace(np.around(scmr[col][i]+step,deci),np.around(scmr[col][i+1]-step,deci),1)]
            else:
                lst += [np.linspace(np.around(scmr[col][i+1]-step,deci),np.around(scmr[col][i]+step,deci),1)]
        hres_cmr.add_row(scmr[i])
        hres_cmr = vstack([hres_cmr,Table(lst,names=scmr.colnames)])
    hres_cmr.add_row(scmr[-1])
    return hres_cmr

def chi2min(crt_lst,kcmr,iter_init_bands=init_bands,binned=False,clipping=True,zhint=0,forcedband=False,itersig=s,colhint=0):
    iterpluscut = pluscut+0
    iterminuscut = minuscut+0
    while True:     #uwoparche para cumulos pesaos (see "prooflst" var)
        bcoef_lst = []
        zcl_lst = []
        bres_lst = []
        blab_lst = []
        contflag = []
        for i in range(len(kcmr)):
            gcol = list(kcmr[kcmr.colnames[2::6]][i])    #extract g, r, i, z magnitudes at 6 different levels 3, 2, 1, 0.5, 0.4, 0.3
            rcol = list(kcmr[kcmr.colnames[3::6]][i])
            icol = list(kcmr[kcmr.colnames[4::6]][i])
            zcol = list(kcmr[kcmr.colnames[5::6]][i])
            cmr_coltab = Table([gcol,rcol,icol,zcol],names=["g","r","i","z"])
            mr = [list(cmr_coltab[cl[1]]) for cl in iter_init_bands]      #sort on an iterable list
            mb = [list(cmr_coltab[cl[0]]) for cl in iter_init_bands]
            if zhint!=0:         #if already know a redshift, use that mstar for all models
                ikcmr = kcmr[kcmr["col1"]==zhint]
                gcol = list(ikcmr[ikcmr.colnames[2::6]][0])    #extract g, r, i, z magnitudes at 6 different levels 3, 2, 1, 0.5, 0.4, 0.3
                rcol = list(ikcmr[ikcmr.colnames[3::6]][0])
                icol = list(ikcmr[ikcmr.colnames[4::6]][0])
                zcol = list(ikcmr[ikcmr.colnames[5::6]][0])
                icmr_coltab = Table([gcol,rcol,icol,zcol],names=["g","r","i","z"])
                zlcut = [icmr_coltab[cl[1]][2] for cl in iter_init_bands]
            else:
                zlcut = [cmr_coltab[cl[1]][2] for cl in iter_init_bands]      #sort level 1 magnitudes (m*) of redder bands on iterable list
            zrlab = [labi[1] for labi in iter_init_bands]                     #sort blue and red band labels on iterable list
            zblab = [labi[0] for labi in iter_init_bands]
             
            coef_lst = []
            res_lst = []
            lab_lst = []
            sig_lst = []
            err_lst = []
            richness = []
            bflag = []
            for mredder, mbluer, lcut, rlab, blab in zip(mr,mb,zlcut,zrlab,zblab):
                can = crt_lst[iter_init_bands.index(blab+rlab)]
                can = can.copy()  #this avoid global variable modify
                if binned==True and forcedband==False:
                    can.rename_column("col0","m_"+rlab)
                #if cmr["col1"][i]<=0.2:    #song12 faint cut
                #    pluscut = 3
                #if cmr["col1"][i]>0.2 and cmr["col1"][i]<=0.6:
                #    pluscut = 2
                #if cmr["col1"][i]>0.6:
                #    pluscut = 1
                #lcrt = can
                if binned==True and forcedband==False:
                    lcrt = can.copy()
                else:
                    lcrt = can[can["m_"+rlab]<=lcut+iterpluscut]
                    lcrt = lcrt[lcrt["m_"+rlab]>=lcut+iterminuscut]
                #lcrt = lcrt[lcrt["mag_"+rlab+"_err"]<0.1]    #sig_mag < 0.1 hennig17
                redder = np.array(lcrt["m_"+rlab])
                #bluer = np.array(lcrt["m_"+blab])
                #rederr = np.array(lcrt["mag_"+rlab+"_err"])
                #bluerr = np.array(lcrt["mag_"+blab+"_err"])
                clipcols=["devs","real_yrcs","devs_err","redder","model_yrcs"]

                if len(lcrt)<5:    #if less than 10 galaxies, then no RS
                    #print("is dis ti rial laif"+str(i)+" -- "+blab+rlab)
                    lcrt = Table([[0.0],[0.0],[0.0],[0.0],[0.0]],names=clipcols)
                    lcrt.remove_row(0)
                    coef_lst += [[0.0,0.0]]
                    err_lst += [0.0]
                    sig_lst += [0.0]
                    res_lst += [lcrt]
                    lab_lst += [blab+rlab]
                    richness += [0.0]
                    bflag += [1]
                    continue
              
                #make rs model line by fitting to the six points
                lmodel = Model(linear)
                ez_rcs = mbluer[2] - mbluer[1]
                params = lmodel.make_params(A=((mbluer[2]-mredder[2])-(mbluer[1]-mredder[1]))/(mredder[2]-mredder[1]),B=ez_rcs) #initial params
                result = lmodel.fit(np.array(mbluer)-np.array(mredder), params, x=mredder)
                Ares = np.array(result.params)[0]
                Bres = np.array(result.params)[1]
                if binned:
                    real_yrcs = np.array(lcrt["col1"])
                    yerror = np.array(lcrt["col2"])   #np.array(0.06/(lcrt["col2"]**2))   #err
                else:
                    real_yrcs = np.array(lcrt["m_"+blab]-lcrt["m_"+rlab])
                    yerror = np.sqrt((lcrt["mag_"+rlab+"_err"]**2) + (lcrt["mag_"+blab+"_err"]**2))
                #real_yrcs = bluer - redder
                model_yrcs = linear(redder,Ares,Bres)
                md_list = model_dist(Ares,Bres,redder,real_yrcs)    #take ortogonal distance to model
                #if colhint!=0:
                #    chdevs = model_dist(Ares,Bres,np.array([(np.max(redder)+np.min(redder))/2]),np.array([colhint]))[0]
                #devs = real_yrcs - model_yrcs 
                devs = np.array(md_list)
                #theta = np.arctan(Ares)
                #sig_proj2 = (np.sqrt(bluerr**2 + rederr**2)*np.cos(theta))**2 + (rederr*np.sin(theta))**2
                #sig_int2 = 0.03**2            #hennig17
                #sig_col2 = sig_proj2 + sig_int2   #estimate ortogonal sigma2
                #clip = Table([lcrt["N_ID"],devs,np.sqrt(sig_proj2),redder,real_yrcs,model_yrcs],names=["N_ID","devs","devs_err","redder","real_yrcs","model_yrcs"])
                clip = Table([devs,real_yrcs,yerror,redder,model_yrcs],names=clipcols)
                #if len(iter_init_bands)==1 and forcedband==False:
                #    clip = clip[abs(clip["devs"]-np.median(clip["devs"]))<=0.22]      #work only with data within +-0.22 from median
                
                if len(clip)<5:    #again, if less than 10 galaxies, then continue
                    if len(clip)!=0:
                        clip = Table(clip[0])
                        clip.remove_row(0)
                    coef_lst += [[Ares,Bres]]
                    err_lst += [0.0]
                    sig_lst += [0.0]
                    res_lst += [clip]
                    lab_lst += [blab+rlab]
                    richness += [0.0]
                    bflag += [2]
                    continue
                #######################testing_model###########################
                #if blab+rlab=="iz":
                #    print(str(np.median(clip["model_yrcs"]))+" N: "+str(len(clip)))
                #--------uwoparche brightness weight
                clip["weight"] = clip["devs_err"]**(-1)            #err-1
                itersig_c = itersig + 0
                if clipping:
                    #if binned==True and forcedband==False:
                    #    clip["weight"] = clip["devs_err"]**(-1)#(clip["devs_err"]/0.06)**(-1/2)   #N
                    #else:
                    cont = 0
                    while True:
                        params, pcov = curve_fit(linear,clip["redder"],clip["real_yrcs"],sigma=clip["weight"]**(-1))
                        iAres = params[0]
                        iBres = params[1]
                        fiterror = np.sqrt(np.diag(pcov))
                        iBres_err = fiterror[1]
                        iAres_err = fiterror[0]
                        overdensity_yrcs = linear(clip["redder"],iAres,iBres)
                        color_high = linear(clip["redder"],iAres,iBres+(itersig_c*iBres_err))
                        color_low = linear(clip["redder"],iAres,iBres-(itersig_c*iBres_err))
                        rej = clip[np.logical_or(clip["real_yrcs"]>color_high,clip["real_yrcs"]<color_low)]
                        if len(rej)==0:
                            break
                        clip = clip[np.logical_and(clip["real_yrcs"]<=color_high,clip["real_yrcs"]>=color_low)]
                        cont += 1
                    #print(params)
                    #print(fiterror)
                    #print(str(iBres+(iAres*21))+" - "+str(colhint))
                    #print(iBres_err)

                    if len(clip)<5:
                        if len(clip)!=0:
                            clip = Table(clip[0])
                            clip.remove_row(0)
                        coef_lst += [[Ares],[Bres]]
                        err_lst += [0.0]
                        sig_lst += [0.0]
                        res_lst += [clip]
                        lab_lst += [blab+rlab]
                        richness += [0.0]
                        bflag += [3]
                        continue

                rcs_dev = np.mean(clip["devs"])
                sig = 0.22
                clip.remove_column("weight")
                
                if False:#if len(iter_init_bands)==1 and blab+rlab=="gr" and forcedband==False:# and kcmr["col1"][i]>0.42 and kcmr["col1"][i]<=0.45: #init_bands and blab+rlab=="ri" and binned==False:
                    print(lcut)
                    print("z = "+str(kcmr["col1"][i])+", "+str(rcs_dev)+","+str(np.median(clip["model_yrcs"]))+" N: "+str(len(clip))+" sig: "+str(sig)+" chi2: "+str(np.sum(((clip["real_yrcs"]-clip["model_yrcs"])**2)/(clip["devs_err"]**2))))#/(len(clip)-1-2)))
                    plt.clf()
                    plt.scatter(clip["redder"],clip["real_yrcs"])
                    plt.plot(clip["redder"],clip["model_yrcs"])
                    plt.xlim([xmi,xma])
                    plt.ylim([ymi,yma])
                    plt.show()

                #print("N = "+str(np.sum((clip["devs_err"]/0.06)**(-1/2)))+" for band "+blab+rlab) 
                coef_lst += [[Ares,Bres]]
                err_lst += [rcs_dev]#np.sqrt(np.sum(clip["devs_err"]**2)/len(clip))]   #save some results
                richness += [len(clip)]#[np.sum((clip["devs_err"]/0.06)**(-1/2))]
                sig_lst += [sig]
                res_lst += [clip]
                lab_lst += [blab+rlab]
                bflag += [0]
                ######################################################
            #---chi2
            chi_lst = []
            for j in range(len(res_lst)):
                clip = res_lst[j]
                if len(clip)<5:
                    chi_lst += [[0.0,lab_lst[j],0.0]]    #add chi2 0.0 if len(clip)<10
                    continue
                c2 = np.sum(((clip["real_yrcs"]-clip["model_yrcs"])**2)/(clip["devs_err"]**2))/(len(clip)-1-2)#/clip["model_yrcs"])/(len(clip)-1-2)
                #if iter_init_bands==init_bands and j==1:
                #print(c2)
                if c2 < 0.0:              #uwoparche: sometimes a negative value appears, idk why
                    c2 = 0.0
                chi_lst += [[c2,lab_lst[j],err_lst[j]]]
            chi_lst = Table(np.array(chi_lst),names=["chi2","band","band_z_err"])
            chi_lst["chi2"] = Table.Column(chi_lst["chi2"],dtype="float")
            #chi_lst["band"] = Table.Column(chi_lst["band"],dtype="str")
            cweight = col_evo_weights(kcmr["col1"][i])
            if forcedband:
                cweight = [2 for cwe in range(len(cweight))]    #if forced band, don't use weights
            cweight = [cweight[cw] for cw in range(len(cweight)) if init_bands[cw] in iter_init_bands]  #resize len to iter_init_bands len
            while len(cweight)<len(chi_lst):    #add weight 0 for other filters ()
                cweight += [0]
            tablon = Table([cweight,list(chi_lst["band"])],names=["cweight","band"]) #run script again when EOL str literal
            chitab = join(chi_lst,tablon,keys="band")                                #the syntaxis is correct >:{L
            chitab["z"] = np.ones(len(chitab))*kcmr["col1"][i]
            #chitab["band_z_err"] = Table.Column(chitab["band_z_err"],dtype="O")
            #if iter_init_bands==["iz"]:
            #    print(chitab)
            if i==0:
                ult_chitab = chitab.copy()
            else:
                ult_chitab = vstack([ult_chitab,chitab])
            chitab = chitab[chitab["chi2"]!=0.0]     #chi2=0.0 are len(clip)<5, not interested
            if len(chitab)!=0:    #for bands with estimated chi2, get weighted average
                if np.sum(chitab["cweight"])==0.0:
                    chi2 = 0.0
                else:
                    chi2 = np.average(chitab["chi2"],weights=chitab["cweight"])
                    color_error = np.sqrt(np.sum(np.array(chitab["band_z_err"],dtype="float")**2)/len(chitab))
            else:    #if no bands with chi2, then make chi2=0.0
                chi2 = 0.0
            ########################################################
            main_sig = sig_lst[np.array(cweight).argmax()]
            main_richness = richness[np.array(cweight).argmax()]
            N_clip = len(res_lst[np.array(cweight).argmax()])
            main_err = err_lst[np.array(cweight).argmax()]
            if chi2==0.0:
                color_error = 0.0
            bcoef_lst += [coef_lst]
            bres_lst += [res_lst]
            blab_lst += [lab_lst]
            zcl_lst += [[kcmr["col1"][i],chi2,main_sig,main_richness,main_err]]
            contflag += [bflag]
        prooflst = np.array(zcl_lst)[:,1]        #uwoparche para cumulos pesaos
        if len(iter_init_bands)==1:
            break                #it is possible to not find a redshift if working with 1 band
        if len(prooflst[prooflst!=0.0])==0:  #if all models give chi2 = 0.0, then repeat and change mag cuts  
            iterminuscut += -1
            if iterminuscut < -4:
                iterminuscut = -4
                iterpluscut += 1
                print("still not enough data points, changing faint cut to m*+"+str(int(abs(iterpluscut))))
                if iterpluscut==10:
                    print("ERROR: sample strangulated!")
                    break
            else:
                print("too few data points, changing bright cut to m*-"+str(int(abs(iterminuscut))))
            #print(iterpluscut)
        else:
            #if iterminuscut != minuscut:
            #    print("too few data points, changing bright cut to m*-"+str(int(abs(iterminuscut))))
            #if iterpluscut != pluscut:
            #    print("too few data points, changing faint cut to m*+"+str(int(abs(iterpluscut))))
            break
    ztab = Table(np.reshape(zcl_lst,[len(zcl_lst),5]))
    #with warnings.catch_warnings():  # Ignore warnings
    #    warnings.simplefilter('ignore')
    #    ztab["col5"] = ztab["col3"]/(ztab["col2"]*ztab["col4"])
    #ztab["col5"][Table.Column(ztab["col5"],dtype="str")=="nan"] = 0.0
    ztab = Table(ztab,names=["photo-z","comb_chi2","main_sig","richness","color_error"],dtype=["float","float","float","int","float"])
    #if len(iter_init_bands)==1:
    #---------chi2 minimization----------#
    #----check minima of each band
    cw_lst = []
    b_lst = []
    for band in iter_init_bands:         
        ichil = ult_chitab[ult_chitab["band"]==band]
        ichi = ichil[ichil["chi2"]!=0.0]
        try:
            bpz = ichi[ichi["chi2"]==np.min(ichi["chi2"])]["z"][0]       #take the minima of each band
        except:
            bpz = -1.0
        cw = col_evo_weights(bpz)                    #check its weight
        cw = cw[init_bands.index(band)]
        if cw!=0.0:                               #if weight is relevant, save band
            cw_lst += [cw]
            b_lst += [band]

    if len(b_lst)==1:                   #if only one relevant band, use his distribution instead if comb_chi2 (to avoid band jump effects)
        ichil = ult_chitab[ult_chitab["band"]==b_lst[0]]
        ztab["comb_chi2"] = ichil["chi2"]
    if len(b_lst)>1:                #if more than one, cry
        #ichi = ult_chitab[ult_chitab["chi2"]!=0.0]
        #band = ichi[ichi["chi2"]==np.min(ichi["chi2"])]["band"][0]
        ichil = ult_chitab[ult_chitab["band"]==b_lst[0]]
        ztab["comb_chi2"] = ichil["chi2"]
        for ic in range(len(ztab)):
            cw = col_evo_weights(ztab["photo-z"][ic])   #weights at that redshift
            for band in init_bands:
                if band not in b_lst:
                    cw[init_bands.index(band)] = 0    #force weight of irrelevant bands to be 0
            if len(np.array(cw)[np.array(cw)!=0])==0:        #if all bands irrelevant at this redshift continue
                continue
            else:
                band = init_bands[np.array(cw).argmax()]   #if there is a relevant band, replace comb_chi2 for chi2 of that band at that z
                ichil = ult_chitab[ult_chitab["band"]==band]
                ztab["comb_chi2"][ic] = ichil["chi2"][ic]
    
    #--------select redshift of minimum comb_chi2
    zt = ztab[ztab["comb_chi2"]!=0.0]
    try:
        pz = zt[zt["comb_chi2"]==np.min(zt["comb_chi2"])]["photo-z"][0]
        final_res_lst = bres_lst[list(ztab["photo-z"]).index(pz)]
        lab_lst = blab_lst[list(ztab["photo-z"]).index(pz)]
        coef = bcoef_lst[list(ztab["photo-z"]).index(pz)]
    except:
        pz = -1.0
        final_res_lst = bres_lst[0] 
        if len(final_res_lst[0])!=0:
            final_res_lst[0] = Table(final_res_lst[0][0])
            final_res_lst[0].remove_row(0)
        coef = [[0.0,0.0]]
        lab_lst = [iter_init_bands[0]]
    ########ESTIMATE UNCERTAINTY##############
    #uztab = ztab[ztab["comb_chi2"]!=0.0]
    ## Compute the minimum reduced chi-square value and the corresponding photometric redshift
    #min_r_chi2 = np.min(uztab["comb_chi2"])
    #z_phot_min_r_chi2 = pz
    ## Select a subset of the data within 0.1 of the minimum photometric redshift
    #z_phot_subset = uztab[np.abs(uztab["photo-z"] - z_phot_min_r_chi2) <= 0.2]["photo-z"]
    #r_chi2_subset = uztab[np.abs(uztab["photo-z"] - z_phot_min_r_chi2) <= 0.2]["comb_chi2"]
    ## Compute the difference between each reduced chi-square value and the minimum value
    #delta_r_chi2 = r_chi2_subset - np.max(uztab["comb_chi2"])
    ## Fit a negative Gaussian to the (z_phot, delta_r_chi2) data
    #def neg_gauss(x, a, mu, sigma):
    #    return -a * np.exp(-0.5 * ((x - mu) / sigma)**2)
    #p0 = [0.1, z_phot_min_r_chi2, 0.1]  # initial guess for Gaussian parameters
    #try:
    #    popt, pcov = curve_fit(neg_gauss, z_phot_subset, delta_r_chi2, p0=p0)
    #    # The uncertainty is given by the absolute value of the standard deviation of the Gaussian
    #    sigma_z_phot = np.abs(popt[2])
    #except:
    #    sigma_z_phot = 0.0
    return ztab, ult_chitab, pz, final_res_lst, coef, lab_lst, iterpluscut


def overdensity_locator(cluster,crt,bkg,pcmr,area_factor,xbsize=0.6,ybsize=0.06):
    rcstab_lst = []
    pz_lst = []
    moffset_lst = []
    notrcstab_lst = []
    bflag_lst = []
    for band in init_bands:
        rlab = band[1]
        blab = band[0]
        #cbkg = bkg[bkg["mag_"+rlab+"_err"]<0.1]   #hennig mag error cut sigma<0.1
        #ccrt = crt[crt["mag_"+rlab+"_err"]<0.1]
        #if band=="gr" or band=="gi":
        #    magerr_cut = 0.25
        #if band=="ri":
        #    magerr_cut = 0.35
        #if band=="iz":
        #    magerr_cut = 0.45
        #magerr_cut = 0.1
        #cbkg = bkg[np.logical_and(bkg["mag_"+rlab+"_err"]<=magerr_cut,bkg["mag_"+blab+"_err"]<=magerr_cut)]   #franklin correction to hennig mag error cut
        #ccrt = crt[np.logical_and(crt["mag_"+rlab+"_err"]<=magerr_cut,crt["mag_"+blab+"_err"]<=magerr_cut)]
        with warnings.catch_warnings():  # Ignore warnings
            warnings.simplefilter('ignore')
            rdepth = -2.5 * (np.log10(10. / np.sqrt(crt["galdepth_"+rlab])) - 9)    #now cutting up to max depth
            bdepth = -2.5 * (np.log10(10. / np.sqrt(crt["galdepth_"+blab])) - 9)
        cbkg = bkg[np.logical_and(bkg["m_"+rlab]<=np.max(rdepth),bkg["m_"+blab]<=np.max(bdepth))]
        ccrt = crt[np.logical_and(crt["m_"+rlab]<=np.max(rdepth),crt["m_"+blab]<=np.max(bdepth))]
        # uwoparche ara evitar contaminaciooon
        #faint01cut = np.max(ccrt["m_"+rlab]) - 0.3*(np.max(ccrt["m_"+rlab])-np.min(ccrt["m_"+rlab])) 
        #ccrt = ccrt[ccrt["m_"+rlab]<=faint01cut]         #remove the fainter 10% of galaxies
        #--------klein radial filter
        #radi = iter_r2cut*r200.value
        #Lpos = np.ones(len(ccrt))#radial_filter(ccrt["PRJ_SEP"],radi)
        #--------uwoparche brightness weight
        woff = ccrt["m_"+rlab]-np.max(ccrt["m_"+rlab])
        Lpos = np.sqrt(abs((woff))/np.max(abs(woff)))      #??? if you dont want weights comment and use the line below
        #Lpos = np.ones(len(ccrt))
        plt.clf()
        tabo = cbkg
        hb = plt.hist2d(tabo["m_"+rlab],tabo["m_"+blab]-tabo["m_"+rlab],range=[[xmi,xma],[ymi,yma]],bins=[int((xma-xmi)/xbsize),int((yma-ymi)/ybsize)],weights=np.ones(len(tabo)))
        tabo = ccrt
        hd = plt.hist2d(tabo["m_"+rlab],tabo["m_"+blab]-tabo["m_"+rlab],range=[[xmi,xma],[ymi,yma]],bins=[int((xma-xmi)/xbsize),int((yma-ymi)/ybsize)],weights=Lpos)
        hb_norm = hb[0]*area_factor    #normalize by area factor
        hres = hd[0] - hb_norm.value   #subtract from r200 bins
        hres[hres<0] = 0.0               #everything below zero is irrelevant
        gridi = np.meshgrid(hd[1][:-1]+(xbsize/2),hd[2][:-1]+(ybsize/2))    #gonna work with centers of bins
        lst = []
        for i in range(len(hres[:,0])):
            for j in range(len(hres[0,:])):
                #if hres[i,j]==0.0:
                #    continue
                lst += [(np.around(gridi[0][j][i],1),np.around(gridi[1][j][i],2),hres[i,j])]
        lst = np.array(lst)
        plt.close()
        bindat = Table([lst[:,0],lst[:,1],lst[:,2]])   #magnitude center, color center, n galaxies (bkg subtracted)
        bindat = bindat[bindat["col2"]!=0.0]
        if list(Lpos) != list(np.ones(len(ccrt))):    #get bindat without weights
            hd2 = plt.hist2d(tabo["m_"+rlab],tabo["m_"+blab]-tabo["m_"+rlab],range=[[xmi,xma],[ymi,yma]],bins=[int((xma-xmi)/xbsize),int((yma-ymi)/ybsize)],weights=np.ones(len(ccrt)))
            hres2 = hd2[0] - hb_norm.value
            hres2[hres2<0] = 0.0
            gridi = np.meshgrid(hd2[1][:-1]+(xbsize/2),hd2[2][:-1]+(ybsize/2))    #gonna work with centers of bins
            lst = []
            for i in range(len(hres2[:,0])):
                for j in range(len(hres2[0,:])):
                    #if hres[i,j]==0.0:
                    #    continue
                    lst += [(np.around(gridi[0][j][i],1),np.around(gridi[1][j][i],2),hres2[i,j])]
            lst = np.array(lst)
            plt.close()
            bindat2 = Table([lst[:,0],lst[:,1],lst[:,2]])   #magnitude center, color center, n galaxies (bkg subtracted)
            bindat2 = bindat2[bindat2["col2"]!=0.0]
 
        if False:
            plt.clf()
            #plt.errorbar(bindat["col0"],bindat["col1"], yerr=ybsize/(bindat["col2"]**2),fmt="none",capsize=5,color="black")
            #plt.scatter(bindat["col0"],bindat["col1"],color="black")
            tabo = ccrt
            plt.hist2d(tabo["m_"+rlab],tabo["m_"+blab]-tabo["m_"+rlab],range=[[xmi,xma],[ymi,yma]],bins=[int((xma-xmi)/xbsize),int((yma-ymi)/ybsize)],weights=np.ones(len(tabo)))
            plt.xlim([xmi,xma])
            plt.ylim([ymi,yma])
            plt.ylabel(blab+"-"+rlab)
            plt.xlabel(rlab)
            plt.show()
        
        #------select bins near max peak--------#
        YY = np.sum(hres,axis=0)    #sum counts of bins with the same color (collapse X axis)
        YY2 = YY.copy()
        XX = hd[2][:-1]+(ybsize/2)  #take color center of bins
        YYun = np.sum(hres2,axis=0) #unweighted
        YY2un = YYun.copy()
        ##############params##############
        Ntresh = 30      #minimum number of galaxies within 3sigma to consider detection
        ptresh = 5 #np.mean(YY) + np.std(YY)   #minimum prominence to consider a detection (height minus lowest contour)
        minsep = 0.22          #minimum color separation between peaks
        ##################################

        #if False:#np.max(YY)<5:   #if max enhanced counts are still below 5, this cluster in this band is shit
        #    pz_lst += [-1.0]
        #    noclip = Table(ccrt[0])
        #    noclip.remove_row(0)
        #    rcstab_lst += [ccrt]
        #    notrcstab_lst += [noclip]
        #    moffset_lst += [99.0]
        #    bflag_lst += [2]
        #    continue

        YY_per_peak = []
        XX_per_peak = []
        cpeaks = []
        cpkprom_lst = []
        while True:
            peaks = signal.find_peaks(YY2,prominence=ptresh)     #find peaks with minimum treshold 
            if len(peaks[0])==0:
                break
            #cp_lst = XX[peaks[0][peaks[1]["prominences"].argsort()]]   #get color of peaks in decreasing order 
            cpind = peaks[1]["prominences"].argsort()                  #get index of peaks in increasing prominence order
            ipk = cpind[0]                                 #get smallest peak
            cpk = XX[peaks[0][ipk]]
            cpk_h = YYun[peaks[0]][ipk]   #un
            cpk_prom = peaks[1]["prominences"][ipk]
            minind = peaks[1]["left_bases"][ipk]
            maxind = peaks[1]["right_bases"][ipk]
            Ypk = YY2un[minind:maxind+1]   #un
            Xpk = XX[minind:maxind+1]
            breaker = 0
            while True:                            #fit gaussian iteratively until gaussian wid lessequal than signal peak width
                if breaker>=0.9:
                    break
                result = gmodel.fit(Ypk,x=Xpk,amp=cpk_h,cen=cpk,wid=0.09)
                cpk_wid = np.array(result.params)[2]
                cpk_cen = np.array(result.params)[1]
                cpk_amp = np.array(result.params)[0]
                cpk_rich = np.sum(YYun[np.logical_and(XX>=cpk_cen-3*cpk_wid,XX<=cpk_cen+3*cpk_wid)])   #un
                if 3*cpk_wid - ((np.max(Xpk)-np.min(Xpk))/2) > 0.22:   #if an entire RS fits into separation, thats too large man :L
                    breaker += 1/10
                    Ypk[Ypk < cpk_h*breaker] = 0
                    if breaker>=0.9:
                        break
                    continue
                break                  #this will ensure a descent gaussian fit, is like an inverted sigma clipping
            if cpk_rich<Ntresh:             #if richness lower than treshold, the peak is below detection level
                YY2[minind:maxind+1] = 0
                YY2un[minind:maxind+1] = 0
                continue
            #YY_per_peak += [Ypk.copy()]          #extract data of smallest peak
            #XX_per_peak += [Xpk.copy()]
            dcpks = np.array(cpeaks) - cpk   #check if we already found another peak within +-0.22
            colliders = np.array(cpeaks)[abs(dcpks)<minsep]
            #since we work less prominent peaks first, the following instance should never yield error
            if len(colliders)==0 or len(cpeaks)==0:    #if no colliders or first iteration, save peak
                cpeaks += [cpk]
                cpkprom_lst += [cpk_prom]
            else:                                      #if there are colliders
                tcp = [cpeaks[cpeaks.index(colliders[i])] for i in range(len(colliders))]   
                tcp += [cpk]                       #take color peaks of colliders plus cpk of iterated peak
                lst = [cpkprom_lst[cpeaks.index(colliders[i])] for i in range(len(colliders))]
                lst += [cpk_prom]                  #take their prominences too
                flcpk = cpeaks[cpeaks.index(tcp[np.array(lst).argsort()[0]])]    #take the color peak of the less prominent peak (should work*)
                #cpkprom_lst = list(np.array(cpkprom_lst)[cpeaks!=flcpk])
                cpeaks = list(np.array(cpeaks)[cpeaks!=flcpk])           #remove it from cpeaks
                trcpk = np.array(tcp)[np.array(lst).argsort()[1:]]       #take color of the rest of the peaks in cuestions (the more prominent)
                cpeaks += [pk for pk in trcpk if pk not in cpeaks]         #add them to cpeaks if not already in (always one*)
            
            if False:
                plt.clf()
                plt.bar(XX,YY2un,ybsize,edgecolor="none",color="black")   #un
                plt.text(yma-0.1,np.max(YYun),s="N: "+str(cpk_rich))   #un
                plt.text(yma-0.1,np.max(YYun)-5,s=r"$\sigma$: "+str(cpk_wid))   #un
                plt.text(yma-0.1,np.max(YYun)-10,s="signal_wid: "+str((np.max(Xpk)-np.min(Xpk))/2))
                plt.xlim([yma,ymi])
                plt.ylim([0,np.max(YYun)+2])    #un
                plt.axhline(ptresh)
                plt.scatter(Xpk,Ypk,color="purple")
                plt.plot(Xpk,result.best_fit,color="yellow")
                plt.xlabel(band[0]+"-"+band[1])
                plt.axvline(cpk_cen,ls="--",color="red")
                plt.axvline(cpk_cen-3*cpk_wid,ls=":",color="red")
                plt.axvline(cpk_cen+3*cpk_wid,ls=":",color="red")
                plt.show()

            YY2[minind:maxind+1] = 0                       #remove smallest peak and surroundings from sample
            YY2un[minind:maxind+1] = 0
        
        #print(band+" "+str(cpeaks))
        if len(cpeaks)==0:   #if N below treshold for all peaks, this cluster in this band is shit
            pz_lst += [[-1.0]]
            noclip = Table(ccrt[0])
            noclip.remove_row(0)
            rcstab_lst += [ccrt]
            notrcstab_lst += [[noclip]]
            moffset_lst += [[99.0]]
            bflag_lst += [2]
            continue

        #print(cpeaks)
        if len(cpeaks)==1:               #if only 1 peak, we are good to go
            bflag = 0
        else:
            print(band+": double peak here!")   #if more than 1 peak, FLAG=1
            bflag = 1
        
        ######doublepeaksaves#######
        noclip_dlst = []
        pz_dlst = []
        Ares_dlst = []
        Bres_dlst = []
        iAres_dlst = []
        iBres_dlst = []
        moffset_dlst = []
        sig_dlst = []
        #--------------------------#
        for color_peak in cpeaks:            #loop for contamination residuals avoiding
            #if len(cpeaks)==1:               #if only 1 peak, we are good to go
            #    color_peak = cpeaks[0]
            #    bflag = 0
            #else:
            #    print(band+": double peak here!")   #if more than 1 peak, take the brightest
            #    XXmag = hd[1][:-1]+(xbsize/2)
            #    lst = []
            #    lst2 = []
            #    for pk in cpeaks:    #for each peak
            #        #if list(Lpos) != list(np.ones(len(ccrt))):  
            #        #    YYmag = np.sum(hres2[:,list(XX).index(pk)-1:list(XX).index(pk)+2],axis=1)   #take the unweighted count sum of bins near color peak for each magnitude bin
            #        #else:
            #        YYmag = np.sum(hres[:,list(XX).index(pk)-1:list(XX).index(pk)+2],axis=1)
            #        YYmag[YYmag<np.median(YYmag[YYmag!=0.0])] = 0.0    #for magnitude bins with counts under median make counts = 0
            #        mag_peak = np.average(XXmag,weights=YYmag)         #take weighted average of magnitudes using counts as weights
            #        lst += [mag_peak]                                  #save the  magnitude peak of each color peak
            #        lst2 += [YY[list(XX).index(pk)]]
            #    lst3 = list(np.array(lst)/np.array(lst2))
            #    color_peak = cpeaks[np.array(lst3).argmin()]  #take the brighter peak
            #    #here must be a filter, not every peak means there are overlapping
            #    bflag = 1      #f**ck, overlapped structures detected
            #-------------------------------------#
            #print(cpeaks)
            #print(lst)
            #print(lst2)
            #if list(Lpos) != list(np.ones(len(ccrt))):
            #    clip = bindat2.copy()
            #else:
            clip = bindat.copy()
            clip = clip[abs(clip["col1"] - color_peak)<=0.22]        #select bins around color peak
            #plt.clf()
            #plt.hist(clip["col2"],bins=30)
            #plt.axvline(np.mean(clip["col2"]))
            #plt.axvline(np.median(clip["col2"]),color="red")
            #plt.text(0.1,1.5,s=str(np.std(clip["col2"])))
            #plt.show()
            #clip = clip[clip["col2"]<np.mean()]
            
            #--------------------finding best linear fit
            cont = 0
            itersig_c = s + 0
            par0 = lmodel.make_params(A=-0.03,B=color_peak) #initial params
            while True:
                params, pcov = curve_fit(linear,clip["col0"],clip["col1"],par0,sigma=(ybsize/(clip["col2"]**2)),absolute_sigma=True)#**(-1))
                iAres = params[0]
                iBres = params[1]
                fiterror = np.sqrt(np.diag(pcov))
                md_list = model_dist(iAres,iBres,clip["col0"],clip["col1"])    #take ortogonal distance to model
                devs = np.array(md_list)
                cen, sig = weighted_avg_and_std(devs,weights=(ybsize/(clip["col2"]**2))**(-1))
                #iBres_err = fiterror[1]/2
                #iAres_err = fiterror[0]
                #sig = iBres_err
                overdensity_yrcs = linear(clip["col0"],iAres,iBres)
                color_high = linear(clip["col0"],iAres,iBres+(itersig_c*sig))
                color_low = linear(clip["col0"],iAres,iBres-(itersig_c*sig))
                rej = clip[np.logical_or(clip["col1"]>color_high,clip["col1"]<color_low)]
                if len(rej)==0:
                    break
                clip = clip[np.logical_and(clip["col1"]<=color_high,clip["col1"]>=color_low)]
                cont += 1
                par0 = lmodel.make_params(A=iAres,B=iBres)
                if len(clip)<5:
                    break
            #    if iBres_err <= 0.22:
            #        break
            #    else:
            #        itersig_c -= 0.5
            #    if itersig_c < 0.5:
            #        break

            #print(params)
            #print(fiterror)
            #print(str(iBres+(iAres*21))+" - "+str(color_peak))
            #print(iAres)
            #print(iBres_err)
            tempx = np.linspace(min(ccrt["m_"+rlab]),max(ccrt["m_"+rlab]),100)
            clip = Table([tempx,linear(tempx,iAres,iBres),np.ones(len(tempx))*sig])
            #print(clip) 
            
            rcs = color_peak
            #fitting Ezgal models
            ztab, ult_chitab, pz, final_res_lst, coef, lab_lst, iterpluscut = chi2min([clip],pcmr,iter_init_bands=[band],colhint=color_peak,binned=True,clipping=False) 
            #print(ult_chitab)

            if False:
                plt.clf()
                col_map = ["indianred","lime","royalblue","magenta"]
                lst = []
                for ik in range(1):
                    bandi = ["gr"][ik]
                    chiband = ult_chitab[ult_chitab["band"]==bandi]
                    chiband = chiband[chiband["chi2"]!=0.0]
                    plt.scatter(chiband["z"],chiband["chi2"],s=5,color=col_map[ik],label=bandi[0]+"-"+bandi[1])
                    try:
                        lst += [np.min(chiband["chi2"])]
                    except:
                        pass
                plt.scatter(ztab["photo-z"],ztab["comb_chi2"],s=20,marker="s",edgecolor="orange",facecolor="none",label="combined")
                #print(plt.ylim())
                fac = 0.1
                #plt.ylim([np.min(lst)-((fac)*np.std(ztab["comb_chi2"])),np.median(ztab["comb_chi2"])+(fac*np.std(ztab["comb_chi2"]))])
                #print(plt.ylim())
                plt.title(cluster)
                plt.xlabel("z")
                plt.ylabel(r"$\chi^{2}$/DOF")
                plt.minorticks_on()
                plt.legend()
                plt.show()
                #plt.savefig(cluster+"/"+cluster+"_chi2.jpg")
            
            if pz==-1.0:    #if no redshift for this band, continue
                #print("HERE")
                clipe = Table(ccrt[0])
                clipe.remove_row(0)
                moffset = 99.0
            else:
                moffset = ztab[list(ztab["photo-z"]).index(pz)]["color_error"]
            #print(moffset)
            #print(band)
            #print(bflag)
            #print(pz)
            #if moffset < 0.09 and band=="gr" and bflag==1 and pz<0.1:  #if residual selected, start again
            #    cpeaks = list(np.sort(cpeaks)[1:])
            #    continue
            Ares,Bres = coef[0] 
            overdensity_yrcs = linear(clip["col0"],Ares,Bres)
            #-----------select RS galaxies---------#
            real_yrcs = np.array(ccrt["m_"+blab]-ccrt["m_"+rlab])
            redder = np.array(ccrt["m_"+rlab])
            #noclip = ccrt[np.logical_and(redder<=linear(redder,Ares,Bres+(s*Bres_err)),redder>=linear(redder,Ares,Bres-(s*Bres_err)))]
            #clip = ccrt.copy()
            #sig = Bres_err
            md_list = model_dist(Ares,Bres,np.array(redder),real_yrcs)    #take ortogonal distance to model
            devs = np.array(md_list)
            ccrt["devs"] = devs
            #sig = model_dist(Ares,Bres,np.array([clip["col0"][0]]),linear(np.array([clip["col0"][0]]),Ares,Bres+(1*sig)))[0]    #convert Bres_err to ortogonal distance 
            if sig < 0.01:
                sig = 0.06
            #if sig >= 0.22:
            #    sig = 0.09
            
            noclip = ccrt[abs(ccrt["devs"])<=(1*sig)]     
            ccrt.remove_column("devs")
            clip = ccrt.copy()
            ####doublepeaksaves###
            noclip_dlst += [noclip]
            pz_dlst += [pz]
            Ares_dlst += [Ares]
            Bres_dlst += [Bres]
            iAres_dlst += [iAres]
            iBres_dlst += [iBres]
            moffset_dlst += [moffset]            
            sig_dlst += [sig]
        
        #main_ipz = pz_dlst.argmax()      #take the furthest redshift as main redshift, everything else is secondary
        #rest_ipz = np.array([6,2,3,4]).argsort()[:-1]
        if plots:
            plt.clf()
            fig = plt.figure(figsize=(6, 4))
            gs = fig.add_gridspec(1, 2,  width_ratios=(4, 1), left=0.15, right=0.95, bottom=0.1, top=0.9, wspace=0.05)
            ax = fig.add_subplot(gs[0, 0])
            ax_histy = fig.add_subplot(gs[0, 1], sharey=ax)
            ax_histy.set_xlim([0,np.max(YY)+2])
            ax_histy.set_ylim([ymi,yma])
            ax_histy.barh(XX,YY,ybsize,edgecolor="none",color="black")
            ax_histy.tick_params(axis="y", labelleft=False)
            if list(Lpos) != list(np.ones(len(ccrt))):
                bindat = bindat2
                ax_histy.set_xlabel(r"$W$")
            ax.hist2d(bindat["col0"],bindat["col1"],weights=bindat["col2"],range=[[xmi,xma],[ymi,yma]],bins=[int((xma-xmi)/xbsize),int((yma-ymi)/ybsize)],cmap=plt.cm.gray_r)
            #plt.errorbar(bindat["col0"],bindat["col1"], yerr=ybsize/(bindat["col2"]**2),fmt="none",capsize=5,color="black")
            #plt.scatter(bindat["col0"],bindat["col1"],color="black")
            #plt.errorbar(clip["col0"],clip["col1"], yerr=ybsize/(clip["col2"]**2),fmt="none",capsize=5,color="red")
            for ipz in range(len(pz_dlst)):
                noclip = noclip_dlst[ipz]
                pz = pz_dlst[ipz]
                Ares = Ares_dlst[ipz]
                Bres = Bres_dlst[ipz]
                iAres = iAres_dlst[ipz]
                iBres = iBres_dlst[ipz]
                ax.scatter(noclip["m_"+rlab],noclip["m_"+blab]-noclip["m_"+rlab],s=5,color="red",alpha=0.5)
                if pz_dlst == list(np.ones(len(pz_dlst))*(-1.0)):
                    ax.text(xmi+1,ymi+0.2,s="no RCS model")
                else:
                    ax.plot(np.linspace(xmi,xma),linear(np.linspace(xmi,xma),Ares,Bres),color="red",ls="--",label="RCS MODEL "+str(ipz+1))
                ax.plot(np.linspace(xmi,xma),linear(np.linspace(xmi,xma),iAres,iBres),color="blue",ls="--",label="LINEAL FIT "+str(ipz+1))
                ax.text(xma-0.5,cpeaks[ipz],s=str(ipz+1),color="blue")
            ax.text(xmi+1,yma-0.2,s=str(iter_r2cut)+r"R$_{200}$")

            moftxt = ", ".join(np.array(np.around(moffset_dlst,3),dtype="str"))
            Ntxt = [len(nci) for nci in noclip_dlst]
            Ntxt = ", ".join(np.array(Ntxt,dtype="str"))
            pztxt = ", ".join(np.array(pz_dlst,dtype="str"))
            sigtxt = ", ".join(np.array(np.around(sig_dlst,3),dtype="str"))
            #Ntxt = ", ".join(np.array(np.around(moffset_dlst,3),dtype="str"))
            #moftxt = str(np.around(moffset_dlst[np.array(pz_dlst).argmax()],3))
            #Ntxt = str(len(noclip_dlst[np.array(pz_dlst).argmax()]))
            #pztxt = str(pz_dlst[np.array(pz_dlst).argmax()])
            #sigtxt = str(np.around(sig_dlst[np.array(pz_dlst).argmax()],3))
            #for ipz in np.array(pz_dlst).argsort()[:-1]:
            #    moftxt += ", "+str(np.around(moffset_dlst[ipz],3))
            #    Ntxt += ", "+str(len(noclip_dlst[ipz]))
            #    pztxt += ", "+str(pz_dlst[ipz])
            #    sigtxt += ", "+str(np.around(sig_dlst[ipz],3))

            ax.text(xmi+1,yma-0.34,s=r"$\bar{\Delta}_{fit,model}$ = "+moftxt,color="red")           
            ax.text(xmi+1,yma-0.48,s="N = "+Ntxt,color="red")
            ax.text(xmi+1,yma-0.62,s=r"z$_{binned}$ = "+pztxt,color="red")
            ax.text(xmi+1,ymi+0.2,s=r"$\sigma_{bin}$ = "+sigtxt,color="blue")
            #ax.text(xmi+1,yma-0.2,s=str(np.around(isi,1))+"-SIGMA CLIPPING")
            #ax.text(xmi+1,yma-0.3,s="center: "+str(np.around(rcs,2)))
            #ax.text(xmi+1,yma-0.4,s="sigma: "+str(np.around(sig,4)))
            #ax.fill_between(clip["col0"],overdensity_yrcs-ferr,overdensity_yrcs+ferr,color="red",alpha=0.5)
            ax.set_xlim([xmi,xma])
            ax.set_ylim([ymi,yma])
            ax.set_ylabel(blab+"-"+rlab)
            ax.set_xlabel(rlab)
            ax.set_title(cluster)
            ax.minorticks_on()
            ax.tick_params(which="minor",left=True,bottom=True,right=True,top=True)
            ax.tick_params(top=True,right=True)
            if pz_dlst == list(np.ones(len(pz_dlst))*(-1.0)):
                pass
            else:
                ax.legend(loc="upper right")
            plt.savefig(cluster+"/"+band+"_overdensity.jpg")
            plt.close()

        #for ipz in range(len(pz_dlst)):
        pz_lst += [pz_dlst]
        notrcstab_lst += [noclip_dlst]
        moffset_lst += [moffset_dlst]
        bflag_lst += [bflag]
        rcstab_lst += [clip]
    return rcstab_lst, notrcstab_lst , pz_lst, moffset_lst, bflag_lst

def z_uncer(pz,red,mstar,band):
    print("estimating confidence intervals...")
    rlab = band[1]
    blab = band[0]
    #red = red[np.logical_and(red["mag_"+blab+"_err"]<=0.1,red["mag_"+rlab+"_err"]<=0.1)]  #uncertainty cut
    #with warnings.catch_warnings():  # Ignore warnings
    #    warnings.simplefilter('ignore')
    #    rdepth = -2.5 * (np.log10(10. / np.sqrt(red["galdepth_"+rlab])) - 9)    #now cutting up to max depth
    #    bdepth = -2.5 * (np.log10(10. / np.sqrt(red["galdepth_"+blab])) - 9)
    #red = red[np.logical_and(red["m_"+rlab]<=np.max(rdepth),red["m_"+blab]<=np.max(bdepth))]
    red = red[np.logical_and(red["m_"+rlab]<=mstar+pluscut,red["m_"+rlab]>=mstar+minuscut)]  #faint and bright cut
    #---------finding not-binned RS
    #rcs = 0    #initial values for biweight (these values are not too important)
    #sig = np.sqrt(np.sum(abs(red["devs"])**2)/len(red))
    ##for l in range(3):                                        #this time i dont want to move the center
    ##    rcs = st.biweight_location(devs,M=np.array([rcs]))    #improve initial guess before 3sigma clipping (beers et al 1990)
    #cont = 0
    ##---3sigmaclipping
    #while True:
    #    #rcs = st.biweight_location(clip["devs"],M=np.array([rcs]))  
    #    sig = st.biweight_scale(red["devs"],M=np.array([rcs]))   #---Sigma Biweight
    #    rej = red[abs(red["devs"]-rcs) >= (s*sig)]   #----rejected_galaxies
    #    if len(rej) == 0:      #---when there are no rejected galaxies end the iteration and calculate sigma error
    #        #print("NUMBER OF ITERATIONS: "+str(cont))
    #        break
    #    red = red[abs(red["devs"]-rcs) < (s*sig)] #---if len(rej)!=0 then cut in s*sigma and start over again
    #    cont += 1
    #print("RS color error: "+str(sig))
    #now find which redshift correspond to best model+sig and best model-sig
     
    gredder = np.array(red["m_"+rlab])
    msig_lst = np.sqrt(red["mag_"+rlab+"_err"]**2 + red["mag_"+blab+"_err"]**2)
    rscolup = red.copy()                   #RS colors up by 1sig
    rscolup["col1"] = (red["m_"+blab]-red["m_"+rlab]) + msig_lst
    rscolup["col2"] = msig_lst #(msig_lst/0.06)**(-(1/2))     #fake N to trick chi2min with binned==True
    rscollw = red.copy()                   #RS colors down by 1sig
    rscollw["col1"] = (red["m_"+blab]-red["m_"+rlab]) - msig_lst
    rscollw["col2"] = msig_lst #(msig_lst/0.06)**(-(1/2))
    scmri = cmr[abs(cmr["col1"]-pz)<=0.1]    #look for errors up to +-0.1 from redshift
    scmri = extrapolator(scmri)   #extrapolate more models for extra resolution
    pz_conf = []
    for colerr_model in [rscollw,rscolup]:
        ztab, ult_chitab, errpz, final_res_lst, coef, lab_lst, iterpluscut = chi2min([colerr_model],scmri,iter_init_bands=[band],binned=True,clipping=False,zhint=pz,forcedband=True)
        pz_conf += [errpz]
    return pz_conf

##################START_CODE#####################
#pz_lst = []
#RS_error = []
#richness_lst = []
for RA, DEC, brick_lst, cluster, r200 in zip(init_RA,init_DEC,init_brick_lst,init_cluster,init_r200):   
    print("============"+cluster+"=============")
    print("loading catalogs...")
    if bricky: 
        for i in range(len(brick_lst)):   #load catalogs  ??? (if you have full catalogs instead of brick catalogs, modify this part)
            brick = brick_lst[i]
            if i==0:
                crt = Table.read(cluster+"/"+brick+"/tractor/tractor-"+brick+".fits",format="fits")
                crt = crt[crt["brick_primary"]==True]
                crt = crt[cols]
            else:
                icrt = Table.read(cluster+"/"+brick+"/tractor/tractor-"+brick+".fits",format="fits")
                icrt = icrt[icrt["brick_primary"]==True]
                icrt = icrt[cols]
                crt = vstack([crt,icrt])
    else:
        crt = Table.read(cluster+"/"+cluster+".csv",format="csv")
        crt = crt[crt["brick_primary"]==True]
        crt = crt[cols]      #???  comment this line if you want all available cols
        
    ########################estimate_magnitudes########################
    print("checking magnitudes...")
    with warnings.catch_warnings():  # Ignore warnings
        warnings.simplefilter('ignore')
        crt["m_g"] = 22.5-2.5*np.log10(crt["flux_g"]/crt["mw_transmission_g"])  #estimate magnitudes
        crt["m_r"] = 22.5-2.5*np.log10(crt["flux_r"]/crt["mw_transmission_r"])
        crt["m_i"] = 22.5-2.5*np.log10(crt["flux_i"]/crt["mw_transmission_i"])
        crt["m_z"] = 22.5-2.5*np.log10(crt["flux_z"]/crt["mw_transmission_z"])
        #crt["sig_mag_err"] = -2.5*np.log10((crt["flux_ivar_i"]**(-(1/2)))/crt["mw_transmission_i"])
        
        #crt["m_g"] = (22.5-2.5*np.log10(crt["flux_g"]))#*crt["mw_transmission_r"]  #estimate magnitudes
        #crt["m_r"] = (22.5-2.5*np.log10(crt["flux_r"]))#*crt["mw_transmission_r"]
        #crt["m_i"] = 22.5-2.5*np.log10(crt["flux_i"])
        #crt["m_z"] = 22.5-2.5*np.log10(crt["flux_z"])
        
        crt = crt[np.isnan(crt["m_g"])==False]    #remove nan values, feos los nan values >:L
        crt = crt[np.isnan(crt["m_r"])==False]
        crt = crt[np.isnan(crt["m_i"])==False]
        crt = crt[np.isnan(crt["m_z"])==False]
        
        ig = len(crt)
        crt = crt[np.isinf(crt["m_g"])==False]    #los inf tampoco me sirven
        crt = crt[np.isinf(crt["m_r"])==False]
        crt = crt[np.isinf(crt["m_i"])==False]
        crt = crt[np.isinf(crt["m_z"])==False]
        ig2 = len(crt)
        if ig2!=ig:
            print("INF values detected: "+str(ig-ig2)+" sources removed")
        if ig2<50:
            print("ERROR: TABLE MISSING BAND INFO")
            continue
        x_err = (crt["flux_ivar_g"]**(-(1/2)))#/crt["mw_transmission_g"]   #estimate mag errors
        x = crt["flux_g"]/crt["mw_transmission_g"]
        crt["mag_g_err"] = (x_err/(x*np.log(10)))*2.5
        x_err = (crt["flux_ivar_r"]**(-(1/2)))#/crt["mw_transmission_r"]  
        x = crt["flux_r"]/crt["mw_transmission_r"]
        crt["mag_r_err"] = (x_err/(x*np.log(10)))*2.5
        x_err = (crt["flux_ivar_i"]**(-(1/2)))#/crt["mw_transmission_i"]
        x = crt["flux_i"]/crt["mw_transmission_i"]
        crt["mag_i_err"] = (x_err/(x*np.log(10)))*2.5 
        x_err = (crt["flux_ivar_z"]**(-(1/2)))#/crt["mw_transmission_z"]
        x = crt["flux_z"]/crt["mw_transmission_z"]
        crt["mag_z_err"] = (x_err/(x*np.log(10)))*2.5 
    ################cuts####################
    iter_r2cut = r2cut+0
    ra, dec = RA, DEC
    d = SkyCoord(ra*u.degree,dec*u.degree)     #find Mpc distance and pos angle of each galaxy from the cluster center
    catalog = SkyCoord(np.array(crt["ra"])*u.deg,np.array(crt["dec"])*u.deg)
    lst = d.separation(catalog).to(u.arcmin)
    crt["PRJ_SEP"] = lst
    grcrt = crt.copy()
    crt = crt[crt["type"]!="PSF"]      #remove stars
    crt = crt[crt["type"]!="DUP"]      #remove gaia sources
    crt = crt[crt["fitbits"]!=2**5]   #BRIGHT STAR
    crt = crt[crt["fitbits"]!=2**6]   #MEDIUM STAR
    crt = crt[crt["fitbits"]!=2**7]   #GAIA SOURCE
    crt = crt[crt["fitbits"]!=2**8]   #TYCHO-2 STAR
    crt = crt[crt["fitbits"]!=2**12]  #GAIA POINTSOURCE
    ecrt = setdiff(grcrt,crt,keys="ls_id")
    bkg = crt[crt["PRJ_SEP"]>2*r200]
    bkg = bkg[bkg["PRJ_SEP"]<=4*r200]
    gcrt = crt[crt["PRJ_SEP"]<=r200] 
    scrt = crt[crt["PRJ_SEP"]<=iter_r2cut*r200]    #0.5r200 cut --> hennig17
    while len(scrt)<mingals:      #uwoparche para cumulos flaquitos
        print("less than "+str(mingals)+" galaxies within "+str(iter_r2cut)+"R200, using "+str(iter_r2cut+0.25)+"R200 instead...")
        iter_r2cut += 0.25
        scrt = crt[crt["PRJ_SEP"]<=iter_r2cut*r200]
    #crt = crt[crt["flux_ivar_i"]**(-(1/2))<10**(0.1/2.5)]     #sig_mag < 0.1 cut
    #scrt["N_ID"] = np.linspace(0,len(scrt)-1,len(scrt),dtype="int")   #give identifier label
    #-------------CMD BINNING
    area_factor = ((iter_r2cut*r200)**2)/((4*r200)**2 - (2*r200)**2)
    print("binning CMD...")
    crt_per_band, clip_per_band, bz_per_band, moffset_per_band, flags = overdensity_locator(cluster,scrt,bkg,cmr,area_factor)   #find CMD overdensity and return rcs objects for each band
    
    if flags == list(np.ones(len(flags))*2):
        print("WARNING: LOW SIGNAL CLUSTER.")
        pz = -1.0
        pz_err = 0.0
        pz_lim = "-0.0,+0.0"
        rich = 0.0
        flag = 2
        dtcols = ["str","float","float","str","float","int"]
        try:
            type(pztab)
            exist = True
        except:
            exist = False
        if exist:#cluster == tab[tab.colnames[nameind]][0] and str(resume)=="False":
            res = Table(np.array([cluster,pz,pz_err,pz_lim,rich,flag]),names=[tab.colnames[nameind],"photo-z","pz_err","pz_lim","RS_richness","FLAG"],dtype=dtcols)
            pztab = vstack([pztab,res])
        else:
            pztab = Table(np.array([cluster,pz,pz_err,pz_lim,rich,flag]),names=[tab.colnames[nameind],"photo-z","pz_err","pz_lim","RS_richness","FLAG"],dtype=dtcols)
        continue

    bz_lst_final = []
    scrit_lst = []
    bz_bands = []
    flag_lst = []
    clipsi_lst = []
    for k in range(len(init_bands)):
        bz_dlst = bz_per_band[k]
        clip_dlst = clip_per_band[k]
        mofs_dlst = moffset_per_band[k]
        has_weight = np.array([col_evo_weights(ibz)[k] for ibz in bz_dlst])
        mask = has_weight!=0        
        signif_z = np.array(bz_dlst)[mask]
        if len(signif_z)==0:
            mask = np.array(bz_dlst)!=-1.0
            signif_z = np.array(bz_dlst)[mask]   #use bands with usable redshift (z != -1.0)
        clips = np.array(clip_dlst,dtype="object")[mask]
        ns = np.array([len(t) for t in clip_dlst])[mask]
        mofs = np.array(mofs_dlst)[mask]
        if len(signif_z)==0:
            clips = clip_dlst[0]
            try:
                clips = Table(clips[0])
                clips.remove_row(0)
            except:
                pass
            ns = [len(clips)]
            mofs = [mofs_dlst[0]]
        scrit = ns/np.abs(mofs)        #uwoparche para seleccionar mejor z_binned 
        bz_lst_final += bz_dlst
        scrit_lst += list(scrit)
        bz_bands += [init_bands[k] for t in range(len(bz_dlst))]
        flag_lst += [flags[k] for t in range(len(bz_dlst))]
        clipsi_lst += clip_dlst
    fcase = flag_lst[np.array(scrit_lst).argmax()]
    if fcase==1:
        flag = 1
        bz_lst_final = list(np.array(bz_lst_final)[np.array(scrit_lst).argsort()[-2:]])    #take the two best peaks
        bz_bands = list(np.array(bz_bands)[np.array(scrit_lst).argsort()[-2:]])
        clipsi_lst = list(np.array(clipsi_lst,dtype="O")[np.array(scrit_lst).argsort()[-2:]])
        short_cmr_lst = []
        clipi_shot = []
        for t in range(len(bz_lst_final)):
            ibz = bz_lst_final[t]
            short_cmr = cmr[abs(cmr["col1"]-ibz)<=0.1]
            short_cmr = extrapolator(short_cmr)
            short_cmr_lst += [short_cmr]
            clipi_shot += [[clipsi_lst[t] for k in range(len(init_bands))]]
    else:
        flag = fcase
        bz_lst_final = [bz_lst_final[np.array(scrit_lst).argmax()]]
        bz_bands = [bz_bands[np.array(scrit_lst).argmax()]]
        clipi_shot = [[clipsi_lst[np.array(scrit_lst).argmax()] for t in range(len(init_bands))]]    #select the best sample and use it for all bands
        short_cmr = cmr[abs(cmr["col1"]-bz_lst_final[0])<=0.1]
        short_cmr_lst = [extrapolator(short_cmr)]

    print("overdensity located in")
    print("band: "+" ".join(bz_bands))

    if False:     #plot uncertainties
        for rlab in ["g","r","i","z"]:
            plt.clf()
            ax = plt.axes([0.1,0.1,0.65,0.8])
            ax_hist = plt.axes([0.78,0.1,0.17,0.8])
            ax.scatter(scrt["m_"+rlab],scrt["mag_"+rlab+"_err"],s=5)
            ax.set_xlabel(rlab)
            ax.set_ylabel(r"$\sigma$")
            ax.set_ylim([-0.2,0.5])
            ax.axhline(0.1,color="black",ls="--",label=r"$\sigma_{mag} = 0.1$")
            mgx = np.max(scrt[scrt["mag_"+rlab+"_err"]<=0.1]["m_"+rlab])
            ax.axvline(mgx,color="red",ls=":",label=r"mag($\sigma$=0.1) = "+str(np.around(mgx,2)))
            ax_hist.hist(scrt["mag_"+rlab+"_err"],orientation="horizontal",bins=30,range=ax.set_ylim())
            ax_hist.set_ylim(ax.set_ylim())
            ax.minorticks_on()
            ax.tick_params(which="both",left=True,bottom=True,right=True,top=True)
            ax_hist.minorticks_on()
            ax_hist.tick_params(which="both",labelleft=False,labelbottom=False,bottom=False,left=False,top=False,right=False)
            with warnings.catch_warnings():  # Ignore warnings
                warnings.simplefilter('ignore')
                rdepth = -2.5 * (np.log10(10. / np.sqrt(scrt["galdepth_"+rlab]))-9)
            ax.axvline(np.max(rdepth),color="blue",ls=":",label="depth = "+str(np.around(np.max(rdepth),2)))
            #msnrx = Table(scrt[np.abs(scrt["snr_"+rlab]-10).argmin()])
            #ax.axvline(msnrx["m_"+rlab][0],ls=":",color="green",label="mag(S/N=10) = "+str(np.around(msnrx["m_"+rlab][0],2)))
            #from statsmodels.stats.diagnostic import normal_ad as adtest
            #xad = np.array(scrt["mag_"+rlab+"_err"])
            #dxad = np.array(list(xad) + list(-xad))
            #a,b = 3.6789468, 0.1749916
            #aa, paa = adtest(np.sort(dxad))
            #aastar = (aa)*(1+(0.75/len(dxad)) + (2.25/(len(dxad)**2)))
            #alpha_ad = a*np.exp(((-1)*aastar)/b)
            #ax_hist.text(5,-0.15,s=r"$\alpha_{AD}$ = "+str(np.around(alpha_ad,4)))
            ax.set_title(cluster+", "+str(iter_r2cut)+r"R$_{200}$")
            ax.legend()
            plt.savefig(cluster+"/"+rlab+"_mag_errors.jpg")
            plt.show()

    ###########GET_REDSHIFT#############
    print("finding redshift...")
    chi_res = []
    for t in range(len(bz_lst_final)):
        conttxt = "_z"+str(t+1)
        print("--------z"+str(t+1)+"--------")
        binned_z = bz_lst_final[t]
        print("z_binned = "+str(binned_z))
        clip_per_band = clipi_shot[t]
        short_cmr = short_cmr_lst[t]
        ztab, fullchitab, pz, res_lst, mcoef, lab_lst, ipcut = chi2min(clip_per_band,short_cmr,clipping=False,zhint=binned_z)
        
        #---------bootstraping
        #def onlyz(crate):
        #    crate = Table(crate)
        #    ztab, pz, res_lst, lab_lst = color_to_redshift(crate,cmr=cmr)
        #    return pz
        #st.bootstrap(np.array(crt),100,bootfunc=onlyz)
        
        if plots:      #plot chisquare distribution
            plt.clf()
            ax = plt.axes([0.1,0.1,0.85,0.8])
            col_map = ["indianred","lime","royalblue","magenta"]
            lst = []
            for il in range(len(init_bands)):
                bandl = init_bands[il]
                chiband = fullchitab[fullchitab["band"]==bandl]
                chiband = chiband[chiband["chi2"]!=0.0]
                ax.scatter(chiband["z"],chiband["chi2"],s=5,color=col_map[il],label=bandl[0]+"-"+bandl[1])
                try:
                    lst += [np.min(chiband["chi2"])]
                except:
                    pass
            ax.scatter(ztab["photo-z"],ztab["comb_chi2"],s=20,marker="s",edgecolor="orange",facecolor="none",label="combined")
            ax.axvline(0.75,color="black",ls="--",lw=0.5)
            ax.axvline(0.35,color="black",ls="--",lw=0.5)
            #print(plt.ylim())
            fac = 0.1
            ax.set_ylim([np.min(lst)-((fac)*np.std(fullchitab["chi2"])),np.median(ztab["comb_chi2"])+((2*fac)*np.std(fullchitab["chi2"]))])
            #print(plt.ylim())
            plt.title(cluster+conttxt)
            ax.set_xlabel("z")
            ax.set_ylabel(r"$\chi^{2}$/DOF")
            ax.set_xlim([np.min(ztab["photo-z"]),np.max(ztab["photo-z"])])
            ax.minorticks_on()
            ax.tick_params(which="both",top=True,right=True)
            plt.legend()
            plt.savefig(cluster+"/"+cluster+conttxt+"_chi2.jpg")
            plt.close()

        #pz_lst += [pz]
        #RS_error += [ztab["main_sig"][ztab["photo-z"]==pz][0]]
        rich = ztab["richness"][ztab["photo-z"]==pz][0]
        #richness_lst += [rich]
        #######################SELECT_BEST_RS_MODEL#######################
        icmr = Table(short_cmr[short_cmr["col1"]==pz])
        gcol = list(icmr[icmr.colnames[2::6]][0])    #extract g, r, i, z magnitudes at 6 different levels 3, 2, 1, 0.5, 0.4, 0.3
        rcol = list(icmr[icmr.colnames[3::6]][0])
        icol = list(icmr[icmr.colnames[4::6]][0])
        zcol = list(icmr[icmr.colnames[5::6]][0])
        mstartab = Table(np.array([gcol[2],rcol[2],icol[2],zcol[2]]),names=["g","r","i","z"])
        wid = col_evo_weights(pz)    #select proper band for this redshift
        widi = wid.index(np.max(wid))
        clip = res_lst[widi]    #take clip of the corresponding band, will be of use later
        if len(clip)==0:   #uwoparche para bandas con pesos iguales != 0    #if clip is empty at this point then...
            widisort = np.array(wid).argsort()[-2:]                         #look at the two highest weights
            if wid[widisort[0]]==wid[widisort[1]]:                          #if they have the same weight then...
                widi = widisort[1]              #take the second band (usually the first is taken and the second ignored if equal) 
                clip = res_lst[widi]
        band = init_bands[widi]     #get RS model of pz
        rlab = band[1]
        blab = band[0]
        #-------------------------------------get R200 data within +-0.22 mags from z-model
        gredder = np.array(gcrt["m_"+rlab])   
        gbluer = np.array(gcrt["m_"+blab])
        cmr_coltab = Table([gcol,rcol,icol,zcol],names=["g","r","i","z"])
        gmr = [list(cmr_coltab[cl[1]]) for cl in init_bands]      #sort on an iterable list
        gmb = [list(cmr_coltab[cl[0]]) for cl in init_bands]
        gmredder = gmr[widi]
        gmbluer = gmb[widi]
        params = lmodel.make_params(A=((gmbluer[2]-gmredder[2])-(gmbluer[1]-gmredder[1]))/(gmredder[2]-gmredder[1]),B=0) #initial params
        result = lmodel.fit(np.array(gmbluer)-np.array(gmredder), params, x=gmredder)
        gAres = np.array(result.params)[0]
        gBres = np.array(result.params)[1]
        greal_yrcs = gbluer-gredder
        gmodel_yrcs = linear(gredder,gAres,gBres)
        md_list = model_dist(gAres,gBres,gredder,greal_yrcs)    #take ortogonal distance to model
        devs = np.array(md_list)
        gcrt["devs"] = devs
        redgal = gcrt[np.logical_and(abs(gcrt["devs"])<=0.22,gcrt["m_"+rlab]>=mstartab[rlab][0]-4)]
        redgal.sort("m_"+rlab)                  #select mark BCGs as true
        truecat = np.zeros(len(redgal))
        truecat[:2] = True
        truecat[2:] = False
        redgal["is_bcg"] = Table.Column(truecat,dtype="bool")    #create column denoting BCG
        redgal.write(cluster+"/"+cluster+conttxt+"_redsequence.cat",format="ascii",overwrite=True)   #save table (this are R200 galaxies within +-0.22 color index from best model)
        ######################ERROR ESTIMATION#############################
        #sgcrt = redgal[redgal["PRJ_SEP"]<=iter_r2cut*r200]
        #with warnings.catch_warnings():  # Ignore warnings
        #    warnings.simplefilter('ignore')
        #    rdepth = -2.5 * (np.log10(10. / np.sqrt(sgcrt["galdepth_"+rlab])) - 9)    #now cutting up to max depth
        #    bdepth = -2.5 * (np.log10(10. / np.sqrt(sgcrt["galdepth_"+blab])) - 9)
        #sgcrt = sgcrt[np.logical_and(sgcrt["m_"+rlab]<=np.max(rdepth),sgcrt["m_"+blab]<=np.max(bdepth))]
        conf = z_uncer(pz,clip_per_band[0],mstartab[rlab][0],band)                #error just as sup and inf
        #conf = [pz+(ss*np.around(np.max(abs(conf-pz)),3)) for ss in [-1,1]]    #max error for both inf and sup
        #conf = [pz+(ss*np.around((np.max(conf)-np.min(conf))/2,3)) for ss in [-1,1]]   #error as (sup-inf)/2 for both
        print("photo-z: "+str(pz)+" +"+str(np.around(conf[1]-pz,3))+" "+str(np.around(conf[0]-pz,3)))
        pz_lim = str(np.around(conf[0]-pz,3))+",+"+str(np.around(conf[1]-pz,3))
        pz_err = np.around((conf[1] - conf[0])/2,4)
        #print("photo-z: "+str(pz)+" +- "+str(np.around(pz_err,2)))
        #RS_error += [pz_err]
    
        dtcols = ["str","float","float","str","float","int"]
        try:
            type(pztab)
            exist = True
        except:
            exist = False
        if exist:#cluster == tab[tab.colnames[nameind]][0] and str(resume)=="False":
            res = Table(np.array([cluster,pz,pz_err,pz_lim,rich,flag]),names=[tab.colnames[nameind],"photo-z","pz_err","pz_lim","RS_richness","FLAG"],dtype=dtcols)
            pztab = vstack([pztab,res])
        else:
            pztab = Table(np.array([cluster,pz,pz_err,pz_lim,rich,flag]),names=[tab.colnames[nameind],"photo-z","pz_err","pz_lim","RS_richness","FLAG"],dtype=dtcols)
        #########ds9reg#########
        #brick_lst = ["3356m487","3357m485","3359m487"]
        #brick = brick_lst[1]
        #mecrt = ecrt[ecrt["fitbits"]==2**5]#ecrt[ecrt["flux_g"]>np.mean(ecrt["flux_g"])]
        #if False:#"coadd" in os.listdir(cluster+"/"+brick):
        #    t = open(cluster+"/"+cluster+"_bcg.reg","w")
        #    t.write("# Region file format: DS9 version 4.1\n")
        #    t.write("global color=green dashlist=8 3 width=1 font=\"helvetica 10 normal roman\" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n")
        #    t.write("fk5\n")
        #    t.write("circle("+str(RA)+","+str(DEC)+","+str(r200.to(u.arcsec).value)+"\") # color=red text={R500}\n")
        #    t.write("circle("+str(RA)+","+str(DEC)+","+str((4*r200).to(u.arcsec).value)+"\") # color=yellow\n")     
        #    t.write("circle("+str(RA)+","+str(DEC)+","+str((2*r200).to(u.arcsec).value)+"\") # color=yellow\n")
        #    for r in range(len(redgal)):
        #        if redgal["is_bcg"][r]==True:
        #            t.write("circle("+str(redgal["ra"][r])+","+str(redgal["dec"][r])+",5.000\") # color=red text={m_"+rlab+" = "+str(np.around(redgal["m_"+rlab][r],2))+"}\n")
        #        else: 
        #            t.write("circle("+str(redgal["ra"][r])+","+str(redgal["dec"][r])+",5.000\")\n")
        #    for r in range(len(mecrt)):
        #            t.write("circle("+str(mecrt["ra"][r])+","+str(mecrt["dec"][r])+","+str(mecrt["nea_g"][r]**(1/2))+"\") # color=blue\n")
        #    t.close()
        #    pth = cluster+"/"+brick+"/coadd/"
        #    print("#----------R500 data with BCGs / ds9 region")
        #    print("ds9 -rgb -red "+pth+"legacysurvey-"+brick+"-image-i.fits.fz -linear -scale 99.5 -blue "+pth+"legacysurvey-"+brick+"-image-g.fits.fz -linear -scale 99.5 -green "+pth+"legacysurvey-"+brick+"-image-r.fits.fz -linear -scale 99.5 -region "+cluster+"/"+cluster+"_bcg.reg &")
        ##############################################################
        ##########################PLOTTING############################
        ############################################################## 
        if plots:
            print("plotting...")
            #####################plotting _cmd.jpg#######################
            plt.clf()     
            fig, ax = plt.subplots()#subplot_kw={'aspect': 'equal'})
            ax.scatter(gcrt["m_"+rlab],gcrt["m_"+blab]-gcrt["m_"+rlab],s=5,c="black",label=r"R$_{200}$ galaxies")
            #--------binned RS galaxies
            ybsize = 0.06
            xbsize = 0.6
            ax.hist2d(clip["redder"],clip["real_yrcs"],weights=(clip["devs_err"]/ybsize)**(-1/2),bins=[int((xma-xmi)/xbsize),int((yma-ymi)/ybsize)],range=[[xmi,xma],[ymi,yma]],cmap=plt.cm.Reds,label="binned RS galaxies",zorder=-5)
            #--------mark BCGs
            bcgtab = redgal[redgal["is_bcg"]==True]
            bcgtab.sort("m_"+rlab)
            for r in range(len(bcgtab)):
                ax.scatter(bcgtab["m_"+rlab][r],bcgtab["m_"+blab][r]-bcgtab["m_"+rlab][r],s=20,c="none",edgecolor="black",marker=["^","v"][r],label="BCG "+str(r+1))
            #-------add z20 BCG
            try:
                bz20 = tab[tab[tab.colnames[nameind]]==cluster] 
                ra, dec = bz20[bcgra][0], bz20[bcgdec][0]
                d = SkyCoord(ra*u.degree,dec*u.degree)     #find Mpc distance and pos angle of each galaxy from the cluster center
                catalog = SkyCoord(grcrt["ra"],grcrt["dec"])
                lst = d.separation(catalog).to(u.arcmin)
                grcrt["PRJ_SEP"] = lst
                fbz20 = grcrt[grcrt["PRJ_SEP"]==np.min(grcrt["PRJ_SEP"])]
                ax.scatter(fbz20["m_"+rlab][0],fbz20["m_"+blab][0]-fbz20["m_"+rlab][0],s=20,c="none",edgecolor="cyan",marker="s",label="BCG Z20")
            except:
                pass
            #--------mark RS model
            #ax.scatter(bcg_row["MOF_BDF_MAG_I_CORRECTED"],bcg_row["MOF_BDF_MAG_R_CORRECTED"]-bcg_row["MOF_BDF_MAG_I_CORRECTED"],s=70,marker="^",color="none",edgecolor="black",label="BCG",zorder=5)
            x = np.linspace(np.min(redgal["m_"+rlab]),np.max(redgal["m_"+rlab]),len(redgal))
            ax.plot(x,linear(x,gAres,gBres),color="red",label="BEST RS MODEL, z="+str(pz))
            ax.plot(x,linear(x,gAres,gBres)+0.22,ls="--",color="red")
            ax.plot(x,linear(x,gAres,gBres)-0.22,ls="--",color="red")
            #--------mark mstar+3
            ax.axvline(mstartab[rlab][0]+ipcut,color="black",ls=":",label=r"m$^{*}$+"+str(ipcut)+" = "+str(np.around(mstartab[rlab][0]+ipcut,2)))
            #--------plot params
            ax.set_ylim([ymi,yma])
            ax.set_xlim([xmi,xma])
            ax.tick_params(top=True,right=True)
            ax.tick_params(which="minor",top=True,right=True)
            plt.xlabel(rlab)
            plt.ylabel(blab+"-"+rlab)
            plt.title(cluster+conttxt)
            plt.minorticks_on()
            plt.legend(fontsize=7)
            plt.tight_layout()
            plt.savefig(cluster+"/"+cluster+conttxt+"_cmd.jpg")
            plt.close() 
            #plt.show()
            ##############################################################

    print("DONE!")

######################################END######################################
if resume:    #----resume exception
    tab = Table.read(filename,format=filename.split(".")[1])      #load input table
    init_cluster = list(tab[tab.colnames[nameind]])

pztab_total = pztab.copy()
pztab_lowsig = pztab_total[pztab_total["FLAG"]==2]
if save:
    pztab_lowsig.write("lowsig_cl_info.ascii",format="ascii",overwrite=True)

pztabs = pztab[pztab["FLAG"]==1]
pztabs.sort("photo-z")
pztabs = pztabs[::-1]
if len(pztabs)==0:
    pztab_main = pztabs.copy()
    pztab_sec = pztabs.copy()
    pztab_sec[z_theo] = []
else:
    pztab_main = unique(pztabs,keys=tab.colnames[nameind],keep="first")    #remove double clusters, keep the one with larger z
    pztab_sec = setdiff(pztabs,pztab_main)      #save secondary redshifts
    pztab_sec = join(pztab_sec,tab)
pztab = pztab_total[pztab_total["FLAG"]!=1]   #remove all double clusters and add only the main redshifts
pztab = vstack([pztab,pztab_main])
pztab = pztab[pztab["FLAG"]!=2]   #remove low signal clusters
#----fuse to first table to join all information together
#pztab = kal.copy()   
try:    #try to calibrate errors if posible (dev only)
    pztab["pz_lim"] = [str(float(i.split(",")[0])*uncer_factor)+",+"+str(float(i.split(",")[1].split("+")[1])*uncer_factor) for i in pztab["pz_lim"]]
    pztab["pz_err"] = pztab["pz_err"]*uncer_factor
except:
    pass
pztab_full = join(pztab,tab)    #fuse to cluster table to join all information together
try:
    pztab_full.sort(z_theo)    #try to sort for z theo if available
except:
    pass

#remove catastrophic results
pzt = pztab_full[pztab_full["pz_err"]!=0.0]    
pzt["stat"] = (pzt["photo-z"] - pzt[z_theo])/pzt["pz_err"]
clip = pzt.copy()
rcs = np.median(clip["stat"])    #initial values for biweight (these values are not too important)
sig = 1
for l in range(3):
    rcs = st.biweight_location(clip["stat"],M=np.array([rcs]))    #improve initial guess before 3sigma clipping (beers et al 1990)
#---3sigmaclipping
cont = 0
while True:
    if len(clip)==1:
        break
    rcs = st.biweight_location(clip["stat"],M=np.array([rcs]))
    sig = st.biweight_scale(clip["stat"],M=np.array([rcs]))   #---Sigma Biweight
    rej = clip[abs(clip["stat"]-rcs) >= (s*sig)]   #----rejected_galaxies
    if len(rej) == 0:      #---when there are no rejected galaxies end the iteration and calculate sigma error
        #print("NUMBER OF ITERATIONS: "+str(cont))
        break
    clip = clip[abs(clip["stat"]-rcs) < (s*sig)] #---if len(rej)!=0 then cut in s*sigma and start over again
    cont += 1
clip["dz"] = (clip["photo-z"]-clip[z_theo])/(1+clip[z_theo])
dz = np.around(np.sqrt(np.sum(clip["dz"]**2)/len(clip)),6)   #RMS
clip.remove_column("dz")
#################USEFUL_THINGS###############################


#-----------------plot photo-z vs theoretical-z
if True:   #??? False if no z to compare
    plt.clf()
    ax_scatter = plt.axes([0.1,0.35,0.82,0.56])
    ax_scatter.plot(np.linspace(-1,2),np.linspace(-1,2),color="blue",ls="--")
    #----data with errors
    for ipztab in [pztab_full,pztab_sec]:
        ax_scatter.scatter(ipztab[z_theo],ipztab["photo-z"],s=7,color=[["black","red","cyan"][i] for i in ipztab["FLAG"]])
        #ax_scatter.errorbar(crt[z_theo],crt["photo-z"],yerr=list(crt["pz_err"]),color="black",fmt="none")
        pyerr = [float(ipztab["pz_lim"][i].split(",")[1].split("+")[1]) for i in range(len(ipztab))]
        myerr = [float(ipztab["pz_lim"][i].split(",")[0]) for i in range(len(ipztab))]
        ax_scatter.errorbar(ipztab[z_theo],ipztab["photo-z"],yerr=[list(np.abs(myerr)),pyerr],color="black",fmt="none")
    #--------------------
    ax_scatter.set_xlim([0.0,1.2])
    ax_scatter.set_ylim([0.0,1.2])
    #ax_scatter.set_xlabel("spec-z")
    ax_scatter.set_ylabel("photo-z")
    ax_scatter.tick_params(top=True,right=True,labelbottom=False)
    ax_scatter.tick_params(which="minor",left=True,bottom=True,top=True,right=True)
    ax_scatter.minorticks_on()
    #-----delta-z scatter
    ax_res = plt.axes([0.1,0.1,0.82,0.2])
    #ax_res.scatter(crt["z"],(crt["photo-z"]-crt["z"])/(1+crt["z"]),s=7,color="black")
    ax_res.scatter(pztab_full[z_theo],(pztab_full["photo-z"]-pztab_full[z_theo])/(1+pztab_full[z_theo]),s=7,color="black")
    ax_res.axhline(0,color="blue",ls="--")
    ax_res.set_xlim([0.0,1.2])
    ax_res.set_ylim([-0.1,0.1])
    #yli = np.max(abs((crt["photo-z"]-crt[z_theo])))
    #ax_res.set_ylim([-yli,yli])
    ax_res.set_xlabel(z_theo_name)
    ax_res.set_ylabel(r"$\Delta$z",labelpad=-5)
    ax_res.minorticks_on()
    ax_res.text(0.02,0.11,s="rms = "+str(dz)+"(1+z)")   #RMS
    ax_res.tick_params(right=True,labelright=False,labelleft=True,left=True,top=True,bottom=True)
    ax_res.tick_params(which="minor",right=True,left=True,top=True,bottom=True)
    if save:
        plt.savefig("fullplot.jpg")
    plt.show()

if True:   #plot error distribution
    plt.clf()
    ax = plt.axes([0.1,0.1,0.85,0.85])
    ax.hist(pzt["stat"],bins=40,range=[-10,10],histtype="step",edgecolor="black")
    ax.hist(clip["stat"],bins=40,range=[-10,10],histtype="stepfilled",facecolor="red",edgecolor="black",alpha=0.6)
    ax.set_xlabel(r"(PHOTO_Z - "+z_theo_name+")/PZ_ERR")
    ax.set_ylabel("count.")
    ax.minorticks_on()
    ax.tick_params(which="both",left=True,bottom=True,top=True,right=True)
    plt.text(-9,2.5,s=r"$\mu$ = "+str(np.around(rcs,6)))#str(np.around(np.median(crt["stat"]),6)))
    plt.text(-9,2,s=r"$\sigma$ = "+str(np.around(sig,6)))#str(np.around(np.std(crt["stat"]),6)))
    uncer_factor = sig 
    if save:
        plt.savefig("pz_error_hist.jpg")
    plt.show()

#-----------------plot photo-z vs theoretical-z with bcg offset (use bcg_offset.py first)
if False:   #??? False if no z to compare
    plt.clf()
    ax_scatter = plt.axes([0.1,0.35,0.75,0.56])
    ax_scatter.plot(np.linspace(-1,2),np.linspace(-1,2),color="black",ls="--",lw=0.7,alpha=0.6)
    #----data with errors
    ax_scatter.scatter(kal[z_theo],kal["photo-z"],s=20,edgecolor="black",c=kal["BCG-SZ_offset"],cmap=plt.get_cmap('RdYlBu', 10),alpha=1,zorder=3)
    pyerr = [float(kal["pz_lim"][i].split(",")[1].split("+")[1]) for i in range(len(kal))]
    myerr = [float(kal["pz_lim"][i].split(",")[0]) for i in range(len(kal))]
    ax_scatter.errorbar(kal[z_theo],kal["photo-z"],yerr=[list(np.abs(myerr)),pyerr],color="black",fmt="none",zorder=-5)
    #-------colorbar
    norm = mpl.colors.Normalize(vmin=0,vmax=np.max(kal["BCG-SZ_offset"]))
    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap('RdYlBu', 10),norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm,cax=plt.axes([0.87,0.35,0.03,0.56]))
    plt.text(2.5,0.2,s="BCG-SZ offset [arcmin]",rotation=270)
    #cbar.ax_scatter.set_yticklabels
    #ax_colorbar = plt.axes([0.95,0.35,0.1,0.56])
    #plt.colorbar(ax_scatter)
    #--------------------
    ax_scatter.set_xlim([0.0,1.0])
    ax_scatter.set_ylim([0.0,1.0])
    #ax_scatter.set_xlabel("spec-z")
    ax_scatter.set_ylabel("photo-z")
    ax_scatter.tick_params(top=True,right=True,labelbottom=False)
    ax_scatter.tick_params(which="minor",left=True,bottom=True,top=True,right=True)
    ax_scatter.minorticks_on()
    #-----delta-z scatter
    ax_res = plt.axes([0.1,0.1,0.75,0.2])
    #ax_res.scatter(crt["z"],(crt["photo-z"]-crt["z"])/(1+crt["z"]),s=7,color="black")
    ax_res.scatter(kal[z_theo],(kal["photo-z"]-kal[z_theo])/(1+kal[z_theo]),c=kal["BCG-SZ_offset"],cmap=plt.get_cmap('RdYlBu', 10),alpha=1,s=20,edgecolor="black")
    ax_res.axhline(0,color="blue",ls="--")
    ax_res.set_xlim([0.0,1.0])
    ax_res.set_ylim([-0.1,0.1])
    #yli = np.max(abs((crt["photo-z"]-crt[z_theo])))
    #ax_res.set_ylim([-yli,yli])
    ax_res.set_xlabel(z_theo_name)
    ax_res.set_ylabel(r"$\Delta$z",labelpad=-5)
    ax_res.minorticks_on()
    ax_res.text(0.02,0.11,s="rms = "+str(dz)+"(1+z)")   #RMS
    ax_res.tick_params(right=True,labelright=False,labelleft=True,left=True,top=True,bottom=True)
    ax_res.tick_params(which="minor",right=True,left=True,top=True,bottom=True)
    #plt.colorbar()
    #if save:
    #    plt.savefig("fullplot.jpg")
    plt.show()

############################ADD USEFUL INFO####################################

bt_lst = []
for t in range(2):
    ipztab = [pztab_full,pztab_sec][t]
    for i in range(len(ipztab[ipztab.colnames[0]])):    #add more useful info to final table and save again
         cluster = ipztab[ipztab.colnames[0]][i]
         if i == 0:
             bcgtab = Table.read(cluster+"/"+cluster+"_z"+str(t+1)+"_redsequence.cat",format="ascii")
             try:
                 bcgtab.remove_column("nea_g")
             except:
                 pass
             bcgtab = Table(bcgtab[bcgtab["is_bcg"]=="True"])
         else:
             ibcgtab = Table.read(cluster+"/"+cluster+"_z"+str(t+1)+"_redsequence.cat",format="ascii")
             try:
                 ibcgtab.remove_column("nea_g")
             except:
                 pass
             ibcgtab = Table(ibcgtab[ibcgtab["is_bcg"]=="True"])
             for j in range(len(ibcgtab)):
                 bcgtab.add_row(ibcgtab[j])
    bt_lst += [bcgtab]
bcgtab = vstack([bt_lst[0],bt_lst[1]])    #bcg full + sec
pztab_ult = vstack([pztab_full,pztab_sec])

bcg1 = []
bcg2 = []
for i in range(len(pztab_ult)):
    ibcg = bcgtab[2*i:(2*i+2)]
    band = init_bands[col_evo_weights(pztab_ult["photo-z"][i]).index(np.max(col_evo_weights(pztab_ult["photo-z"][i])))]
    rlab = band[1]
    ibcg.sort("m_"+rlab)
    bcg1 += [str(ibcg["ra"][0])+","+str(ibcg["dec"][0])]
    bcg2 += [str(ibcg["ra"][1])+","+str(ibcg["dec"][1])]
pztab_ult["bcg1"] = bcg1
pztab_ult["bcg2"] = bcg2

kal = pztab_ult.copy()      
kal["z_diff"] = abs(kal["photo-z"] - kal[z_theo])  #??? if no "redshift" or theoretical redshift to compare, comment

#------estimate BCG to theoretical BCG spatial offset   
try:
    kal.sort("photo-z")   #find bcg1 and bcg2 in table (the 2:4 index)
    lst = []
    for i in range(len(kal)):
        catalog = SkyCoord(np.array((','.join(list(kal[i][tab.colnames[raind],tab.colnames[decind]]))).split(','),dtype="float")[::2]*u.deg, np.array((','.join(list(kal[i][tab.colnames[raind],tab.colnames[decind]]))).split(','),dtype="float")[1::2]*u.deg)   #bcgs found by this script (careful with ra dec indexing if adding columns to ztab)
        d = SkyCoord(kal[bcgra][i]*u.deg,kal[bcgdec][i]*u.deg)
        d.separation(catalog)
        lst += [np.min(d.separation(catalog).to(u.arcsec)).value]
    kal["bcg_offset"] = lst
except:
    pass    #pass if no BCG to compare
#-----------------------------------------------
kal.sort("z_diff")


#kal = Table.read("photo-z_nodc.ascii",format="ascii")
bcg1_tab = Table(np.array([[float(i.split(",")[0]),float(i.split(",")[1])] for i in kal["bcg1"]]))
lst = []
for i in range(len(kal)):
    szcen = SkyCoord(kal[tab.colnames[raind]][i]*u.degree,kal[tab.colnames[decind]][i]*u.degree)
    bcgcen = SkyCoord(bcg1_tab["col0"][i]*u.degree,bcg1_tab["col1"][i]*u.degree)
    lst += [szcen.separation(bcgcen).to(u.arcmin).value]
kal["BCG-X[\']"] = lst/kal["R200"]

if True:          #BCG-Xray offset
    plt.clf()    
    plt.hist(kal["BCG-X[\']"],bins=np.arange(0,1.05,0.05))
    plt.xlabel(r"R/R$_{200}$")
    plt.ylabel("N")
    plt.title(str(len(kal))+" clusters")
    if save:
        plt.savefig("bcgoffset.jpg")
    plt.show()

lst = []     #BCG-center offset in kpc 
for i in range(len(kal)):
    ra, dec = tab[tab.colnames[raind],tab.colnames[decind]][list(tab[tab.colnames[nameind]]).index(kal[tab.colnames[nameind]][i])]
    bcgra, bcgdec = [float(kal["bcg1"][i].split(",")[0]),float(kal["bcg1"][i].split(",")[1])]
    d = SkyCoord(ra*u.degree,dec*u.degree)
    bcg = SkyCoord(bcgra*u.degree,bcgdec*u.degree)
    sep = d.separation(bcg).to(u.arcsec)
    d_A = cosmo.angular_diameter_distance(z=kal[z_theo][i])
    td = d_A*(2*np.pi/(360*u.deg))
    td = td.to("kpc arcsec-1")
    #d_A.to("Mpc arcmin-1")
    mpc_sep = sep*td
    lst += [mpc_sep.value]
kal["BCG-X[kpc]"] = lst

if save:   #final save
    print("saving results...")
    kal.write("photo-z.ascii",format="ascii",overwrite=True)
else:
    print("kal: final table")
    print("pztab_lowsig: low signal clusters rejected")


######PLOTTING ALL RS MODELS#####
if False:
    import matplotlib as mpl
    lst = []
    for i in range(len(cmr)):
        pz = cmr["col1"][i]
        gcol = list(cmr[cmr.colnames[2::6]][i])    #extract g, r, i, z magnitudes at 6 different levels 3, 2, 1, 0.5, 0.4, 0.3
        rcol = list(cmr[cmr.colnames[3::6]][i])
        icol = list(cmr[cmr.colnames[4::6]][i])
        zcol = list(cmr[cmr.colnames[5::6]][i])
        cmr_coltab = Table([gcol,rcol,icol,zcol],names=["g","r","i","z"])
        for band in init_bands:
            gmredder = cmr_coltab[band[1]]      #sort on an iterable list
            gmbluer = cmr_coltab[band[0]]
            params = lmodel.make_params(A=((gmbluer[2]-gmredder[2])-(gmbluer[1]-gmredder[1]))/(gmredder[2]-gmredder[1]),B=0) #initial params
            result = lmodel.fit(np.array(gmbluer)-np.array(gmredder), params, x=gmredder)
            gAres = np.array(result.params)[0]
            gBres = np.array(result.params)[1]
            lst += [(pz,gAres,gBres,band)]

    megacmrtab = Table(np.array(lst),names=["photo-z","Ares","Bres","band"])
    mymap = mpl.colors.LinearSegmentedColormap.from_list('mycolors',['blue','red'])
    min, max = (np.min(cmr["col1"]), np.max(cmr["col1"]))
    step = 0.01
    Z = [[0,0],[0,0]]
    levels = np.arange(min,max+step,step)
    CS3 = plt.contourf(Z, levels, cmap=mymap)
    for band in init_bands:
        plt.clf()
        imctab = megacmrtab[megacmrtab["band"]==band]
        for i in range(len(imctab)):
            z = imctab["photo-z"][i]
            r = (float(z)-min)/(max-min)
            g = 0
            b = 1-r
            plt.plot(np.linspace(xmi,xma),linear(np.linspace(xmi,xma),float(imctab["Ares"][i]),float(imctab["Bres"][i])),color=(r,g,b))
        plt.xlim([xmi,xma])
        plt.ylim([ymi,yma])
        plt.ylabel(band[0]+"-"+band[1])
        plt.xlabel(band[1])
        plt.colorbar(CS3,label="photo-z")
        plt.legend()
        plt.show()
    
if False:    #mstar evolution
    gmstar = cmr[cmr.colnames[2::6][2]]    #get m* for each bandi
    rmstar = cmr[cmr.colnames[3::6][2]]
    imstar = cmr[cmr.colnames[4::6][2]]
    zmstar = cmr[cmr.colnames[5::6][2]]
    mstartab = Table([cmr["col1"],gmstar,rmstar,imstar,zmstar],names=["photo-z","g","r","i","z"])
    plt.clf()
    col_map = ["indianred","lime","royalblue","magenta"]
    for band in init_bands:
        rlab = band[1]
        blab = band[0]
        plt.scatter(mstartab["photo-z"],mstartab[blab]-mstartab[rlab],color=col_map[init_bands.index(band)],label=blab+"-"+rlab,s=5)
    plt.legend()
    plt.xlabel("redshift")
    plt.ylabel("color")
    plt.title(r"m$^{*}$ color evolution")
    plt.show()

