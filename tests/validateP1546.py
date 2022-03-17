# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 15:50:46 2014
  This script is used to validate an implementation of ITU-R P.1546
  recommendation as defined in the function P1546FieldStrMixed.m
  This script reads the test profiles from the folder <pathname>
  and for the input parameters defined in those files
  (in Fryderyk_csv format) computes the field strength and logs all the
  intermediate results in <out_dir>\*log.csv files.
 
  Author: Ivica Stevanovic (IS), Federal Office of Communications, Switzerland
  Revision History:
  Date            Revision
  16Mar2022       Updates related to Recommendation ITU-R P.1546-6  
  13May2016       Introduced pathinfo flag (IS)
  29May2015       Modified backward to forward stroke so the code runs on
                  Linux as suggested by M. Rohner (IS) 
  29Apr2015       Introduced 'default' as an option in ClutterCode (IS)
  26Nov2014       Initial version (IS)

"""
import csv
import sys
import os
from traceback import print_last
import numpy as np
import matplotlib.pyplot as pl


from Py1546 import P1546

# path to the folder containing test profiles 
pathname = './validation_profiles/'
 
 # path to the folder where the resulting log files will be saved
out_dir = './validation_results/'

 # format of the test profile (measurement) files
fileformat='Fryderyk_csv'

# Clutter code type 
ClutterCode = 'P1546'

#     ClutterCode='default';  # default clutter code assumes land, rural area with R1 = R2 = 10;
#     ClutterCode='TBD'
#     ClutterCode='OFCOM'
#     ClutterCode='NLCD'
#     ClutterCode='LULC'
#     ClutterCode='GlobCover'
#     ClutterCode='DNR1812'
#     ClutterCode = 'P1546'

# set to 1 if the csv log files need to be produced (together with stdout)
flag_debug = 1

# set to 1 if the plots of the height profile are to be shown
flag_plot = 0

# pathprofile is available (=1), not available (=0)
flag_path = 1

# Dimension of a square area for variability calculation
wa = 500

# begin code
# Collect all the filenames .csv in the folder pathname that contain the profile data
try:
    filenames = [f for f in os.listdir(pathname) if f.endswith('.csv')]
except:
    print ("The system cannot find the given folder " + pathname)

# create the output directory

try: 
    os.makedirs(out_dir)
except OSError:
    if not os.path.isdir(out_dir):
        raise

if (flag_debug==1):
    fid_all = open( out_dir + 'combined_results.csv', 'w')
    if (fid_all == -1):
        raise IOError('The file combined_results.csv could not be opened')
       
    fid_all.write('# # %s, %s, %s, %s, %s, %s,\n' % ('Folder','Filename','Dataset #','Reference','Predicted','Deviation: Predicted-Reference'))
if (len(filenames) < 1):
    raise IOError('There are no .csv files in the test profile folder ' + pathname)

# figure counter
fig_cnt = 0

for filename1 in filenames:
    
    print ('***********************************************\n')
    print ('Processing file ' + pathname + filename1 + '\n')
    print ('***********************************************\n')
    
     # read the file and populate sg3db input data structure
    
    sg3db = P1546.read_sg3_measurements2( pathname + filename1,fileformat)
   
    # collect intermediate results in log files (=1), or not (=0)
    sg3db.debug = flag_debug
    
    # pathprofile is available (=1), not available (=0)
    sg3db.pathinfo = flag_path
    
    # update the data structure with the Tx Power (kW)
    for kindex in range(0, sg3db.Ndata):
        PERP= sg3db.ERPMaxTotal[kindex]
        HRED= sg3db.HRPred[kindex]
        PkW = 10.0**(PERP/10.0)*1e-3  #kW
        
        if(np.isnan(PkW)):
            # use complementary information from Basic Transmission Loss and
            # received measured strength to compute the transmitter power + gain
            E = sg3db.MeasuredFieldStrength[kindex]
            PL = sg3db.BasicTransmissionLoss[kindex]
            f = sg3db.frequency[kindex]
            PdBkW = -137.2217+E-20*np.log10(f)+PL
            PkW=10**(PdBkW/10.0)
        
        
        sg3db.TransmittedPower = np.append(sg3db.TransmittedPower, PkW)
    
    
    # discriminate land and sea portions
    dland=0
    dsea=0

    if(len(sg3db.radio_met_code)>0 and len(sg3db.coveragecode)>0):
        for i in range(0,len(sg3db.x)):
            if (i==len(sg3db.x)-1):
                dinc=(sg3db.x[-1]-sg3db.x[-2])/2.0
            elif (i==0):
                dinc=(sg3db.x[1]-sg3db.x[0])/2.0
            else:
                dinc=(sg3db.x[i+1]- sg3db.x[i-1])/2.0
            
            
            if ( sg3db.radio_met_code[i]==1 or  sg3db.radio_met_code[i]==3):  #sea and coastal land
                dsea=dsea+dinc
            else:
                dland=dland+dinc
            
    elif(len( sg3db.radio_met_code)==0 and len( sg3db.coveragecode)>0):
        for i in range(0,len( sg3db.x)):
            if (i==len( sg3db.x)-1):
                dinc=( sg3db.x[-1]- sg3db.x[-2])/2.0
            elif (i==0):
                dinc=( sg3db.x[1]- sg3db.x[0])/2.0
            else:
                dinc=( sg3db.x[i+1] - sg3db.x[i-1])/2.0
            
            if ( sg3db.coveragecode[i]==2):  #sea - when radio-met code is missing, it is supposed that the file is organized as in DNR p.1812...
                dsea=dsea+dinc
            else:
                dland=dland+dinc
           
    else:
        dland=np.nan
        dsea =np.nan
   
    hTx = sg3db.hTx
    hRx = sg3db.hRx
        
    for measID in range(0,len(hRx)):
        print ('Computing the fields for Dataset # %d\n' % (measID))

        # Determine clutter heights
        if( len(sg3db.coveragecode) >0 ):
                            
            i=sg3db.coveragecode[-1]
            RxClutterCode, RxP1546Clutter, R2external = P1546.clutter(i, ClutterCode)
            i=sg3db.coveragecode[0]
            TxClutterCode, TxP1546Clutter, R1external = P1546.clutter(i, ClutterCode)
            
            if TxP1546Clutter.find('Rural') != -1: # do not apply clutter correction at the transmitter side
                R1external = 0
            
            # if clutter heights are specified in the input file, use those instead of representative clutter heights
            if((np.size(sg3db.h_ground_cover) != 0) and (ClutterCode.find('default') == -1 ) ):
                if(not np.isnan(sg3db.h_ground_cover[-1])):
                    sg3db.RxClutterHeight = sg3db.h_ground_cover[-1]
                    
                else:
                    sg3db.RxClutterHeight = R2external
                
                
                if( not np.isnan(sg3db.h_ground_cover[0])):
                    sg3db.TxClutterHeight = sg3db.h_ground_cover[0]
                    
                else:
                    sg3db.TxClutterHeight = R1external
         
            else:
                sg3db.RxClutterHeight = R2external
                sg3db.TxClutterHeight = R1external
            
        else:                
            
            # cov-code is empty, use default
            
            [RxClutterCode, RxP1546Clutter, R2external] = P1546.clutter(1, ClutterCode)
            
            [TxClutterCode, TxP1546Clutter, R1external] = P1546.clutter(1, ClutterCode)
                
           
            sg3db.RxClutterCodeP1546 = RxP1546Clutter
            sg3db.RxClutterHeight = R2external
            sg3db.TxClutterHeight = R1external
            
        
        xx= sg3db.x[-1]- sg3db.x[0]

        sg3db.LandPath=dland
        sg3db.SeaPath=dsea         

        # implementation of P1546-6 Annex 5 Paragraph 1.1
        # if both terminals are at or below the levels of clutter in their respective vicinities,
        # then the terminal  with the greater height above ground should be treated as the transmitting/base station
        # Once the clutter has been chosen, the second terminal becomes a
        # transmitter in the following cases according to S5 1.11
        # a) both 1 and 2 are below clutter (h1<R1, h2<R2) and h2>h1
        # b) 2 is above clutter and 1 is below clutter (h1<R1, h2>R2)
        # c) both 1 and 2 are above clutter (h1>R1 h2>R2) and h2eff > h1eff
        
        hhRx=hRx[measID]
        hhTx=hTx[measID]
        
        x=sg3db.x
        h_gamsl=sg3db.h_gamsl
        
        x_swapped = x[-1]-x[::-1]
        h_gamsl_swapped = h_gamsl[::-1]
        
        swap_flag = False
        heff=P1546.heffCalc(x,h_gamsl,hTx[measID])
        heff_swapped=P1546.heffCalc(x_swapped,h_gamsl_swapped, hRx[measID])

        if (sg3db.first_point_transmitter == 0):
            swap_flag = True
        
       
        TxSiteName = sg3db.TxSiteName
        RxSiteName = sg3db.RxSiteName

            
        if (swap_flag):
            # exchange the positions of Tx and Rx
            
            print('Annex 5 Paragraph 1.1 applied, terminals are swapped.\n')

            dummy = hhRx
            hhRx = hhTx
            hhTx = dummy
            
            x = x_swapped
            h_gamsl = h_gamsl_swapped
            
            dummy = sg3db.TxClutterHeight
            sg3db.TxClutterHeight = sg3db.RxClutterHeight
            sg3db.RxClutterHeight = dummy
            
            TxSiteName = sg3db.RxSiteName
            RxSiteName = sg3db.TxSiteName
            
            dummy = RxP1546Clutter
            RxP1546Clutter = TxP1546Clutter
            TxP1546Clutter = dummy
            
            dummy = RxClutterCode
            RxClutterCode = TxClutterCode
            TxClutterCode = dummy
    
        sg3db.h2= hhRx
        sg3db.ha= hhTx
        
        # path info is available (in the sg3db files)
        sg3db.htter = h_gamsl[0]
        sg3db.hrter = h_gamsl[-1]
    
       
        sg3db.RxClutterCodeP1546 = RxP1546Clutter

        # # plot the profile
        if (flag_plot):
            fig_cnt = fig_cnt + 1
            newfig = pl.figure(fig_cnt)
            h_plot = pl.plot(x,h_gamsl,linewidth = 2,color = 'k')
            pl.xlim(np.min(x), np.max(x))
            hTx = sg3db.hTx
            hRx = sg3db.hRx
            
            pl.title('Tx: ' + sg3db.TxSiteName + ', Rx: '  + sg3db.RxSiteName + ', ' +  sg3db.TxCountry +  sg3db.MeasurementFileName)
            pl.grid(True)
            pl.xlabel('distance [km]')
            pl.ylabel('height [m]')
                
    
        # # plot the position of transmitter/receiver
    
        hTx=sg3db.hTx
        hRx=sg3db.hRx
        
        if(flag_plot):
            ax=pl.gca()

        if(measID != []):
            if (measID > len(hRx) or measID < 0):
                raise ValueError('The chosen dataset does not exist.')
            
            sg3db.userChoiceInt = measID
            # this will be a separate function
                # Transmitter
            if(flag_plot):
                if (sg3db.first_point_transmitter == 1):
                    pl.plot(np.array([ x[0], x[0]]), np.array([h_gamsl[0], h_gamsl[0]+ hhTx]),linewidth = 2, color = 'b')
                    pl.plot(x[0], h_gamsl[0]+hTx[0], marker='v',color='b')
                    pl.plot(np.array([ x[-1], x[-1]]), np.array([h_gamsl[-1],h_gamsl[-1]+hhRx]),linewidth = 2,color = 'r')
                    pl.plot(x[-1], h_gamsl[-1]+hhRx, marker = 'v',color = 'r')
                else:
                    pl.plot(np.array([ x[-1], x[-1]]), np.array([h_gamsl[-1],h_gamsl[-1]+hhTx]),linewidth = 2,color ='b')
                    pl.plot(x[-1], h_gamsl[0]+hTx[0], marker='v',color  ='b')
                    pl.plot(np.array([ x[0], x[0] ]), np.array([h_gamsl[0],h_gamsl[0]+hhRx]),linewidth = 2,color = 'r')
                    pl.plot(x[0], h_gamsl[0]+hhRx, marker = 'v',color = 'r')
        
                ax = pl.gca()
          
        
        if (measID != []):    
            #if(get(handles.heffCheck,'Value'))
            heff=P1546.heffCalc(x,h_gamsl,hhTx)
            sg3db.heff = heff
            
            # compute the terrain clearance angle
            tca = P1546.tcaCalc(x,h_gamsl,hhRx,hhTx)
            sg3db.tca = tca
            
            if(flag_plot):
                P1546.plotTca(ax,x,h_gamsl,hhRx,tca)

            # compute the terrain clearance angle at transmitter side
            teff1 = P1546.teff1Calc(x,h_gamsl,hhTx,hhRx)
            sg3db.eff1 = teff1
            if(flag_plot):
                P1546.plotTeff1(ax,x,h_gamsl,hhTx,teff1)
                

            # plot the average height above the ground
            if (flag_plot):
                x1=x[0]
                x2=x[-1]
                if x2 > 15:
                    x2=15
                
                yy=ax.get_ylim()
                dy=yy[1]-yy[0]
                y1=hhTx+h_gamsl[0]-heff
                y2=y1
                
                pl.plot(np.array([x1, x2]), np.array([y1, y2]),color = 'r')


                if x[-1] < 15:
                    
                    pl.text((x1+x2)/2,y1+0.05*dy,'hav(0.2d,d) = '+ str(y1))
                else:
                    
                    pl.text(x2,y1+0.05*dy,'hav(3,15) = ' + str(y1))
                #pl.show()   
                pl.draw()
                pl.pause(0.01)
                #input("Press [enter] to continue.")


        
         # Execute P.1546
        fid_log = -1
        if (flag_debug==1):

            filename2 = out_dir + filename1[0:-4] + '_' + str(measID) + '_log.csv'
            fid_log = open(filename2, 'w')
            if (fid_log == -1):
                error_str = filename2 + ' cannot be opened.'
                raise IOError(error_str)
            
        
        sg3db.fid_log = fid_log
        sg3db.wa = wa
        
        sg3db = P1546.Compute(sg3db)
     
        
        if (flag_debug):
            fid_log.close()
  
             # print the deviation of the predicted from the measured value,
             # double check this line
             # Measurement folder | Measurement File | Dataset | Measured Field Strength | Predicted Field Strength | Deviation from Measurement
            fid_all.write(' %s, %s, %d, %.2f, %.2f, %.2f\n' % (sg3db.MeasurementFolder,sg3db.MeasurementFileName,measID, sg3db.MeasuredFieldStrength[measID], sg3db.PredictedFieldStrength, sg3db.PredictedFieldStrength - sg3db.MeasuredFieldStrength[measID]))
            #print(' %s, %s, %d, %.2f, %.2f, %.2f\n' % (sg3db.MeasurementFolder,sg3db.MeasurementFileName,measID, sg3db.MeasuredFieldStrength[measID], sg3db.PredictedFieldStrength, sg3db.PredictedFieldStrength - sg3db.MeasuredFieldStrength[measID]))

if (flag_debug==1):
    fid_all.close()


