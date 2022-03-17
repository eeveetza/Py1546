# -*- coding: utf-8 -*-
"""
Created on Mon May 23 09:56:54 2016

@author: eeveetza
"""

import os
import math as mt
import datetime
from importlib.resources import files

import numpy as np
import matplotlib.pyplot as pl

exceltables = {}
with np.load(files("Py1546").joinpath("P1546.npz")) as exceltablesNpz:
    exceltables = exceltablesNpz['data'].copy()


def bt_loss(f, t, heff, h2, R2, area, d_v, path_c, pathinfo, *args):
    """
    P1546.bt_loss: Basic tranmission loss calculation according to Recommendation ITU-R P.1546-6

    E, L = P1546.bt_loss(f,t,heff,h2,R2,area,d_v,path_c,pathinfo,varargin)

    where:    Units,  Definition                             Limits
    f:        MHz     Required frequency                     30 MHz - 4000 MHz
    t:        %       Required percentage time               1% - 50 %
    heff:     m       the effective height of the
                        transmitting/base antenna, height over
                        the average level of the ground between
                        distances of 3 and 15 km from the
                        transmitting/base antenna in the
                        direction of the receiving/mobile antenna.
    h2:       m       Receiving/mobile antenna height above ground
    R2:       m       Representative clutter height around receiver
                        Typical values:
                        R2=10 for area='Rural' or 'Suburban' or 'Sea'
                        R2=20 for area='Urban'
                        R2=30 for area='Dense Urban'
    area:     string  area around the receiver              'Rural', 'Urban',
                                                            'Dense Urban'
                                                            'Sea'
    d_v:      km      Array of horizontal path lenghts      1 km <= sum(d) <= 1000 km
                        over different path zones starting
                        from transmitter/base terminal
    path_c:   string  Array of strings defining the path      'Land', 'Sea',
                        zone for a given path length in d      'Warm', 'Cold'
                        starting from transmitter/base terminal
    pathinfo: 0/1     0 - no terrain profile information available,
                      1 - terrain information available
    q:        %       Location variability (default 50%)      1% - 99%
    wa:       m       the width of the square area over which the variability applies (m)
                        needs to be defined only if pathinfo is 1 and q <> 50
                        typically in the range between 50 m and 1000 m
    PTx:      kW      Transmitter (e.r.p) power in kW (default 1 kW)
    ha:       m       Transmitter antenna height above        > 1 m
                        ground. Defined in Annex 5 sec 3.1.1.
                        Limits are defined in Annex 5 sec 3.
    hb:       m       Height of transmitter/base antenna
                        above terrain height averaged
                        0.2 d and d km, where d is less than
                        15 km and where terrain information
                        is available.
    R1:       m       Representative clutter height around transmitter
    tca:      deg     Terrain clearance angle                 0.55 - 40 deg
    htter:    m       Terrain heights in meters above
                        sea level at the transmitter/base
    hrter:    m       Terrain height in meters above
                        sea level at the receiver/base
    eff1:     deg     the h1 terminal's terrain clearance
                        angle calculated using the method in
                        Paragraph 4.3 case a, whether or not h1 is
                        negative
    eff2:     deg     the h2 terminal's clearance angle
                        as calculated in Paragraph 11, noting that
                        this is the elevation angle relative to
                        the local horizontal
    debug:    0/1     Set to 1 if the log files are to be written, otherwise
                        set to default 0
    fidlog:           if debug == 1, file identifier of the log file can be
                        provided, if not, the default file with a random file name will be
                        written

    This function implements Recommendation ITU-R P.1546-6 recommendation
    describing a method for point-to-area radio propagation predictions for
    terrestrial services in the frequency range 30 MHz to 4000 MHz. It is
    intended for use on tropospheric radio circuits over land paths, sea paths,
    and/or mixed land-sea paths up to 1000 km length for effective
    transmitting antenna heights less than 3000 m. The method is based on
    interpolation/extrapolation from empirically derived field-strength
    curves as functions of distance, antenna height, frequency, and percentage
    time. The caluculation procedure also includes corrections to the results
    obtained from this interpolation/extrapolation to account for terrain
    clearance, terminal clutter obstructions, and location variability.

    Notes:
    If sea path is selected for a t value less then 50% the default 10% table
    use is a cold sea path.

    Not implemented in this version of the code:
        - Annex 7: Adjustment for different climatic regions
        - Annex 5, Section 4.3a): C_h1 calculation (terrain database is
        available and the potential of discontinuities around h1 = 0 is of no
        concern)

    How to use:
    The function can be called only by using the first 9 required input
    parameters:

    E, L=P1546.bt_loss(2700,50,1600,1.5,10,'Suburban',[20],['Land'],1)

    Or calling all the input parameters:

    E, L=P1546.bt_loss(f,t,heff,h2,R2,area,d_v,path_c,pathinfo,q,wa,PTx,ha,hb,R1,tca,htter,hrter,eff1,eff2,debug,fid_log)

    To use the function and have no input for a variable, the "standard" way
    is pass empty array (or empty cell) for undefined inputs:

    E=P1546.bt_loss(2700,50,1600,1.5,10,'Suburban',[20],['Land'],1,50,500,1,[],[],[],1.2,[],[],[],[])


    Numbers refer to Rec. ITU-R P.1546-6

    Rev   Date        Author                          Description
    -------------------------------------------------------------------------------
    v4    14NOV23     Ivica Stevanovic, OFCOM         Moved Exceltables to P1546.npz file
    v3    16MAR22     Ivica Stevanovic, OFCOM         Update corresponding to ITU-R P.1546-6
    v2    18MAY16     Ivica Stevanovic, OFCOM         Update corresponding to the MATLAB version
                                                        v9 (13MAY16)
    v1    03DEC13     Ivica Stevanovic, OFCOM         First implementation in python


    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
    IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
    OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
    ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
    OTHER DEALINGS IN THE SOFTWARE.

    THE AUTHORS AND OFCOM (CH) DO NOT PROVIDE ANY SUPPORT FOR THIS SOFTWARE


    """

    # Read the input arguments and check them

    if len(args) > 23:
        print("P1546.bt_loss: Too many input arguments, The function requires at most 22")
        print("input arguments. Additional values ignored. Input values may be wrongly assigned.")

    if len(args) < 9:
        raise ValueError("P1546.bt_loss: Function requires at least 9 input parameters.")

    is_out_of_bounds(f, 30, 4000, "f")

    if is_out_of_bounds(t, 1, 50, "t"):
        raise ValueError("Out of bounds")

    # path is is defined as an array of path types

    d_np = np.array(d_v)
    dtot = d_np.sum()

    if is_out_of_bounds(dtot, 0, 1000, "dtot"):
        raise ValueError("Out of bounds")

    NN = len(d_np)

    # the number of elements in d and path need to be the same
    if len(path_c) != NN:
        raise ValueError('The number of elements in the array "d" and cell "path" must be the same.')

    # Optional arguments

    ha = []
    hb = []
    R1 = []
    tca = []
    htter = []
    hrter = []
    eff1 = []
    eff2 = []
    wa = []
    q = 50
    PTx = 1
    debug = 0
    fid_log = []

    icount = 9
    nargin = icount + len(args)
    if nargin >= icount + 1:
        q = args[0]
        if nargin >= icount + 2:
            wa = args[1]
            if nargin >= icount + 3:
                PTx = args[2]
                if nargin >= icount + 4:
                    ha = args[3]
                    if nargin >= icount + 5:
                        hb = args[4]
                        if nargin >= icount + 6:
                            R1 = args[5]
                            if nargin >= icount + 7:
                                tca = args[6]
                                if nargin >= icount + 8:
                                    htter = args[7]
                                    if nargin >= icount + 9:
                                        hrter = args[8]
                                        if nargin >= icount + 10:
                                            eff1 = args[9]
                                            if nargin >= icount + 11:
                                                eff2 = args[10]
                                                if nargin >= icount + 12:
                                                    debug = args[11]
                                                    if nargin >= icount + 13:
                                                        fid_log = args[12]

    # handle number fid_log is reserved here for writing the files
    # if fid_log is already open outside of this function, the file needs to be
    # empty (nothing written), otherwise it will be closed and opened again
    # if fid is not open, then a file with a name corresponding to the time
    # stamp will be opened and closed within this function.

    inside_file = 0
    if debug == 1:
        if isempty(fid_log):
            fid_log = open("P1546_" + datetime.datetime.now().strftime("%y%m%d%H%M%S%f") + "_log.csv", "w")
            inside_file = 1
            if fid_log == -1:
                raise ValueError("The log file could not be opened.")

        else:
            inside_file = 0

    #              time, path, freq, figure (as defined in ITU-R P.1546-6)
    figure_rec_array = [
        [50, 1, 100, 1],
        [10, 1, 100, 2],
        [1, 1, 100, 3],
        [50, 2, 100, 4],
        [10, 3, 100, 5],
        [1, 3, 100, 6],
        [10, 4, 100, 7],
        [1, 4, 100, 8],
        [50, 1, 600, 9],
        [10, 1, 600, 10],
        [1, 1, 600, 11],
        [50, 2, 600, 12],
        [10, 3, 600, 13],
        [1, 3, 600, 14],
        [10, 4, 600, 15],
        [1, 4, 600, 16],
        [50, 1, 2000, 17],
        [10, 1, 2000, 18],
        [1, 1, 2000, 19],
        [50, 2, 2000, 20],
        [10, 3, 2000, 21],
        [1, 3, 2000, 22],
        [10, 4, 2000, 23],
        [1, 4, 2000, 24],
    ]

    figure_rec = np.mat(figure_rec_array)

    # 3 Determination of transmitting/base antenna height, h1
    # In case of mixed paths, h1 should be calculated using Annex 5, sec. 3
    # taking the height of any sea surface as though land. Normally this value
    # of h1 will be used for both Eland(d) and Esea(d).
    # HOWEVER, if h1 < 3m it should
    # be used as such for Eland, but a value of 3 m should be used for Esea(d)

    if NN > 1:  # mixed paths
        path = "Land"
    else:  # single path zone, Land or [Cold or Warm] Sea
        path = path_c[0]

    if path.lower() == "warm" or path.lower() == "cold" or path.lower() == "sea":
        generalPath = "Sea"
        (idx, idxn) = np.where((figure_rec[:, 1] > 1))
        fig = figure_rec[idx, :]
    else:
        generalPath = "Land"
        (idx, idxn) = np.where((figure_rec[:, 1] == 1))
        fig = figure_rec[idx, :]

    h1 = h1_calc(dtot, heff, ha, hb, generalPath, pathinfo)

    if h1 > 3000:
        h1 = 3000
        print("P1546.bt_loss warning: h1 > 3000 m. Setting h1 = 3000 m")

    if np.isnan(h1):
        raise ValueError("P1546.bt_loss error: h1 is nan")

    if isempty(h1):
        raise ValueError("P1546.bt_loss error: h1 is empty")

    if pathinfo == 1 and q != 50:
        if np.isnan(wa) or isempty(wa) or wa <= 0:
            raise ValueError(' "wa" needs to be defined when path is known (pathinfo = 1)')

    Epath = []

    # In case a propagation path is mixed, check if there exist warm and cold
    # path types. In that case, all cold types should be regarded as warm
    # types

    iswarm = False
    iscold = False

    if NN > 1:
        for ii in range(0, NN):
            if path_c[ii].lower().find("warm") != -1:
                iswarm = True

            if path_c[ii].lower().find("cold") != -1:
                iscold = True

        if iswarm and iscold:
            for ii in range(0, NN):
                if path_c[ii].lower().find("cold") != -1:
                    path_c[ii] = "Warm"

    # Step 1: Determine the type of the propagation path as land, cold sea or
    # warm sea. If the path is mixed then determine two path types which are
    # regarded as first and second propagation types. If the path can be
    # represented by a single type then this is regarded as the first
    # propagation type and the mixed-path method given in Step 11 is not
    # required.
    #  Where,
    #  time is percentage 1%,10%,50%
    #  path is 1 = Land, 2 = Sea, 3 = Cold Sea, 4 = Warm Sea
    #  figure is excel figure number 1 - 24

    # Step 2: For any given percentage of time (in the range 1% to 50% time)
    # determine two nominal time percentages as follows:
    # - wanted time percentage > 1 and < 10, the lower and higher nominal
    #   percentages are 1 and 10, respectively;
    # - wanted time percentage > 10 and < 50, the lower and higher nominal
    #   percentages are 10 and 50, respectively.
    # If the required percentage of time is equal to 1% or 10% or 50%, this
    # value should be regarded as the lower nominal percentage time and the
    # interpolation process of Step 10 is not required.
    #
    #        done in start of step 6-10
    #
    # Step 3: For any wanted frequency (in the range 30 to 3 000 MHz) determine
    # two nominal frequencies as follows:
    # - where the wanted frequency < 600 MHz, the lower and higher nominal
    #   frequencies are 100 and 600 MHz, respectively;
    # - where the wanted frequency > 600 MHz, the lower and higher nominal
    #   frequencies are 600 and 2 000 MHz, respectively.
    # If the wanted frequency equals 100 or 600 or 2 000 MHz, this value should
    # be regarded as the lower nominal frequency and the
    # interpolation/extrapolation process of Step 9 is not required.
    #
    #       done in start of step normal step 7-9
    #
    # Step 4: Determine the lower and higher nominal distances from Table 1
    # closest to the required distance. If the required distance coincides with
    # a value in Table 1, this should be regarded as the lower nominal distance
    # and the interpolation process of Step 8.1.5 is not required.

    d = dtot

    dinf, dsup = find_d_nominals(d)

    # if (d> dsup or d< dinf):
    #    print dinf, d, dsup
    #    return

    # In case the field needs to be computed for distances < 1 km

    if dtot < 1:
        dinf = 1
        dsup = 1

    El = []
    dl = []
    Es = []
    ds = []

    path_sea_str = ""

    for ii in range(0, NN):
        path = path_c[ii]
        if path.lower().find("land") != -1:
            dl.append(d_v[ii])
        elif path.lower().find("sea") != -1:
            ds.append(d_v[ii])
            path_sea_str = "Cold"
        elif path.lower().find("cold") != -1:
            ds.append(d_v[ii])
            path_sea_str = "Cold"
        elif path.lower().find("warm") != -1:
            ds.append(d_v[ii])
            path_sea_str = "Warm"
        else:
            raise ValueError("P1546.bt_loss error: Wrong value in the variable " "path" ". ")

    # Print the data in the log file if debug option is set

    floatformat = "%g,\n"

    if debug == 1:
        fid_log.write("# Input Parameters,Ref,,Value,\n")
        fid_log.write("Tx Power (kW),,," + floatformat % (PTx))
        fid_log.write("Frequency f (MHz),,," + floatformat % (f))
        fid_log.write("Horizontal path length d (km),,," + floatformat % (dtot))
        fid_log.write("Land path (km),,," + floatformat % (np.sum(dl)))
        fid_log.write("See path (km),,," + floatformat % (np.sum(ds)))
        fid_log.write("Percentage time t (%),,," + floatformat % (t))
        fid_log.write("Percentage location q (%),,," + floatformat % (q))
        fid_log.write("Tx antenna height h1 (m),S3 (4)-(7),," + floatformat % (h1))
        if isempty(ha):
            fid_log.write("Tx antenna height a. g. ha (m),,,\n")
        else:
            fid_log.write("Tx antenna height a. g. ha (m),,," + floatformat % (ha))
        if isempty(hb):
            fid_log.write("Tx antenna height hb (m),S3 (6),,\n")
        else:
            fid_log.write("Tx antenna height hb (m),S3 (6),," + floatformat % (hb))
        fid_log.write("Rx antenna height a. g. h2 (m),,," + floatformat % (h2))
        fid_log.write("Tx clutter height R1 (m),,," + floatformat % (R1))
        fid_log.write("Rx clutter height R2 (m),,," + floatformat % (R2))
        fid_log.write("Rx clutter type, , ,%s,\n" % (area))
        if isempty(wa):
            fid_log.write("Square area width wa for variability (m),S12 (34),,,\n")
        else:
            fid_log.write("Square area width wa for variability (m),S12 (34),," + floatformat % (wa))
        if isempty(pathinfo):
            fid_log.write("Terrain profile information available, ,,,\n")
        else:
            fid_log.write("Terrain profile information available, ,," + floatformat % (pathinfo))

        if isempty(eff1):
            fid_log.write("Tx effective TCA  theta_eff1 (deg),S4.3a),,\n")
        else:
            fid_log.write("Tx effective TCA  theta_eff1 (deg),S4.3a),," + floatformat % (eff1))
        if isempty(tca):
            fid_log.write("Terrain clearance angle tca (deg),S11 (31),,\n")
        else:
            fid_log.write("Terrain clearance angle tca (deg),S11 (31),," + floatformat % (tca))
        fid_log.write("\n# Computed Parameters,Ref,Step,Value,\n")

    # Compute the maximum value of the field strength as given in Annex 5, Sec. 2 for the case of mixed path

    EmaxF = step_19a(t, sum(dl), sum(ds))

    # In case, the slope path correction is necessary for the calculation of Emax
    # Step 16: Apply the slope-path correction given in annex 5, Sec. 14

    if (not isempty(ha)) and (not isempty(h2)):
        # print ('16: Slope-path correction.')
        if isempty(htter) and isempty(hrter):
            EmaxF = EmaxF + step_16a(ha, h2, d)

        else:
            EmaxF = EmaxF + step_16a(ha, h2, d, htter, hrter)

    if debug == 1:
        fid_log.write("Maximum field strength Emax (dBuV/m),S2 (1),," + floatformat % (EmaxF))

    for ii in range(0, NN):
        path = path_c[ii]
        if debug == 1:
            print("5: Following Steps 6 --> 10 for propagation path " + str(ii + 1) + " of type: " + path)

        # Step 5: For each propagation type follow Steps 6 to 10.

        if path.find("Warm") != -1 or path.find("Cold") != -1 or path.find("Sea") != -1:
            generalPath = "Sea"
            (idx, idxn) = np.where((figure_rec[:, 1] > 1))
            fig = figure_rec[idx, :]
        else:
            generalPath = "Land"
            (idx, idxn) = np.where((figure_rec[:, 1] == 1))
            fig = figure_rec[idx, :]

        Epath = 0.0

        if d >= 1:
            Epath = step6_10(fig, h1, dinf, dsup, d, path, f, EmaxF, t)
        else:
            Epath = step6_10(fig, h1, dinf, dsup, 1.0, path, f, EmaxF, t)

        if path.find("Land") != -1:
            El.append(Epath)

        elif path.find("Sea") != -1:
            Es.append(Epath)

        elif path.find("Cold") != -1:
            Es.append(Epath)

        elif path.find("Warm") != -1:
            Es.append(Epath)

        else:
            raise ValueError('P1546.bt_loss: Wrong value in the variable "path".')

    # Step 11: If the prediction is for a mixed path, follow the step-by-step
    # procedure given in Annex 5, Par 8. This requires use of Steps 6 to 10 for
    # paths of each propagation type. Note that if different sections of the
    # path exist classified as both cold and warm sea, all sea sections should
    # be classified as warm sea.

    if debug == 1:
        print("11: Combining field strengths for mixed paths if any")

    E = step_11a_rrc06(El, Es, dl, ds)
    if debug == 1:
        fid_log.write("Field strength (dBuV/m),S8 (17),11," + floatformat % (E))

    # Step 12: If information on the terrain clearance angle at a
    # receiving/mobile antenna adjacent to land is available, correct the field
    # strength for terrain clearance angle at the receiver/mobile using the
    # method given in Annex 5, Sec. 11.
    if not isempty(tca):
        if debug == 1:
            print("12: Terrain clearence angle correction")
        Correction, nu = step_12a(f, tca)
        E = E + Correction
        if debug == 1:
            fid_log.write("TCA nu,S11 (32c),12," + floatformat % (nu))
            fid_log.write("TCA correction (dB),S11 (32a),12," + floatformat % (Correction))

    # Step 13: Calculate the estimated field strength due to tropospheric
    # scattering using the method given in Annex 5, Sec. 13 and take the
    # maximum of E and Ets.
    if (not isempty(eff1)) and (not isempty(eff2)):
        if debug == 1:
            print("13: Calculating correction due to trophospheric scattering")
        Ets = 0.0
        theta_s = 0.0
        if d >= 1.0:
            Ets, theta_s = step_13a(d, f, t, eff1, eff2)
        else:
            Ets, theta_s = step_13a(1.0, f, t, eff1, eff2)
        E = max(E, Ets)
        if debug == 1:
            fid_log.write("Path scattering theta_s (deg),S13 (35),13," + floatformat % (theta_s))
            fid_log.write("Trop. Scatt. field strength Ets (dBuV/m),S13 (36),13," + floatformat % (Ets))

    # Step 14: Correct the field strength for receiving/mobile antenna height
    # h2 using themethod given in Annex 5, Sec. 9
    # CHECK: if R2 corresponds to the clutter or something else?
    if isempty(R2) or isempty(h2) or isempty(area):
        print("warning: R2, h2, and area are not defined. The following default values used:")
        print("Rx in Rural area: R2 = 10 m, h2 = R2")
        R2 = 10.0
        h2 = R2
        area = "Rural"

    if debug == 1:
        print("14: Receiving/mobile antenna height correction.")
    path = path_c[-1]

    Correction = 0.0
    R2p = 0.0
    if d >= 1.0:
        Correction, R2p = step_14a(h1, d, R2, h2, f, area)
    else:
        Correction, R2p = step_14a(h1, d, R2, h2, f, area)

    E = E + Correction

    if debug == 1:
        fid_log.write("Rx repr. clutter height R2" " (m),S9 (27),14," + floatformat % (R2p))
        fid_log.write("Rx antenna height correction (dB),S9 (28-29),14," + floatformat % (Correction))

    # Step 15: If there is clutter around the transmitting/base terminal, even
    # if at lower height above ground than the antenna, correct for its effect
    # using the  method given in Annex 5, Sec. 10

    if (not isempty(ha)) and (not isempty(R1)):
        if debug == 1:
            print("15: Correct for the transmitter/base clutter.")
        E = E + step_15a(ha, R1, f)
        if debug == 1:
            fid_log.write("Tx clutter correction (dB),S10 (30),15," + floatformat % (step_15a(ha, R1, f)))

    # Step 16: Apply the slope-path correction given in annex 5, Sec. 14
    if (not isempty(ha)) and (not isempty(h2)):
        if debug == 1:
            print("16: Slope-path correction.")
        if isempty(htter) and isempty(hrter):
            if d >= 1.0:
                Correction = step_16a(ha, h2, d)
            else:
                Correction = step_16a(ha, h2, 1.0)

            E = E + Correction

            if debug == 1:
                fid_log.write("Rx slope-path correction (dB),S14 (37),16," + floatformat % (Correction))

        else:
            if d >= 1.0:
                Correction = step_16a(ha, h2, d, htter, hrter)

            else:
                Correction = step_16a(ha, h2, 1.0, htter, hrter)

            E = E + Correction

            if debug == 1:
                fid_log.write("Rx slope-path correction (dB),S14 (37),16," + floatformat % (Correction))

    # Step 17: % In case the path is less than 1 km

    if dtot < 1:
        if debug == 1:
            print("17: Extrapolating the field strength for paths less than 1 km")
        if isempty(htter) and isempty(hrter):
            E = step_17a(ha, h2, d, E)
        else:
            E = step_17a(ha, h2, d, E, htter, hrter)

    if debug == 1:
        if dtot < 1:
            Edebug = E
            fid_log.write("Field strength for d < 1 km (dB),S15 (38),17," + floatformat % (Edebug))
        else:
            Edebug = []
            fid_log.write("Field strength for d < 1 km (dB),S15 (38),17,,\n")

    # Step 18: Correct the field strength for the required percentage of
    # locations using the method given in Annex 5, Sec. 12.
    Edebug = []
    if True:
        if abs(q - 50.0) > 0:
            if debug == 1:
                print("18: Correct for percentage locations different from 50 %.")
            # print q, E
            E = step_18a(E, q, f, pathinfo, wa, area)
            Edebug = E
            # print q, E

    if debug == 1:
        if isempty(Edebug):
            fid_log.write("Field strength for q <> 50 %,S12 (33),18,,\n")
        else:
            fid_log.write("Field strength for q <> 50 %,S12 (33),18," + floatformat % (Edebug))

    # Step 19: If necessary, limit the resulting field strength to the maximum
    # given in Annex 5, Sec. 2. If a mixed path calculation has been made for a
    # percentage time less than 50% use the method given by (42) in Draft
    # Revision 2013

    # EmaxF = Step_19a(t, sum(dl), sum(ds)),
    if E > EmaxF:
        if debug == 1:
            print("19: Limitting the maximum value of the field strength.")
        E = EmaxF

    if debug == 1:
        fid_log.write("Resulting field strength for Ptx = 1kW (dBuV/m), , ," + floatformat % (E))

    # Step 20: If required, convert field strength to eqivalent basic
    # transmission loss for the path using the method given in Annex 5, Sec 17
    # for 1 kW

    L = step_20a(f, E)

    # Scale to the transmitter power

    E = E + 10.0 * np.log10(PTx)

    if debug == 1:
        fid_log.write("Resulting field strength for given PTx (dBuV/m), , ," + floatformat % (E))
        fid_log.write("Resulting basic transmission loss (dB), S17 (40),20," + floatformat % (L))
        if inside_file == 1:
            try:
                fid_log.close()
            except (RuntimeError, TypeError, NameError):
                pass

    return E, L


def is_out_of_bounds(var, low, hi, name):
    """
    Function is_out_of_bounds(var, low, hi, name)
    Checks the var to see if it's inbetween the range low <= var <= hi
    returns true if it's out of bounds false if not
    """
    if (var < low) or (var > hi):
        print(name + " = " + str(var) + " is out of bounds.")
        return True
    else:
        return False


def h1_calc(d, heff, ha, hb, path, flag):
    """
    3 Determination of transmitting/base antenna height, h1
    Input Variables

    d     -   path length (km)
    heff  -   effective height of the transmitting/base antenna, defined as
            its height over the average level of the ground between
            distances of 3 km and 15 km from the transmitting/base antenna
            in the direction of the receiving/mobile antenna (m)
    ha    -   transmitting/base antenna height above ground (height of the
            mast) used when terrain information is not available (m)
    hb    -   transmitting/base antenna heightabove terrain height averaged
            between 0.2d and d (used when terrain information is available
            (m)
    path  -   type of the path ('Land' or 'Sea')
    flag  -   = 1(terrain information available), 0 (not available)

    Usage:

        h1eff = h1_calc(d, heff, ha, hb, path, flag)
        h1eff = h1_calc(d, heff, [], [], path, flag)

    The transmitting/base antenna height, h1, to be used in calculation
    depends on the type and length of the path and on various items of height
    information, which may not all be available.
    For sea paths h1 is the height of the antenna above sea level.
    For land paths, the effective height of the transmitting/base antenna,
    heff, is defined as its height in metres over the average level of the
    ground between distances of 3 and 15 km from the transmitting/base
    antenna in the direction of the receiving/mobile antenna. Where the value
    of effective transmitting/base antenna height, heff, is not known it
    should be estimated from general geographic information. This
    Recommendation is not valid when the transmitting/base antenna is below
    the height of surrounding clutter.
    The value of h1 to be used in calculation should be obtained using the
    method given in Par. 3.1, 3.2 or in Par. 3.3 as appropriate.

        3.1 Land paths shorter than 15 km
        For land paths less than 15 km one of the following two methods
        should be used:

            3.1.1 Terrain information not available
            Where no terrain information is available when propagation
            predictions are being made, the value of h1 is calculated
            according to path length, d, as follows:
            h1 =ha m for d <= 3 km (4)
            h1 = ha +(heff -ha )(d -3)/12 m for 3 km < d < 15 km (5)
            where ha is the antenna height above ground (e.g. height of the mast).

            3.1.2 Terrain information available
            Where terrain information is available when propagation
            predictions are being made:
            h1 =hb m (6)
            where hb is the height of the antenna above terrain height
            averaged between 0.2d and d km.

        3.2 Land paths of 15 km or longer
        For these paths:
        h1 =heff m (7)

        3.3 Sea paths
        The concept of h1 for an all-sea path is that it represents the
        physical height of the antenna above the surface of the sea. This
        Recommendation is not reliable in the case of a sea path for h1
        values less than about 3 m, and an absolute lower limit of 1 m should
        be observed.

    This function will return a NaN value if missing inputs according to ITU-R
    p.1546-5
    Rev   Date        Author                          Description
    -------------------------------------------------------------------------------
    v4    19MAY16     Ivica Stevanovic, OFCOM         Modified to distinguish between
                                                        terrain information (not) available
    v3    04DEC13     Ivica Stevanovic, OFCOM         Initial python version
    v2    08AUG13     Ivica Stevanovic, OFCOM         Edited comments and checked
                                                        the code
    v1    13AUG09     Jef Satham, Industry Canada      Initial version
    """

    if path.lower() == "land":
        if d < 15:
            if flag == 0:  # terrain info not available
                if d <= 3:
                    if not isempty(ha):
                        h1 = ha  # eq'n (4)
                        return h1
                    else:
                        print("h1_calc warning: d <= 3 km. No value for ha. Setting h1 = heff.")
                        h1 = heff
                        return h1
                else:  # 3 < d < 15
                    if not isempty(ha):
                        h1 = ha + (heff - ha) * (d - 3.0) / 12.0  # equ'n (5)
                        return h1
                    else:
                        print("h1_calc warning: 3 < d < 15. No value for ha or hb. Setting h1 = heff.")
                        h1 = heff
                        return h1

            else:  # terrain info available
                if not isempty(hb):
                    h1 = hb  # equ'n (6)
                    return h1
                else:
                    print("h1_calc warning: d < 15, terrain info available, No value for hb. Setting h1 = heff.")
                    h1 = heff
                    return h1

        else:  # d> 15 (Section 3.2)
            h1 = heff  # equ'n (7)
            return h1

    if path.lower() == "sea":
        if heff < 3:
            print("h1_calc warning: heff is too low for sea paths. Setting h1 = 3 m.")
            h1 = 3
            return

        h1 = heff
        return h1

    h1 = float("nan")
    return h1


def find_d_nominals(d):
    distance = [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        25,
        30,
        35,
        40,
        45,
        50,
        55,
        60,
        65,
        70,
        75,
        80,
        85,
        90,
        95,
        100,
        110,
        120,
        130,
        140,
        150,
        160,
        170,
        180,
        190,
        200,
        225,
        250,
        275,
        300,
        325,
        350,
        375,
        400,
        425,
        450,
        475,
        500,
        525,
        550,
        575,
        600,
        625,
        650,
        675,
        700,
        725,
        750,
        775,
        800,
        825,
        850,
        875,
        900,
        925,
        950,
        975,
        1000,
    ]

    # if value is not found dsup = 'nothing'
    dinf, dsup = search_closest(distance, d)

    return dinf, dsup


def search_closest(x, v):
    """
    The following code tidbit is by Dr. Murtaza Khan, modified to return
    vector y instead of index i if no exact value found. Also added
    functionality for when v is outside of vector x min and max.
    23 May 2007 (Updated 09 Jul 2009)
    Obtained from http://www.mathworks.com/matlabcentral/fileexchange/15088

    Algorithm
    First binary search is used to find v in x. If not found
    then range obtained by binary search is searched linearly
    to find the closest value.

    INPUT:
    x: vector of numeric values,
    x should already be sorted in ascending order
        (e.g. 2,7,20,...120)
    v: numeric value to be search in x

    OUTPUT:
    i: lower or equal value to v in x
    cv: value that is equal or higher to v in x

    Rev   Date        Author                          Description
    -------------------------------------------------------------------------------
    v3    04DEC13     Ivica Stevanovic, OFCOM          Initial python version
    v1    09JUL09     Jef Satham, Industry Canada      Initial searclosest version
    v0                Murzaza Khan, drkhanmurtaza@gmail.com   Initial tidbit version

    """

    if x[-1] < v:
        i = [x[-2], x[-1]]
        cv = i[1]
        i = i[0]
        return i, cv
    elif x[0] > v:
        i = [x[0], x[1]]
        cv = i[1]
        i = i[0]
        return i, cv

    fr = 0
    to = len(x) - 1

    ##Phase 1: Binary Search
    while fr <= to:
        mid = int(round((fr + to) / 2.0))

        diff = x[mid] - v
        if diff == 0:
            i = v
            cv = v
            return i, cv
        elif diff < 0:  # % x(mid) < v
            fr = mid + 1
        else:  # x(mid) > v
            to = mid - 1

    i = [x[to], x[fr]]
    cv = i[1]
    i = i[0]
    return i, cv


def step_19a(t, dland, dsea):
    """
    E = step_19a(t, dtotal, dsea)

    Step 19: If necessary, limit the resulting field strength to the maximum
    given in Annex 5, Paragraph 2. If a mixed path calculation has been made
    for a percentage time less than 50% it will be necessary to calculate the
    maximum field strength by linear interpolation between the all-land and
    all-sea values.
    Input variables are
    t       - percentage time (%)
    dland   - total distance of land paths (km)
    dsea    - total distance of sea paths (km)

    Rev   Date        Author                          Description
    --------------------------------------------------------------------------
    v2    04Dec13     Ivica Stevanovic, OFCOM         Initial python version
    v1    12AUG13     Ivica Stevanovic, OFCOM         Initial version
    """

    if t < 1 or t > 50:
        raise ValueError("step_19a: The percentage time out of band [1%, 50%].")

    dtotal = dland + dsea

    Efs = 106.9 - 20 * np.log10(1.0 * dtotal)  # (2)
    Ese = 2.38 * (1.0 - np.exp(-dtotal / 8.94)) * np.log10(50.0 / t)  # (3)

    # Linearly interpolate between all-land and all-sea values:

    Emax = Efs + dsea * Ese / dtotal

    return Emax


def step6_10(fig, h1, dinf, dsup, d, path, f, Emaxvalue, t):
    """
    E = step6_10(fig, h1, dinf, dsup, d, path, f, Emaxvalue, t)

    where:
    fig: the table of figures with the 3 parameter, time, frequency,
    path and figure number
    h1: the calculated height of the reciving antenna
    dsup: the higher measured distance value about d
    dinf: the lower measured distance value about d
    path: 'Warm' 'Cold' 'Sea' 'Land'
    f: frequency
    Emaxvalue: maximum calculated field strength
    t: percent time

    7 Interpolation of field strength as a function of percentage time
    Field-strength values for a given percentage of time between 1% and 50% time should be calculated
    by interpolation between the nominal values 1% and 10% or between the nominal values 10% and
    50% of time using:
    E = Esup (Qinf - Qt)/(Qinf - Qsup )+Einf (Qt - Qsup )/(Qinf - Qsup) dB(uV/m) (16)
    where:
    t : percentage time for which the prediction is required
    tinf : lower nominal percentage time
    tsup : upper nominal percentage time
    Qt = Qi (t/100)
    Qinf = Qi (tinf /100)
    Qsup = Qi (tsup /100)
    Einf : field-strength value for time percentage tinf
    Esup : field-strength value for time percentage tsup
    where Qi (x) is the inverse complementary cumulative normal distribution function.
    This Recommendation is valid for field strengths exceeded for percentage times in the range 1% to
    50% only. Extrapolation outside the range 1% to 50% time is not valid.
    A method for the calculation of Qi (x) is given in Annex 5, Par 15.
    """
    percentage = [1, 10, 50]

    tinf, tsup = search_closest(percentage, t)

    # Step 6: For the lower nominal percentage time follow Steps 7 to 10.
    Ep = np.zeros(2)
    argl = [tinf, tsup]
    if tinf == tsup:
        st = 1
    else:
        st = 0

    for l in range(st, 2):  # from st to 1
        # slight modification due to figure limits. When any sea path is
        # selected all figure tables for warm and cold are kept in the figure
        # matrix. This modification sorts out when a sea path Warm figure is
        # needed otherwise it will default to a Cold sea path.
        # ex. [1 2 600 x] does not exist nor do any t = 50 curves for warm or
        # cold sea specific tables.

        if ((argl[l] == 10) or (argl[l] == 1)) and path.find("Warm") != -1:
            # idx1 = ml.find(fig[:, 0] == argl[l])
            (idx1, idxn) = np.where((fig[:, 0] == argl[l]))
            # idx2 = ml.find(fig[:, 1] == 4)
            (idx2, idxn) = np.where((fig[:, 1] == 4))
            idx = np.intersect1d(idx1, idx2)
        else:
            # idx = ml.find(fig[:,0] == argl[l])
            (idx, idxn) = np.where((fig[:, 0] == argl[l]))

        figureStep6 = fig[idx, :]

        # Step 7-9: For the lower nominal frequency follow Steps 8 and 9.
        df = d06(f, h1, 10)
        d600 = d06(600, h1, 10)
        if path.find("Warm") != -1 or path.find("Cold") != -1 or path.find("Sea") != -1:
            generalPath = "Sea"
        else:
            generalPath = "Land"

        if generalPath.find("Sea") != -1 and (f < 100) and (d < d600):
            if d <= df:
                Ep[l] = Emaxvalue  # equ'n (15a)
            else:
                Edf = 0
                Edf = emax(df, t, generalPath)
                d600inf, d600sup = find_d_nominals(d600)
                Ed600 = step7_normal(figureStep6, h1, d600inf, d600sup, d600, generalPath, f, Emaxvalue, t)
                Ep[l] = Edf + (Ed600 - Edf) * mt.log10(1.0 * d / df) / mt.log10(1.0 * d600 / df)  # equ'n (15b)

        else:
            Ep[l] = Emaxvalue
            Ep[l] = step7_normal(figureStep6, h1, dinf, dsup, d, generalPath, f, Emaxvalue, t)

        # Step 10: If the required percentage time does not coincide with the
        # lower nominal percentage time, repeat Steps 7 to 9 for the higher
        # nominal percentage time and interpolate the two field strengths using
        # the method given in Annex 5, Par 7.

    if tinf != tsup:
        Qsup = qi(tsup / 100.0)
        Qinf = qi(tinf / 100.0)
        Qt = qi(t / 100.0)
        E = Ep[1] * (Qinf - Qt) / (Qinf - Qsup) + Ep[0] * (Qt - Qsup) / (Qinf - Qsup)  # equ'n (16)
        return E
    else:
        E = Ep[l]
        return E
    # For sea paths where the required frequency is less than 100 MHz an alternative method should be
    # used, based upon the path lengths at which 0.6 of the first Fresnel zone is just clear of obstruction
    # by the sea surface. An approximate method for calculating this distance
    # is given in Par 17.
    # The alternative method should be used if all of the following conditions are true:
    # - The path is a sea path.
    # - The required frequency is less than 100 MHz.
    # - The required distance is less than the distance at which a sea path would have 0.6 Fresnel
    # clearance at 600 MHz, given by D06(600, h1, 10) as given in Par 17.
    # If any of the above conditions is not true, then the normal interpolation/extrapolation method given
    # by equation (14) should be used.
    # If all of the above conditions are true, the required field strength, E, should be calculated using:
    # E = Emax dB(?V/m) for d ? df (15a)
    # Edf (Ed600 Ed )log(d /d f )/log(d600 /d f ) f = + ? dB(?V/m) for d > df (15b)
    # where:
    # Emax : maximum field strength at the required distance as defined in Par 2
    #: Edf maximum field strength at distance df as defined in Par 2
    # d600 : distance at which the path has 0.6 Fresnel clearance at 600 MHz calculated as
    # D06(600, h1, 10) as given in Par 17
    # df : distance at which the path has 0.6 Fresnel clearance at the required frequency
    # calculated as D06( f, h1, 10) as given in Par 17
    # : Ed600 field strength at distance d600 and the required frequency calculated using
    # equation (14).


def d06(f, h1, h2):
    """
    17 An approximation to the 0.6 Fresnel clearance path length
    The path length which just achieves a clearance of 0.6 of the first
    Fresnel zone over a smooth curved Earth, for a given frequency and
    antenna heights h1 and h2, is given approximately by (38)
    where:
    Df : frequency-dependent term
    = 0.0000389 f h1h2 km (39a)
    Dh : asymptotic term defined by horizon distances
    = 4.1( h1+ h2 ) km (39b)
    f : frequency (MHz)
    h1, h2 : antenna heights above smooth Earth (m).
    In the above equations, the value of h1 must be limited, if necessary,
    such that it is not less than zero. Moreover, the resulting values of D06
    must be limited, if necessary, such that it is not less than 0.001 km.
    """

    h1 = max(h1, 0.0)
    Df = 0.0000389 * f * h1 * h2  # equ'n (39a)
    Dh = 4.1 * (mt.sqrt(h1) + mt.sqrt(h2))  # equ'n (39b)
    D = Df * Dh / (Df + Dh + 0.0)  # equ'n (38)
    if D < 0.001:
        D = 0.001
    return D


def qi(x):
    """
    15 An approximation to the inverse complementary cumulative normal distribution
    function
    The following approximation to the inverse complementary cumulative normal distribution
    function, Qi (x), is valid for 0.01 <= x <= 0.99:
    Qi (x)=T(x)-out(x) if x <= 0.5 (36a)
    Qi (x)=-{T(1-x)-out(1-x)} if x > 0.5 (36b)
    where:
    T(x) = [-2ln(x)] (36c)
    out(x) = (((C2*T(z)+C1)*T(z))+C0)/(((D3*T(z)+D2)*T(z)+D1)*T(z)+1) (36d)
    C0 = 2.515517
    C1 = 0.802853
    C2 = 0.010328
    D1 = 1.432788
    D2 = 0.189269
    D3 = 0.001308
    Values given by the above equations are given in Table 3.
    """
    if x <= 0 or x >= 1:
        raise ValueError("qi: argument out of range")

    if x <= 0.5:
        out = T(x) - C(x)  # (36a)
    else:
        out = -(T(1 - x) - C(1 - x))  # (36b)
    return out


def T(y):
    outT = mt.sqrt(-2.0 * mt.log(y))  # (36c)
    return outT


def C(z):
    C0 = 2.515517
    C1 = 0.802853
    C2 = 0.010328
    D1 = 1.432788
    D2 = 0.189269
    D3 = 0.001308
    outC = (((C2 * T(z) + C1) * T(z)) + C0) / (((D3 * T(z) + D2) * T(z) + D1) * T(z) + 1)  # %(36d)
    return outC


def emax(d, t, path):
    """
    Maximum field-strength values
    Input Variables
    d     -   path length (km)
    t     -   percentage time (%)
    path  -   type of the path ('Land' or 'Sea')

    A field strength must not exceed a maximum value, Emax, given by:
        Emax = Efs dB(uV/m) for land paths (1a)
        Emax = Efs + Ese dB(uV/m) for sea paths (1b)
    where Efs is the free space field strength for 1 kW e.r.p. given by:
        Efs =106.9-20log(d) dB(uV/m) (2)
    and Ese is an enhancement for sea curves given by:
        Ese =2.38{1-exp(-d /8.94)}log(50/t) dB (3)

    In principle any correction which increases a field strength must not be
    allowed to produce values greater than these limits for the family of
    curves and distance concerned. However, limitation to maximum values
    should be applied only where indicated in Annex 6.

    Function will return NaN if missing path value.

    Rev   Date        Author                          Description
    -------------------------------------------------------------------------------
    v2    08JUN13     Ivica Stevanovic, OFCOM         Edited comments and checked the code
    v1    13AUG09     Jef Satham, Industry Canada     Initial version
    """

    Efs = 106.9 - 20 * np.log10(d)
    if path.find("Land") != -1:
        E = Efs
        return E

    if path.find("Sea") != -1:
        E = Efs + 2.38 * (1.0 - np.exp(-d / 8.94)) * np.log10(50.0 / t)
        return E

    E = float("nan")
    return E


def step7_normal(figureStep6, h1, dinf, dsup, d, generalPath, f, Emaxvalue, t):
    """
    step7_normal(figureStep6,h1,dinf,dsup,d,generalPath,f,Emax,t)

    where:

    figureStep6: the figure matrix with the correct time curve figures
        selected.
    h1: is the caculated h1 from h1Calc
        dinf and dsup must correspond to values in Table 1 of ITU-r p.1546 on
        page 38 otherwise NaN will be returned
    d: is the distance
    generalPath: either 'Land' or 'Sea'
    finf and fsup: the high low pairs of the f frequency value either 100,
        600, 2000
    f: frequency
    Emax: the the max field strength
    t: percent time

    returns a field strength otherwise NaN.

    Step 7: For the lower nominal frequency follow Steps 8 and 9.
    comment out # if direct excel access is preffered for better
    accuracy. Not as effecient for repeated use.
    """

    #exceltables = Exceltables()
    frequencies = [100, 600, 2000]
    finf, fsup = search_closest(frequencies, f)

    # print finf, f, fsup
    Ef = np.zeros(2)
    argj = [finf, fsup]
    if finf == fsup:
        st = 1
    else:
        st = 0
    for j in range(st, 2):
        # idx = ml.find(figureStep6[:, 2] == argj[j])

        (idx, idxn) = np.where((figureStep6[:, 2] == argj[j]))

        figureStep7 = figureStep6[idx, :]

        # Following 2 lines can be swapped interchangebly line 62 is best used
        # for hundredes of successful excel look ups.
        # tabulatedValues = xlsread('Rec_P_1546_2_Tab_values.xls',figureStep7(1,4), 'B6:K84')

        tabulatedValues = np.mat(exceltables[figureStep7[0, 3] - 1])

        # Step 8: Obtain the field strength exceeded at 50% locations for a
        # receiving/mobile antenna at the height of representative clutter, R,
        # above ground for the required distance and transmitting/base antenna
        # height as follows:

        if h1 >= 10:
            Ef[j] = step81(tabulatedValues, h1, dinf, dsup, d, Emaxvalue)
            Ef[j] = min(Ef[j], Emaxvalue)
        else:
            Ef[j] = step82(tabulatedValues, h1, dinf, dsup, d, generalPath, argj[j], f, Emaxvalue, t)

    if finf != fsup:
        E = Ef[0] + (Ef[1] - Ef[0]) * mt.log10(1.0 * f / finf) / mt.log10(1.0 * fsup / finf)  # %eq'n (14)
        if f > 2000.0:  # % STI: TBC should be limited only for f>2000 MHz
            E = min(E, Emaxvalue)

        return E
    else:
        E = Ef[j]
        return E

    # 6 Interpolation and extrapolation of field strength as a function of frequency
    # Field-strength values for the required frequency should be obtained by interpolating between the
    # values for the nominal frequency values of 100, 600 and 2 000 MHz. In the case of frequencies
    # below 100 MHz or above 2 000 MHz, the interpolation must be replaced by an extrapolation from
    # the two nearer nominal frequency values. For most paths interpolation or extrapolation for log
    # (frequency) can be used, but for some sea paths when the required frequency is less than 100 MHz
    # it is necessary to use an alternative method.
    # For land paths, and for sea paths where the required frequency is greater than 100 MHz, the
    # required field strength, E, should be calculated using:
    # E = Einf +(Esup ?Einf )log( f / finf )/log( fsup / finf ) dB(?V/m) (14)
    # where:
    # f : frequency for which the prediction is required (MHz)
    # finf : lower nominal frequency (100 MHz if f < 600 MHz, 600 MHz otherwise)
    # fsup : higher nominal frequency (600 MHz if f < 600 MHz, 2 000 MHz otherwise)
    # Einf : field-strength value for finf
    # Esup : field-strength value for fsup.
    # The field strength resulting from extrapolation for frequency above 2 000 MHz should be limited if
    # necessary such that it does not exceed the maximum value given in Par 2.


def step81(tabulatedValues, h1, dinf, dsup, d, Emax):
    """
    Step 8: Obtain the field strength exceeded at 50% locations for a
    receiving/mobile antenna at the height of representative clutter, R,
    above ground for the required distance and transmitting/base antenna
    height as follows:
        Step 8.1: For a transmitting/base antenna height h1 equal to or
        greater than 10 m follow Steps 8.1.1 to 8.1.6:
            Step 8.1.1: Determine the lower and higher nominal h1 values
            using the method given in Annex 5, Par 4.1. If h1 coincides with
            one of the nominal values 10, 20, 37.5, 75, 150, 300, 600 or
            1 200 m, this should be regarded as the lower nominal value of h1
            and the interpolation process of Step 8.1.6 is not required.

            Step 8.1.2: For the lower nominal value of h1 follow Steps 8.1.3
            to 8.1.5.

            Step 8.1.3: For the lower nominal value of distance follow 8.1.4.

            Step 8.1.4:
            Step 8.1.5:

            Step 8.1.6: If the required transmitting/base antenna height, h1,
            does not coincide with one of the nominal values, repeat Steps
            8.1.3 to 8.1.5 and interpolate/extrapolate for h1 using the
            method given in Annex 5, Par 4.1. If necessary limit the result to
            the maximum given in Annex 5, Par 2.
    E = step81(tabulatedValues,h1,dinf,dsup,d,Emax)

    tabulatedValues a matrix of value from a figure 1-24 of excel sheet of
        values for recommendation. Expected range from table 'B6:K84' may still
        work with different range value. results unexpected.
    h1 is the caculated h1 from h1Calc
    dinf and dsup must correspond to values in Table 1 of ITU-r p.1546 on
        page 38 otherwise NaN will be returned
    d is the distance
    Emax is the max field strength if necessary to limit the result.

    returns field strength otherwise NaN.
    """

    # double check h1
    if not (h1 >= 10):
        raise ValueError("h1 is less then 10 for step81")

    # obtain hsup and hinf
    height = [10, 20, 37.5, 75, 150, 300, 600, 1200]

    hinf, hsup = search_closest(height, h1)

    #    if (h1<hinf or h1>hsup):
    #        print hinf,h1,hsup
    #        return

    Eh1 = np.zeros(2)
    arg = [hinf, hsup]
    if hinf == hsup:
        st = 1
    else:
        st = 0

    for x in range(st, 2):
        Eh1[x] = step814_815(tabulatedValues, arg[x], dinf, dsup, d)

    if hinf != hsup:
        Eh1 = Eh1[0] + (Eh1[1] - Eh1[0]) * mt.log10(1.0 * h1 / hinf) / mt.log10(1.0 * hsup / hinf)  # equ'n (8)
    else:
        Eh1 = Eh1[1]

    if Eh1 > Emax:
        E = Emax
        return E
    else:
        E = Eh1
        return E

    E = float("nan")
    return E


#     4 Application of transmitting/base antenna height, h1
#     The value of h1 controls which curve or curves are selected from which to
#     obtain field-strength values, and the interpolation or extrapolation
#     which may be necessary. The following cases are distinguished.
#
#         4.1 Transmitting/base antenna height, h1, in the range 10 m to 3 000m
#         If the value of h1 coincides with one of the eight heights for which
#         curves are provided, namely 10, 20, 37.5, 75, 150, 300, 600 or 1 200m
#         the required field strength may be obtained directly from the plotted
#         curves or the associated tabulations. Otherwise the required field
#         strength should be interpolated or extrapolated from field strengths
#         obtained from two curves using:
#         E = Einf +(Esup ?Einf )log(h1 /hinf )/log(hsup /hinf ) dB(?V/m) (8)
#         where:
#             hinf : 600 m if h1 > 1 200 m, otherwise the nearest nominal effective
#             height below h1
#             hsup : 1 200 m if h1 > 1 200 m, otherwise the nearest nominal
#             effective height above h1
#             Einf : field-strength value for hinf at the required distance
#             Esup : field-strength value for hsup at the required distance.
#         The field strength resulting from extrapolation for h1 > 1 200 m
#         should be limited if necessary such that it does not exceed the
#         maximum defined in Par 2.
#         This Recommendation is not valid for h1 > 3 000 m.


def step814_815(tabulatedValues, h1, dinf, dsup, d):
    """
    Step 8.1.4: Obtain the field strength exceeded at 50% locations
    for a receiving/mobile antenna at the height of representative
    clutter, R, for the required values of distance, d, and
    transmitting/base antenna height, h1.

    Step 8.1.5: If the required distance does not coincide with the
    lower nominal distance, repeat Step 8.1.4 for the higher nominal
    distance and interpolate the two field strengths for distance
    using the method given in Annex 5, Par 5.

    function E = step814_815(tabulatedValues,h1,dinf,dsup)
    if only step 8.1.4 is needed pass the same value for dinf and dsup.

    tabulatedValues a matrix of value from a figure 1-24 of excel sheet of
        values for recommendation. Expected range from table 'B6:K84' may still
        work with different range value. results unexpected.
    h1 must equal one of the nominal value 10,20,37.5,75,150,300,600,1200 if
        not return is unknown.
    dinf and dsup must correspond to values in Table 1 of ITU-r p.1546 on
        page 38 otherwise NaN will be returned

    Returns a single field strength value for the given parameters
    """

    # kLookUp = ml.find(tabulatedValues[0,:] == h1)
    (idxn, kLookUp) = np.where((tabulatedValues[0, :] == h1))

    eLookUp = np.hstack((tabulatedValues[:, 0], tabulatedValues[:, kLookUp]))

    # kinf = ml.find(eLookUp[:, 0] == dinf)
    (kinf, idxn) = np.where((eLookUp[:, 0] == dinf))
    Einf = eLookUp[kinf, 1].item(0, 0)
    # ksup = ml.find(eLookUp[:, 0] == dsup)
    (ksup, idxn) = np.where((eLookUp[:, 0] == dsup))
    Esup = eLookUp[ksup, 1].item(0, 0)

    if dsup != dinf:
        # ksup = ml.find(eLookUp[:, 0] == dsup ) ## STI: this seams to be redundant ?
        (ksup, idxn) = np.where((eLookUp[:, 0] == dsup))
        Esup = eLookUp[ksup, 1].item(0, 0)

        E = Einf + (Esup - Einf) * np.log10(1.0 * d / dinf) / np.log10(1.0 * dsup / dinf)  #    %equ'n (13)

        return E
    else:
        E = Esup

        return E
    E = float("nan")
    return E


def step82(tabulatedValues, h1, dinf, dsup, d, path, fnom, f, Emaxvalue, t):
    """
    Step 8.2: For a transmitting/base antenna height h1 less than 10 m
    determine the field strength for the required height and distance
    using the method given in Annex 5, Par 4.2. If h1 is less than zero,
    the method given in Annex 5, Par 4.3 should also be used.

    function E = step82(tabulatedValues,h1,dinf,dsup,d,path,fnom,f,Emaxvalue,t)

    tabulatedValues a matrix of value from a figure 1-24 of excel sheet of
        values for recommendation. Expected range from table 'B6:K84' may still
        work with different range value. results unexpected.
    h1 is the caculated h1 from h1Calc
    dinf and dsup must correspond to values in Table 1 of ITU-r p.1546 on
        page 38 otherwise NaN will be returned
    d is the distance
    path either 'Land' or 'Sea'
    fnom the nominal frequency (100 600 or 1200 MHz)
    f frequency
    Emaxvalue the the max field strength
    t percent time

    returns a field strength otherwise NaN.

    Rev   Date        Author                          Description
    -------------------------------------------------------------------------------
    v3    05DEC13     Ivica Stevanovic, OFCOM         Initial python version
    v2    08JUN13     Ivica Stevanovic, OFCOM         Edited comments and checked
                                                        and corrected the code
                                                        for h1<0 and land paths
    v1    13AUG09     Jef Satham, Industry Canada     Initial version
    """

    if h1 >= 10:
        raise ValueError("incorrect h1 value for step82: Greater than 10 m.")

    # look up figure values for E10 and E20
    E10 = step814_815(tabulatedValues, 10.0, dinf, dsup, d)
    E20 = step814_815(tabulatedValues, 20.0, dinf, dsup, d)

    v = V(fnom, -10.0)

    if v > -0.7806:
        J = 6.9 + 20.0 * np.log10(np.sqrt((v - 0.1) ** 2 + 1) + v - 0.1)  #  % equ'n (12a)
    else:
        J = 0.0

    Ch1neg10 = 6.03 - J  # equ'n (12)
    C1020 = E10 - E20  # equ'n (9b)
    Ezero = E10 + 0.5 * (C1020 + Ch1neg10)  # equ'n (9a)

    if path.find("Land") != -1:
        if h1 >= 0:
            E = Ezero + 0.1 * h1 * (E10 - Ezero)  # equ'n (9)
            return E
        elif h1 < 0:  
            v = V(fnom, h1)
            if v > -0.7806:
                J = 6.9 + 20 * np.log10(np.sqrt((v - 0.1) ** 2 + 1) + v - 0.1)  # equ'n (12a)
            else:
                J = 0

            E = Ezero + 6.03 - J

            return E

    elif path.find("Sea") != -1:
        if h1 < 1:
            ValueError("h1 cannot be less than 1 m for calculating sea path")

        Dh1 = d06(f, h1, 10.0)  # equ'n (10a)
        D20 = d06(f, 20.0, 10.0)  # equ'n (10b)
        if d <= Dh1:
            E = Emaxvalue  # equ'n (11a)
            return E
        elif (d > Dh1) and (d < D20):
            dinf1, dsup1 = find_d_nominals(D20)
            E10D20 = step814_815(tabulatedValues, 10.0, dinf1, dsup1, D20)
            E20D20 = step814_815(tabulatedValues, 20.0, dinf1, dsup1, D20)
            ED20 = E10D20 + (E20D20 - E10D20) * np.log10(h1 / 10.0) / np.log10(20.0 / 10.0)
            # EDh1 = emax(Dh1, t, 'Sea')
            EDh1 = step_19a(t, 0, Dh1)
            # E = (ED20 - EDh1)*log10(d/Dh1)/log10(D20/Dh1)% equ'n (11b) <--
            # this equation has a typo and it has to be double checked... most
            # probably it should look like
            E = EDh1 + (ED20 - EDh1) * np.log10(1.0 * d / Dh1) / np.log10(1.0 * D20 / Dh1)
            return E
        elif d >= D20:
            E1 = E10 + (E20 - E10) * np.log10(h1 / 10.0) / np.log(20.0 / 10.0)
            v = V(fnom, -10.0)
            if v > -0.7806:
                J = 6.9 + 20 * np.log10(np.sqrt((v - 0.1) ** 2 + 1) + v - 0.1)  # equ'n (12a)
            else:
                J = 0.0

            Ch1neg10 = 6.03 - J  # equ'n (12)
            C1020 = E10 - E20  # equ'n (9b)
            Ezero = E10 + 0.5 * (C1020 + Ch1neg10)  # equ'n (9a)
            E2 = Ezero + 0.1 * h1 * (E10 - Ezero)  # equ'n (9)
            Fs = (d - D20) / d
            E = E1 * (1.0 - Fs) + E2 * Fs  # equ'n (11c)
            return E

    raise ValueError("no path selected in step82")

    # 4.2 Transmitting/base antenna height, h1, in the range 0 m to 10 m
    # The method when h1 is less than 10 m depends on whether the path is over
    # land or sea.
    # For a land path:
    # For a land path the field strength at the required distance d km
    # for 0 < h1 < 10 m is calculated using:
    # E = Ezero +0.1*h1(E10 -Ezero ) dB(?V/m) (9)
    # where:
    # Ezero = E10 +0.5(C1020 +Ch1neg10 ) dB(?V/m) (9a)
    # C1020 = E10 - E20 dB (9b)
    # Ch1neg10: the correction Ch1 in dB calculated using equation (12)
    # in Par 4.3 below at the required distance for h1 = -10 m
    # E10 and E20: the field strengths in dB(?V/m) calculated according
    # to Par 4.1 above at the required distance for h1 = 10m and h1 = 20m
    # respectively.
    # Note that the corrections C1020 and Ch1neg10 should both evaluate
    # to negative quantities.
    # For a sea path:
    # Note that for a sea path, h1 should not be less than 1 m. The
    # procedure requires the distance at which the path has 0.6 of the
    # first Fresnel zone just unobstructed by the sea surface. This is
    # given by:
    # Dh1 =D06 ( f , h1,10) km (10a)
    # where f is the nominal frequency (MHz) and the function D06 is
    # defined in Par 17.
    # If d > Dh1 it will be necessary to also calculate the 0.6 Fresnel
    # clearance distance for a sea path where the transmitting/base
    # antenna height is 20 m, given by:
    # D20 =D06 ( f , 20,10) km (10b)
    # The field strength for the required distance, d, and value of h1,
    # is then given by:
    # E = Emax dB(?V/m) for d ? Dh1 (11a)
    #  = EDh1 =(ED20 ?EDh1) log(d /Dh1)/log(D20 /Dh1) dB(?V/m)
    #                                          for Dh1 < d < D20 (11b)
    #  = E?(1?Fs )+E??Fs dB(?V/m) for d ? D20 (11c)
    # where:
    # Emax : maximum field strength at the required distance given in Par 2
    # EDh1 : Emax for distance Dh1 as given in Par 2
    # ED20 = E10(D20) + (E20(D20) ? E10(D20)) log (h1/10)/log (20/10)
    # E10(x) : field strength for h1 = 10 m interpolated for distance x
    # E' = E10(d) + (E20(d) ? E10(d)) log (h1/10)/log (20/10)
    # E??: field strength for distance d calculated using equation (9)
    # FS = (d ? D20)/d.
    #
    # 4.3 Negative values of transmitting/base antenna height, h1
    # For land paths it is possible for the effective transmitting/base antenna
    # height heff to have a negative value, since it is based on the average
    # terrain height at distances from 3 km to 15 km. Thus h1 may be negative.
    # In this case, the effect of diffraction by nearby terrain obstacles
    # should be taken into account.
    #   % The procedure for negative values of h1 is to obtain the field strength
    # for h1 = 0 as described in  Par 4.2, and to add a correction Ch1 calculated
    # as follows.
    # The effect of diffraction loss is taken into account by a correction,
    # Ch1, given by cases a) or b) as follows:
    #   % a) In the case that a terrain database is available and the potential for
    # discontinuities at the transition around h1 = 0 is of no concern in the
    # application of this Recommendation, the terrain clearance angle, ?eff1,
    # from the transmitting/base antenna should be calculated as the elevation
    # angle of a line which just clears all terrain obstructions up to 15 km
    # from the transmitting/base antenna in the direction of (but not going
    # beyond) the receiving/mobile antenna. This clearance angle, which will
    # have a positive value, should be used instead of ?tca in equation (30c)
    # in the terrain clearance angle correction method given in Par 11 to obtain
    # Ch1. Note that using this method can result in a discontinuity in field
    # strength at the transition around h1 = 0.
    #   %% This implementation is 4.3 a) IS NOT ACCOUNTED FOR.
    #   % b) In the case where a terrain database is not available or where a
    # terrain database is available, but the method must never produce a
    # discontinuity in the field strength at the transition around h1 = 0, the
    # (positive) effective terrain clearance angle, ?eff2, may be estimated
    # assuming an obstruction of height h1 at a distance of 9 km from the
    # transmitting/base antenna. Note that this is used for all path lengths,
    # even when less than 9 km. That is, the ground is regarded as
    # approximating an irregular wedge over the range 3 km to 15 km from the
    # transmitting/base antenna, with its mean value occurring at 9 km, as
    # indicated in Fig. 25. This method takes less explicit account of terrain
    # variations, but it also guarantees that there is no discontinuity in
    # field strength at the transition around h1 = 0. The correction to be
    # added to the field strength in this case is calculated using:
    # Ch1 =6.03? J (?) dB (12)
    # where:
    # J(?)= 6.9+20*log10(sqrt((v-0.1)^2+1)+v-0.1) (12a)
    # ? =K?*?eff2 (12b)
    # and
    # ?eff 2 =arctan(?h1/9000) degrees (12c)
    # K? = 1.35 for 100 MHz
    # K? = 3.31 for 600 MHz
    # K? = 6.00 for 2000 MHz
    #   % The above correction, which is always less than zero, is added to the
    # field strength obtained for h1 = 0.


def V(Kv, h1):
    """
    Calculates v for Annex 5 section 4.3 case b
    error if input for kv is not a nominal frequency.

    deg = V(Kv,h1)
    """
    if Kv == 100:
        deg = 1.35 * atand(-h1 / 9000.0)  # equ'n (12c and 12b)
        return deg
    elif Kv == 600:
        deg = 3.31 * atand(-h1 / 9000.0)  # equ'n (12c and 12b)
        return deg
    elif Kv == 2000:
        deg = 6.0 * atand(-h1 / 9000.0)  # equ'n (12c and 12b)
        return deg
    raise ValueError("Invalid frequency input in V(Kv, h1)")


def step_11a_rrc06(Eland, Esea, dland, dsea):
    """
    E = step_11a_rrc06(Eland, Esea, dtotal, dsea)
    Step 11: If there are two or more different propagation zones which
    involve at least one land/sea or land/costal land boundary, the following
    method approved by RRC-06 shall be used
    Input parameters
    Eland  - a vector of field strengths. ith element is a field strength for
            land path i equal in length to the mixed path (dB(uV/m))
    Esea   - a vector of field strengths. jth element is a field strength for
            sea-and-coastal-land path j equal in lenth to the mixed path
            (dB(uV/m))
    dland  - a vector of land-path lengths: ith element is the length
            of a path in land zone i (km)
    dsea   - a vector of sea-and-coastal-land path lengths: jth element is
            the length of a path in sea-and-coastal zone j (km).

    Rev   Date        Author                          Description
    --------------------------------------------------------------------------
    v2    05DEC13     Ivica Stevanovic, OFCOM         Initial version in python
    v1    13AUG13     Ivica Stevanovic, OFCOM         Initial version
    """
    # Verify that Eland and dland, Esea and dsea have the same length

    if len(Eland) != len(dland):
        ValueError("Vectors Eland and dland must be of the same length in step_11a_rrc06.")

    if len(Esea) != len(dsea):
        ValueError("Vectors Esea and dsea must be of the same length in step_11a_rrc06.")

    # Compute the mixed path interpolation factor

    dlT = sum(dland)
    dsT = sum(dsea)

    dtotal = dlT + dsT

    casea = False

    if dlT == 0:
        dd = np.array(dsea)
        EE = np.array(Esea)
        casea = True
    elif dsT == 0:
        dd = np.array(dland)
        EE = np.array(Eland)
        casea = True

    if casea:
        # In case there is no land/sea or land/coastal-land transitions (meaning
        # that either dlT=0 or dsT=0) the following procedure is used

        E = np.dot(dd, EE) / dtotal

    else:
        Fsea = 1.0 * dsT / dtotal

        Delta = 1.0 * np.dot(Esea, dsea) / dsT - 1.0 * np.dot(Eland, dland) / dlT

        V = max(1.0, 1.0 + Delta / 40.0)

        A0 = 1 - (1 - Fsea) ** (2.0 / 3.0)

        A = A0**V

        E = (1.0 - A) * np.dot(Eland, dland) / dlT + 1.0 * A * np.dot(Esea, dsea) / dsT

    return E


def step_12a(f, tca):
    """
    e=Step_12a(f, tca)
    Step 12: If information on the terrain clearance angle at a
    receiving/mobile antenna adjacent to land is available, correct the field
    strength for terrain clearance angle at the receiver/mobile using the
    method given in Annex 5, Par 11 of ITU-R P.1546-5.
    Input parameters
    f - frequency (MHz)
    tca - terrain clearance angle (deg) is the elevation angle of the line
    from the receiving/mobile antenna which just clears all terrain
    obstructions in the direction of the transmitter/base antenna over a
    distance up to 16 km but not going beyond the transmitting/base antenna
    The calculation of the tca angleshould no take Earth curvature into
    account. and should be limited such that it is not less than 0.55 deg or
    not more than +40 deg.

    Rev   Date        Author                          Description
    --------------------------------------------------------------------------
    v4    19MAY16     Ivica Stevanovic, OFCOM         introduced nu as output variable
    v3    05DEC13     Ivica Stevanovic, OFCOM         Initial version in python
    v2    18SEP13     Ivica Stevanovic, OFCOM         Re-written to limit between 0.55 and 40
    v1    12AUG13     Ivica Stevanovic, OFCOM         Initial version
    """

    e = 0.0

    if tca > 40:
        tca = 40.0

    if tca < 0.55:
        tca = 0.55

    nup = 0.036 * np.sqrt(f)
    nu = 0.065 * tca * np.sqrt(f)
    J1 = 0
    if nup > -0.7806:
        J1 = J(nup)
    J2 = 0
    if nu > -0.7806:
        J2 = J(nu)
    e = J1 - J2

    return e, nu


def J(nu):
    """
    This function computes the value of equation (12)
    according to Annex 5 Paragraph 4.3 of ITU-R P.1546-5

    Rev   Date        Author                          Description
    -------------------------------------------------------------------------------
    v2    05DEC13     Ivica Stevanovic, OFCOM         Initial version in python
    v1    12AUG13     Ivica Stevanovic, OFCOM         Initial version
    """
    return 6.9 + 20.0 * np.log10(np.sqrt((nu - 0.1) ** 2.0 + 1.0) + nu - 0.1)


def step_13a(d, f, t, eff1, eff2):
    """
    e = Step_13a(d,f,t,eff1,eff2)
    Step 13: Calculate the estimated field strength due to tropospheric
    scattering using the method given in Annex 5 Par 13 of ITU-R P.1546-5 and,
    if necessary, limit the final predicted field strength accordingly.
    Input variables
    d - path length (km)
    f - required frequency (MHz)
    t - required percentage of time (%)
    eff1 - the h1 terminal's terrain clearance angle in degrees calculated
        using the method in Paragraph 4.3 case a, whether or not h1 is
        negative (degrees)
    eff2 - the h2 terminal's clearance angel in degrees as calculated in
            Paragraph 11, noting that this is the elevation angle relative to
            the local horizontal (degrees)

    Rev   Date        Author                          Description
    ------------------------------------------------------------------------------
    v4    19MAY16     Ivica Stevanovic, OFCOM         Introduced thetaS as output variable
    v3    05DEC13     Ivica Stevanovic, OFCOM         Initial version in python
    v2    12AUG13     Ivica Stevanovic, OFCOM         Modified version
    v1    13AUG09     Jef Satham, Industry Canada     Initial version

    """
    thetaS = 180.0 * d / np.pi / 4.0 * 3.0 / 6370.0 + eff1 + eff2  # %equ'n (35)
    if thetaS < 0:
        thetaS = 0.0

    e = 24.4 - 20.0 * np.log10(1.0 * d) - 10.0 * thetaS - (5.0 * np.log10(f * 1.0) - 2.5 * (np.log10(1.0 * f) - 3.3) ** 2.0) + 0.15 * 325.0 + 10.1 * (-np.log10(0.02 * t)) ** 0.7  # equ'n (36),(36a),(36b)
    return e, thetaS


def step_14a(h1, d, R2, h2, f, area):
    """
    Correction = step_14a(h1, d, R, h2, f, area)
    This function computes correction for recieving/mobile antenna height
    according to Annex 5 Paragraph 9 of ITU-R P.1546-5
    Input variables are
    h1 - transmitting/base antenna height (m)
    d  - receiving/mobile antenna distance (km)
    R2  - height of the ground cover surrounding the receiving/mobile antenna,
        subject to a minimum height value of 10 m (m)
        Examples of reference heights are 20 m for an urban area, 30 m for
        a dense urban area and 10 m for a suburban area. For sea paths
        the notional value of R is 10 m.
    h2 - receiving/mobile antenna height (m)
    f  - frequency (MHz)
    area - 'Suburban' 'Urban', 'Dense Urban', 'Rural', 'Sea'

    This Recommendation (function) is not valid for receiving/mobile antenna
    height, h2, less than 1 m when adjacent to land, or less than 3 m when
    adjacent to sea

    Rev   Date        Author                          Description
    -------------------------------------------------------------------------------
    v5    19MAY16     Ivica Stevanovic, OFCOM         Made sure the value of R2 used in calculation is reported in the log file
    v4    05DEC13     Ivica Stevanovic, OFCOM         Initial version in python
    v3    18SEP13     Ivica Stevanovic, OFCOM         Modified theta_clut (radians instead of degrees),
                                                        for Rp<10 both equations 28a and 28b should be reduced
    v2    21AUG13     Ivica Stevanovic, OFCOM         introduced urban,
                                                        dense, rural, suburban, and sea
    v1    12AUG13     Ivica Stevanovic, OFCOM         Initial version
    """
    path = []
    Rp = []

    if area.lower().find("urban") != -1 or area.lower().find("dense urban") != -1 or area.lower().find("rural") != -1 or area.lower().find("suburban") != -1:
        path = "Land"
    elif area.lower().find("sea") != -1:
        path = "Sea"
    else:
        raise ValueError("Wrong area in step_14a")

    if isempty(h2):
        print("warning: h2 must be defined for step_14a to be applied")
        Correction = 0
        return Correction, Rp

    # This value will be used throughout the function:

    K_h2 = 3.2 + 6.2 * np.log10(1.0 * f)

    # R is subject to a minimum height value of 10 m  -- this does not seem to be true

    if path.find("Land") != -1:
        if h2 < 1:
            raise ValueError("This recommendation is not valid for receiving/mobile antenna height h2 < 1 m when adjacent to land.")

        # if the receiving/mobile antenna is on land account should first be
        # taken of the elevation angle of the arriving ray by calculating a
        # modified representative clutter height Rp (m) given by:

        Rp = (1000.0 * d * R2 - 15.0 * h1) / (1000.0 * d - 15.0)

        # the value of Rp must be limited if necessary such that it is not less
        # than 1 m

        if Rp < 1:
            Rp = 1
            print("warning: The value of modified representative clutter height is smaller than 1 m.")
            print("Setting the value to 1 m.")

        if area.find("Urban") != -1 or area.find("Dense Urban") != -1 or area.find("Suburban") != -1:
            # When the receiving/mobile antenna is in an urban environment the
            # corection is then given by:

            if h2 < Rp:
                h_dif = Rp - h2
                K_nu = 0.0108 * np.sqrt(f)

                theta_clut = atand(h_dif / 27.0)

                nu = K_nu * np.sqrt(h_dif * theta_clut)

                # J_nu = 6.9 + 20*log10(sqrt((nu-0.1).^2+1) + nu - 0.1)
                Correction = 6.03 - J(nu)  # (28a)

            else:
                Correction = K_h2 * np.log10(1.0 * h2 / Rp)  # (28b)

            if Rp < 10:
                # In cases of an urban environment where Rp is less than 10 m,
                # the correction given by equation (28a) or (28b) should be reduced by

                Correction = Correction - K_h2 * np.log10(10.0 / Rp)

        else:
            # When the receiving/mobile antenna is on land in a rural or open
            # environment, the correction is given by equation (28b) for all
            # values of h2 with Rp set to 10 m
            Rp = 10
            Correction = K_h2 * np.log10(h2 / 10.0)

    else:  # receiver adjacent to sea
        if h2 < 3:
            raise ValueError("This recommendation is not valid for receiving/mobile antenna height h2 < 3 m when adjacent to sea.")

        # In the following, the expression "adjacent to sea" applies to cases
        # where the receiving/mobile antenna is either over sea, or is
        # immediately adjacent to the sea, with no significant obstruction in
        # the direction of the transmitting/base station.

        if h2 >= 10:
            # Where the receiving/mobile antenna is adjacent to sea for h2>=10m,
            # the correction should be calculated using equation (28b) wih
            # Rp set to 10
            Rp = 10
            Correction = K_h2 * np.log10(h2 / 10.0)

        else:
            # Where the receiving/mobile antenna is adjacent to sea for h2<10m,
            # an alternative method should be used, based upon the path lengths
            # at which 0.6 of the first Fresnel zone is just clear of
            # obstruction by the sea surface. An approximate method for
            # calculating this distance is given in Paragraph 18

            # distance at which the path just has 0.6 Fresnel clearance for h2
            # = 10 m calculated as D06(f, h1, 10):

            d10 = d06(f, h1, 10.0)

            # Distance at which the path just has 0.6 Fresnel clearance for the
            # required value of h2 calculated as D06(f,h1,h2):
            dh2 = d06(f, h1, h2)

            # Correction for the required value of h2 at distance d10 using
            # equation (27b) with Rp set to 10m
            Rp = 10
            C10 = K_h2 * np.log10(h2 / 10.0)

            if d >= d10:
                # If the required distance is equal to or greater than d10, then
                # again the correction for the required value of h2 should be
                # calculated using equation (28b) with Rp set to 10
                Rp = 10
                Correction = C10

            else:
                # if the required distance is less than d10, then the
                # correction to be added to the field strength E should be
                # calculated using

                if d <= dh2:
                    Correction = 0.0

                else:  #  dh2 < d < d10
                    Correction = C10 * np.log10(1.0 * d / dh2) / np.log10(1.0 * d10 / dh2)
    return Correction, Rp


def atand(x):
    """
    This function computes the atan and returns the value in degrees
    """
    return mt.atan(x) * 180.0 / mt.pi


def step_15a(ha, R1, f):
    """
    Correction = Step_10(ha, R1, f)
    Step 15: If there is clutter around the transmitting/base terninal, even
    if at a lower height above ground than the antenna, correctg for its
    effect using method given in Annex 5, Par 10 of ITU-R P.1546-5.

    This correction applies when the transmitting/base terminal is over or
    adjacent to land on which there is clutter. The correction should be used
    in all such cases, including when the antenna is above the clutter
    height. The correction is zero wehn the therminal is higher than a
    frequency-dependent clearance height above the clutter.
    Input variables are
    ha - transmitting/base terminal antenna height above ground (m) (i.e., height of the mast)
    R  - representative of the height of the ground cover surrounding the
        transmitting/base antenna, subject to a minimum height value of 10 m (m)
        Examples of reference heights are 20 m for an urban area, 30 m for
        a dense urban area and 10 m for a suburban area. For sea paths
        the notional value of R is 10 m.
    f  - frequency (MHz)

    Rev   Date        Author                          Description
    -------------------------------------------------------------------------------
    v3    05DEC13     Ivica Stevanovic, OFCOM         Initial version in python
    v1    12AUG13     Ivica Stevanovic, OFCOM         Initial version
    """

    K_nu = 0.0108 * np.sqrt(f)
    hdif1 = ha - R1
    theta_clut = atand(hdif1 / 27.0)

    if R1 >= ha:
        nu = K_nu * np.sqrt(hdif1 * theta_clut)
    else:
        nu = -K_nu * np.sqrt(hdif1 * theta_clut)

    Correction = 0

    if nu > -0.7806:
        Correction = -J(nu)

    return Correction


def step_16a(ha, h2, d, *argc):
    """
    Where terrain information is available
    Correction = Step_16a(ha, h2, d, htter, hrter)
    Where terrain information is not available
    Correction = Step_16a(ha,h2,d)

    Step 16: A correction is required to take account of the difference in
    height between the two antennas
    Input variables are
    ha - transmitting/base terminal antenna height above ground (m)
    h2 - receiving/mobile terminal antenna height above ground (m)
    d  - horizontal path distance (km)
    htter - terrain height in meters above sea level at the transmitter/base
            terminal (m)
    hrter - terrain height in meters above sea level at the receiving/mobile
            terminal (m)
    Rev   Date        Author                          Description
    -------------------------------------------------------------------------------
    v2    05DEC13     Ivica Stevanovic, OFCOM         Initial version in python
    v1    12AUG13     Ivica Stevanovic, OFCOM         Initial version
    """

    d_slope = dslope(ha, h2, d, *argc)

    Correction = 20.0 * np.log10(1.0 * d / d_slope)

    return Correction


def dslope(ha, h2, d, *argc):
    """
    Where terrain information is available
    outVal = dslope(ha, h2, d, htter, hrter)
    Where terrain information is not available
    outVal = dslope(ha,h2,d)

    This function computes slope distance as given by equations (37) in ITU-R
    P.1546-5
    Input variables are
    ha - transmitting/base terminal antenna height above ground (m)
    h2 - receiving/mobile terminal antenna height above ground (m)
    d  - horizontal path distance (km)
    htter - terrain height in meters above sea level at the transmitter/base
            terminal (m)
    hrter - terrain height in meters above sea level at the receiving/mobile
            terminal (m)

    Rev   Date        Author                          Description
    --------------------------------------------------------------------------
    v2    05DEC13     Ivica Stevanovic, OFCOM         Initial version in python
    v1    12AUG13     Ivica Stevanovic, OFCOM         Initial version
    """

    if len(argc) == 0:  # function call without terrain information
        outVal = np.sqrt(d * d + 1e-6 * (ha - h2) ** 2.0)

    elif len(argc) == 2:  # function call with terrain information
        htter = argc[0]
        hrter = argc[1]

        outVal = np.sqrt(d * d + 1e-6 * ((ha + htter) - (h2 + hrter)) ** 2)

    else:
        raise ValueError("Function called with wrong number of input variables")

    return outVal


def step_18a(Emedian, q, f, pathinfo, wa, area):
    """
    E = step_18a(Emedian, q, f, env)

    Step 18: If the field strength at a receiving/mobile antenna adjacent to
    land exceeded at percentage locations other than 50% is required, correct
    the field strength for the required percentage of locations using the
    method given in Annex 5, Paragraph 12
    Input variables are
    Emedian  - field strength exceeded for 50 % of locations (as computed in
                steps 1-17 (dB(uV/m))
    q  - percentage location (betwee 1% and 99%)
    f  - required frequency (MHz)
    env - environment = 1 - for receivers with antennas below clutter height
                            in urban/suburban areas for mobile systems with
                            omnidirectional antennas at car-roof height
                    = 2 - for receivers with rooftop antennas near the
                            clutter height
                    = 3 - for receivers in rural areas
    NOTE: The location variability correction is NOT applied when the
        receiver/mobile is adjacent to sea.

    Rev   Date        Author                          Description
    -------------------------------------------------------------------------------
    v2    05DEC13     Ivica Stevanovic, OFCOM         Initial version in python
    v1    12AUG13     Ivica Stevanovic, OFCOM         Initial version
    """
    if q < 1 or q > 99:
        raise ValueError("The percentage location out of band [1%, 99%].")

    sigma_L = 0.0
    if pathinfo == 0:
        if area.find("Urban") != -1 or area.find("Dense Urban") != -1:
            sigma_L = 8.0
        elif area.find("Suburban") != -1:
            sigma_L = 10.0
        elif area.find("Rural") != -1:
            sigma_L = 12.0
        elif area.find("Sea") != -1:
            sigma_L = 0.0
        else:
            raise ValueError("Wrong area: Allowed area types: Sea, Rural, Suburban, Urban, or Dense Urban.")

    else:  # pathinfo == 1
        if area.find("Rural") != -1 or area.find("Suburban") != -1 or area.find("Urban") != -1 or area.find("Dense Urban") != -1:
            # land area only
            sigma_L = ((0.024 * f) / 1000.0 + 0.52) * wa ** (0.28)
        elif area.find("Sea") != -1:
            sigma_L = 0.0
        else:
            raise ValueError("Wrong area! Allowed area types: Sea, Rural, Suburban, Urban, or Dense Urban.")

    E = Emedian + qi(q / 100.0) * sigma_L

    return E


def step_17a(ha, h2, d, Esup, *argc):
    """
    Where terrain information is available
    Correction = step_17a(ha, h2, d, Esup, htter, hrter)
    Where terrain information is not available
    Correction = step_17a(ha,h2,d,Esup)

    Step 17: Extrapolation to distances less than 1 km
    Input variables are
    ha - transmitting/base terminal antenna height above ground (m)
    h2 - receiving/mobile terminal antenna height above ground (m)
    d  - horizontal path distance (km)
    Esup - the field strength given computed by steps 1-16 of P.1546-5
            (dB(uV/m))
    htter - terrain height in meters above sea level at the transmitter/base
            terminal (m)
    hrter - terrain height in meters above sea level at the receiving/mobile
            terminal (m)
    Note: the extension to arbitrarily short horizontal distance is based on
    the assumption that as a pah decreases in length below 1 km there is an
    increasing probability that a lower-loss path will exist passing around
    obstacles rather than over them. For paths of 0.04 km horizontal distance
    or shorter it is assumed that line-of-sight with full Fresnel clearance
    esists between the terminals, and the field strength is calulated as the
    free-space value based on slope distance.
    If these assumptions do not fit the required short-range scenario,
    appropriate adjustments should be made to account for effects such as
    street-canyon propagation, building entry, indoor sections of path, or
    body effects.
    This extension to short distances can allow the path to have a steep
    inclination, or even be vertical if ha > h2. It is important to note that
    the predicted field strength does not take into account of the vertical
    radiation pattern of the transmitting/base antenna. The field strength
    corresponds to 1 kW e.r.p. in the direction of radiation.

    Rev   Date        Author                          Description
    -------------------------------------------------------------------------------
    v2    05DEC13     Ivica Stevanovic, OFCOM         Initial version in python
    v1    12AUG13     Ivica Stevanovic, OFCOM         Initial version
    """
    if isempty(ha) or isempty(h2) or isempty(d) or isempty(Esup):
        raise ValueError("Input arguments ha, h2, d, or Esup not defined.")

    d_slope = dslope(ha, h2, d, *argc)
    dinf = 0.04
    dsup = 1

    # For paths less than 1 km the model is extended to arbitrarily short
    # horizontal distances as follows.

    # If the horizontal distance is less than or equal to 0.04 km, the field
    # strength E is given by:

    if d <= dinf:
        E = 106.9 - 20 * mt.log10(d_slope)
    else:
        dinf_slope = dslope(ha, h2, dinf, *argc)
        dsup_slope = dslope(ha, h2, dsup, *argc)
        Einf = 106.9 - 20 * np.log10(dinf_slope)
        E = Einf + (Esup - Einf) * np.log10(d_slope / dinf_slope) / np.log10(dsup_slope / dinf_slope)

    return E


def step_20a(f, E):
    """
    E = Step_20a(f, E)

    Step 20: If required, convert field strength to equivalent
    basic transmission loss for the path using the method given in Annex 5,
    Paragraph 16
    Input variables are
    f       - required frequency (MHz)
    E       - field strength (dB(uV/m)) for 1 kW of e.r.p.

    Rev   Date        Author                          Description
    ------------------------------------------------------------------------------
    v2    19MAY16     Ivica Stevanovic, OFCOM         Initial version (python)
    v1    13AUG13     Ivica Stevanovic, OFCOM         Initial version (MATLAB)
    """

    Lb = 139.3 - E + 20 * np.log10(f)

    return Lb


def read_sg3_measurements2(filename, fileformat):
    """
    sg3db=read_sg3_measurements2(filename,fileformat)

    This function reads the file <filename> from the ITU-R SG3 databank
    written using the format <fileformat> and returns output variables in the
    cell structure vargout.
    <filename> is a string defining the file in which the data is stored
    <fileformat> is the format with which the data is written in the file:
                    = 'fryderyk_cvs' (implemented)
                    = 'cvs'          (tbi)
                    = 'xml'          (tbi)

    Output variable is a struct sg3db containing the following fields
    d               - distance between Tx and Rx
    Ef              - measured field strength at distance d
    f_GHz           - frequency in GHz
    Tx_AHaG_m       - Tx antenna height above ground in m
    RX_AHaG_m       - Rx antenna height above ground in m
    etc

    Author: Ivica Stevanovic, Federal Office of Communications, Switzerland
    Revision History:
    Date            Revision
    23MAY2016       Initial python version
    06SEP2013       Introduced corrections in order to read different
                    versions of .csv file (netherlands data, RCRCU databank and kholod data)
    22JUL2013       Initial version (IS)
    """
    filename1 = filename

    sg3db = SG3DB()

    #  read the file
    try:
        fid = open(filename1, "r")
    except:
        raise IOError("P1546: File cannot be opened: " + filename1)
    if fid == -1:
        return sg3db

    #    [measurementFolder, measurementFileName, ext] = fileparts(filename1)
    measurementFolder, measurementFileName = os.path.split(filename1)
    sg3db.MeasurementFolder = measurementFolder
    sg3db.MeasurementFileName = measurementFileName

    #
    if fileformat.find("Fryderyk_csv") != -1:
        # case 'Fryderyk_csv'

        sg3db.first_point_transmitter = 1
        sg3db.coveragecode = np.array([])
        sg3db.h_ground_cover = np.array([])
        sg3db.radio_met_code = np.array([])

        # read all the lines of the file

        lines = fid.readlines()

        fid.close()

        # strip all new line characters
        lines = [line.rstrip("\n") for line in lines]

        count = 0

        while True:
            if count >= len(lines):
                break

            line = lines[count]

            dummy = line.split(",")

            if strcmp(dummy[0], "First Point Tx or Rx"):
                # if dummy[1].find('T') != -1:
                if strcmp(dummy[1], "T"):
                    sg3db.first_point_transmitter = 1
                else:
                    sg3db.first_point_transmitter = 0

            if dummy[0].find("Tot. Path Length(km):") != -1:
                TxRxDistance_km = float(dummy[1])
                sg3db.TxRxDistance = TxRxDistance_km

            if strcmp(dummy[0], "Tx site name:"):
                TxSiteName = dummy[1]
                sg3db.TxSiteName = TxSiteName

            if strcmp(dummy[0], "Rx site name:"):
                RxSiteName = dummy[1]
                sg3db.RxSiteName = RxSiteName

            if strcmp(dummy[0], "Tx Country:"):
                TxCountry = dummy[1]
                sg3db.TxCountry = TxCountry

            ## read the height profile
            if strcmp(dummy[0], "Number of Points:"):
                Npoints = int(dummy[1])

                #                sg3db.coveragecode = np.zeros((Npoints))
                #                sg3db.h_ground_cover = np.zeros((Npoints))
                #                sg3db.radio_met_code = np.zeros((Npoints))
                sg3db.x = np.zeros((Npoints))
                sg3db.h_gamsl = np.zeros((Npoints))
                for i in range(0, Npoints):
                    count = count + 1

                    readLine = lines[count]

                    dummy = readLine.split(",")
                    sg3db.x[i] = float(dummy[0])
                    sg3db.h_gamsl[i] = float(dummy[1])
                    if len(dummy) > 2:
                        value = np.nan
                        if dummy[2] != "":
                            value = float(dummy[2])
                        sg3db.coveragecode = np.append(sg3db.coveragecode, value)
                        if (len(dummy)) > 3:
                            value = np.nan
                            if dummy[3] != "":
                                value = float(dummy[3])
                            sg3db.h_ground_cover = np.append(sg3db.h_ground_cover, value)
                            if (len(dummy)) > 4:
                                # Land=4, Coast=3, Sea=1
                                value = np.nan
                                if dummy[4] != "":
                                    value = float(dummy[4])
                                sg3db.radio_met_code = np.append(sg3db.radio_met_code, value)

            ## read the field strength
            if strcmp(dummy[0], "Frequency"):
                # read the next line that defines the units
                count = count + 1
                readLine = lines[count]

                # the next line should be {Begin Measurements} and the one
                # after that the number of measurement records. However, in
                # the Dutch implementation, those two lines are missing.
                # and in the implementations of csv files from RCRU, {Begin
                # Mof Measurements} is there, but the number of
                # measurements (line after) may be missing
                # This is the reason we are checking for these two lines in
                # the following code
                f = np.array([])

                dutchflag = True
                count = count + 1
                readLine = lines[count]
                if strcmp(readLine, "{Begin of Measurements"):
                    # check if the line after that contains only one number
                    # or the data
                    count = count + 1
                    readLine = lines[count]  # the line with the number of records or not
                    dummy = readLine.split(",")
                    if len(dummy) > 2:
                        # if isempty([dummy{2:end}])

                        if dummy[1:-1] == "":
                            # this is the number of data - the info we do
                            # not use, read another line
                            count = count + 1
                            readLine = lines[count]
                            dutchflag = False
                        else:
                            dutchflag = True

                    else:
                        dutchflag = False
                        count = count + 1
                        readLine = lines[count]

                # read all the lines until the {End of Measurements} tag

                kindex = 0
                while True:
                    if kindex == 0:
                        # do not read the new line, but use the one read in
                        # the previous step
                        kindex = 0
                    else:
                        count = count + 1
                        readLine = lines[count]

                    if count >= len(lines):
                        break

                    if strcmp(readLine, "{End of Measurements}"):
                        break

                    dummy = readLine.split(",")

                    f = np.append(f, float(dummy[0]))
                    sg3db.frequency = np.append(sg3db.frequency, float(dummy[0]))

                    col = 1
                    value = np.nan
                    if dummy[col] != "":
                        value = float(dummy[col])

                    sg3db.hTx = np.append(sg3db.hTx, value)

                    col = 2
                    value = np.nan
                    if dummy[col] != "":
                        value = float(dummy[col])

                    sg3db.hTxeff = np.append(sg3db.hTxeff, value)

                    col = 3
                    value = np.nan
                    if dummy[col] != "":
                        value = float(dummy[col])
                    sg3db.hRx = np.append(sg3db.hRx, value)

                    col = 4
                    value = np.nan
                    if dummy[col] != "":
                        value = float(dummy[col])
                    sg3db.polHVC = np.append(sg3db.polHVC, value)

                    col = 5
                    value = np.nan
                    if dummy[col] != "":
                        value = float(dummy[col])
                    sg3db.TxdBm = np.append(sg3db.TxdBm, value)

                    col = 6
                    value = np.nan
                    if dummy[col] != "":
                        value = float(dummy[col])
                    sg3db.MaxLb = np.append(sg3db.MaxLb, value)

                    col = 7
                    value = np.nan
                    if dummy[col] != "":
                        value = float(dummy[col])
                    sg3db.Txgn = np.append(sg3db.Txgn, value)

                    col = 8
                    value = np.nan
                    if dummy[col] != "":
                        value = float(dummy[col])
                    sg3db.Rxgn = np.append(sg3db.Rxgn, value)

                    col = 9
                    value = np.nan
                    if dummy[col] != "":
                        value = float(dummy[col])
                    sg3db.RxAntDO = np.append(sg3db.RxAntDO, value)

                    col = 10
                    value = np.nan
                    if dummy[col] != "":
                        value = float(dummy[col])
                    sg3db.ERPMaxHoriz = np.append(sg3db.ERPMaxHoriz, value)

                    col = 11
                    value = np.nan
                    if dummy[col] != "":
                        value = float(dummy[col])
                    sg3db.ERPMaxVertical = np.append(sg3db.ERPMaxVertical, value)

                    col = 12
                    value = np.nan
                    if dummy[col] != "":
                        value = float(dummy[col])
                    sg3db.ERPMaxTotal = np.append(sg3db.ERPMaxTotal, value)

                    col = 13
                    value = np.nan
                    if dummy[col] != "":
                        value = float(dummy[col])
                    sg3db.HRPred = np.append(sg3db.HRPred, value)

                    col = 14
                    value = np.nan
                    if dummy[col] != "":
                        value = float(dummy[col])

                    if np.isnan(value):
                        print("Time percentage not defined. Default value 50% assumed.")
                        value = 50

                    sg3db.TimePercent = np.append(sg3db.TimePercent, value)

                    col = 15
                    value = np.nan
                    if dummy[col] != "":
                        value = float(dummy[col])
                    sg3db.LwrFS = np.append(sg3db.LwrFS, value)

                    col = 16
                    value = np.nan
                    if dummy[col] != "":
                        value = float(dummy[col])
                    sg3db.MeasuredFieldStrength = np.append(sg3db.MeasuredFieldStrength, value)

                    #
                    col = 17
                    value = np.nan
                    if dummy[col] != "":
                        value = float(dummy[col])
                    sg3db.BasicTransmissionLoss = np.append(sg3db.BasicTransmissionLoss, value)

                    #
                    if len(dummy) > 18:
                        col = 18

                        value = np.nan
                        if dummy[col] != "":
                            value = float(dummy[col])
                        sg3db.RxHeightGainGroup = np.append(sg3db.RxHeightGainGroup, value)

                        col = 19
                        value = np.nan
                        if dummy[col] != "":
                            value = float(dummy[col])
                        sg3db.IsTopHeightInGroup = np.append(sg3db.IsTopHeightInGroup, value)

                    else:
                        sg3db.RxHeightGainGroup = np.append(sg3db.RxHeightGainGroup, np.nan)
                        sg3db.IsTopHeightInGroup = np.append(sg3db.IsTopHeightInGroup, np.nan)

                    kindex = kindex + 1

                # Number of different measured data sets
                Ndata = kindex
                sg3db.Ndata = Ndata

            count = count + 1

    elif fileformat.find("csv") != -1:
        print("csv format not yet implemented.")

    elif fileformat.find("xml") != -1:
        print("xml format not yet implemented.")

    return sg3db


class SG3DB:
    def __init__(self):
        # sg3db : structure containing
        #  - data from ITU-R SG3 databank file
        #  - data used for P1546Compute

        self.MeasurementFolder = ""
        self.MeasurementFileName = ""
        self.first_point_transmitter = 1
        self.coveragecode = np.array([])
        self.h_ground_cover = np.array([])
        self.radio_met_code = np.array([])
        self.TxRxDistance = []
        self.TxSiteName = ""
        self.RxSiteName = ""
        self.TxCountry = ""
        self.x = np.array([])
        self.h_gamsl = np.array([])
        self.frequency = np.array([])
        self.hTx = np.array([])
        self.hTxeff = np.array([])
        self.hRx = np.array([])
        self.polHVC = np.array([])
        self.TxdBm = np.array([])
        self.MaxLb = np.array([])
        self.Txgn = np.array([])
        self.Rxgn = np.array([])
        self.RxAntDO = np.array([])
        self.ERPMaxHoriz = np.array([])
        self.ERPMaxVertical = np.array([])
        self.ERPMaxTotal = np.array([])
        self.HRPred = np.array([])
        self.TimePercent = np.array([])
        self.LwrFS = np.array([])
        self.MeasuredFieldStrength = np.array([])
        self.BasicTransmissionLoss = np.array([])
        self.RxHeightGainGroup = np.array([])
        self.IsTopHeightInGroup = np.array([])
        self.RxHeightGainGroup = np.array([])
        self.IsTopHeightInGroup = np.array([])

        self.debug = 0
        self.pathinfo = 0
        self.fid_log = 0
        self.TransmittedPower = np.array([])
        self.LandPath = 0
        self.SeaPath = 0
        self.ClutterCode = []
        self.userChoiceInt = []
        self.RxClutterCodeP1546 = ""
        self.RxClutterHeight = []
        self.TxClutterHeight = []
        self.PredictedFieldStrength = []
        self.q = 50
        self.heff = []
        self.Ndata = []
        self.eff1 = []
        self.tca = []

    def __str__(self):
        userChoiceInt = self.userChoiceInt
        out = "The following input data is defined:" + "\n"
        out = out + " PTx            = " + str(self.TransmittedPower[userChoiceInt]) + "\n"
        out = out + " f              = " + str(self.frequency[userChoiceInt]) + "\n"
        out = out + " t              = " + str(self.TimePercent[userChoiceInt]) + "\n"
        out = out + " q              = " + str(self.q) + "\n"
        out = out + " heff           = " + str(self.heff) + "\n"
        out = out + " area           = " + str(self.RxClutterCodeP1546) + "\n"
        out = out + " pathinfo       = " + str(self.pathinfo) + "\n"
        out = out + " h2             = " + str(self.hRx[userChoiceInt]) + "\n"
        out = out + " ha             = " + str(self.hTx[userChoiceInt]) + "\n"
        out = out + " htter          = " + str(self.h_gamsl[0]) + "\n"
        out = out + " hrter          = " + str(self.h_gamsl[-1]) + "\n"
        out = out + " R1             = " + str(self.TxClutterHeight) + "\n"
        out = out + " R2             = " + str(self.RxClutterHeight) + "\n"
        out = out + " eff1           = " + str(self.eff1) + "\n"
        out = out + " eff2           = " + str(self.tca) + "\n"
        out = out + " debug          = " + str(self.debug) + "\n"
        # out = out + " fid_log        = " + str(self.fid_log           ) + "\n"
        out = out + " Predicted Field Strength        = " + str(self.PredictedFieldStrength) + "\n"

        return out

    def update(self, other):
        self.MeasurementFolder = other.MeasurementFolder
        self.MeasurementFileName = other.MeasurementFileName
        self.first_point_transmitter = other.first_point_transmitter
        self.coveragecode = other.coveragecode
        self.h_ground_cover = other.h_ground_cover
        self.radio_met_code = other.radio_met_code
        self.TxRxDistance = other.TxRxDistance
        self.TxSiteName = other.TxSiteName
        self.RxSiteName = other.RxSiteName
        self.TxCountry = other.TxCountry
        self.coveragecode = other.coveragecode
        self.h_ground_cover = other.h_ground_cover
        self.radio_met_code = other.radio_met_code
        self.x = other.x
        self.h_gamsl = other.h_gamsl
        self.frequency = other.frequency
        self.hTx = other.hTx
        self.hTxeff = other.hTxeff
        self.hRx = other.hRx
        self.polHVC = other.polHVC
        self.TxdBm = other.TxdBm
        self.MaxLb = other.MaxLb
        self.Txgn = other.Txgn
        self.Rxgn = other.Rxgn
        self.RxAntDO = other.RxAntDO
        self.ERPMaxHoriz = other.ERPMaxHoriz
        self.ERPMaxVertical = other.ERPMaxVertical
        self.ERPMaxTotal = other.ERPMaxTotal
        self.HRPred = other.HRPred
        self.TimePercent = other.TimePercent
        self.LwrFS = other.LwrFS
        self.MeasuredFieldStrength = other.MeasuredFieldStrength
        self.BasicTransmissionLoss = other.BasicTransmissionLoss
        self.RxHeightGainGroup = other.RxHeightGainGroup
        self.IsTopHeightInGroup = other.IsTopHeightInGroup
        self.RxHeightGainGroup = other.RxHeightGainGroup
        self.IsTopHeightInGroup = other.IsTopHeightInGroup

        self.debug = other.debug
        self.pathinfo = other.pathinfo
        self.fid_log = other.fid_log
        self.TransmittedPower = other.TransmittedPower
        self.LandPath = other.LandPath
        self.SeaPath = other.SeaPath
        self.ClutterCode = other.ClutterCode
        self.userChoiceInt = other.userChoiceInt
        self.RxClutterCodeP1546 = other.RxClutterCodeP1546
        self.RxClutterHeight = other.RxClutterHeight
        self.TxClutterHeight = other.TxClutterHeight
        self.PredictedFieldStrength = other.PredictedFieldStrength
        self.q = other.q
        self.heff = other.heff
        self.Ndata = other.Ndata


def strcmp(str1, str2):
    """
    This function compares two strings (by previously removing any white
    spaces and open/close brackets from the strings, and changing them to
    lower case).
    Author: Ivica Stevanovic, Federal Office of Communications, Switzerland
    Revision History:
    Date            Revision
    23MAY2016       Initial python version (IS)
    22JUL2013       Initial version (IS)
    """

    str1 = str1.replace(" ", "")
    str2 = str2.replace(" ", "")
    str1 = str1.replace("(", "")
    str2 = str2.replace("(", "")
    str1 = str1.replace(")", "")
    str2 = str2.replace(")", "")
    str1 = str1.lower()
    str2 = str2.lower()

    if str1.find(str2) == -1:
        return False
    else:
        return True


def heffCalc(d, h, hT):
    """
    Effective transmitter height calculation
    heff = heffCalc(d,h,hT)
    where
    d - vector of distances (km) measured from the transmitter
    h - height profile (m), i.e., height at distance d(i)
    hT - height of the transmiter antenna above ground (m)
        This function calculates the effective height of the transmitting/base
    antenna heff defined as its height in meters over the average level of
    the ground between distances of 3 and 15 km from the transmitting/base
    antenna, in the direction of the receiving/mobile antenna. In case the
    paths are shorter than 15 km, this function returns the height in meters
    over the terrain height averaged between 0.2d and d km (or hb) as defined
    in ITU-R P.1546-4.

    Rev   Date        Author                          Description
    ------------------------------------------------------------------------------
    v3    23MAY16     Ivica Stevanovic, OFCOM         Initial python version
    v2    03OCT14     Ivica Stevanovic, OFCOM         Use trapezoids for the average height
    v1    23AUG13     Ivica Stevanovic, OFCOM         Initial version

    """

    # check for the distance between transmitter and receiver

    if d[-1] >= 15:
        # kk = ml.find(np.abs(d-d[0]) >= 3 and (d-d[0] <= 15))
        kk = np.logical_and(np.abs(d - d[0]) >= 3, (d - d[0] <= 15)).nonzero()

    else:
        # kk = ml.find(np.abs(d-d[0]) >= 0.2*d[-1] and (d-d[0] <= d[-1]))
        kk = np.logical_and(np.abs(d - d[0]) >= 0.2 * d[-1], (d - d[0] <= d[-1])).nonzero()

    x = d[kk]
    y = h[kk]

    # area=(x(2)-x(1))/2*y(1) + (x(end)-x(end-1))/2*y(end)
    area = 0

    for ii in range(0, len(x) - 1):
        area = area + (y[ii] + y[ii + 1]) * (x[ii + 1] - x[ii]) / 2.0

    hav = area / (x[-1] - x[0])
    hGlevel = h[0]
    heff = hT + hGlevel - hav

    return heff


def tcaCalc(d, h, hR, hT):
    """
    Terrain clearance angle calculator
    tca = tcaCalc(d,h)
    where
    d - vector of distances (km) measured from the transmitter
    h - height profile (m), i.e., height at distance d(i)
    hR - height of the receiver antenna above ground (m)
    hT - height of the transmitter antenna above ground (m)
    This function calculates the terrain clearance angle (tca) in [deg] using
    the method described in ITU-R P.1546-4
    tca is the elevation angle of the line from the receiving/mobile antenna
    which just clears all the terrain obstructions over a distance of up to 16 km
    but does not go beyond the transmitting/base antenna.
    Rev   Date        Author                          Description
    -------------------------------------------------------------------------------
    v3    23MAY16     Ivica Stevanovic, OFCOM         Initial python version
    v2    06SEP13     Ivica Stevanovic, OFCOM         Modified to account for hTx
    v1    22AUG13     Ivica Stevanovic, OFCOM         Initial version
    """

    # find points that satisfy d<=16km
    # kk = find( d(end)-d <= 16 )
    kk = (d[-1] - d <= 16).nonzero()

    # Find all the elevation angles for the entire height profile up to 16 km

    h1 = h[kk]
    d1 = d[kk] * 1000

    if len(d1) < 2:
        tca = 0
        return tca

    theta = np.arctan((h1[0:-1] - hR - h1[-1]) / (d1[-1] - d1[0:-1])) * 180 / np.pi

    # Find the elevation angle of the transmitter

    # thetar = np.arctan( (h[0]+hT- hR - h[-1])/(1000*(d[-1]-d[0])) )*180/np.pi

    # are there any obstructions within up to 16 km from the receiver

    # if (max(theta) < thetar) % there is no obstruction in the direction of transmitter antenna
    #    tca=0
    # else
    tca = max(theta)  # % in version -2 it was max(theta)-thetar
    # end
    return tca


def teff1Calc(d, h, hT, hR):
    """
    Terrain clearance angle calculator for transmitting/base antenna
    teff1 = teff1Calc(d,h,hT,hR)
    where
    d - vector of distances (km) measured from the transmitter
    h - height profile (m), i.e., height at distance d(i)
    hR - height of the receiver antenna above ground (m)
    hT - height of the transmitter antenna above ground (m)

    This function calculates the terrain clearance angle (tca) in [deg] using
    the method described in ITU-R P.1546-5 in §4.3a)
    tca is the elevation angle of the line from the transmitting/base antenna
    which just clears all the terrain obstructions over a distance of up to 15 km
    but does not go beyond the receiving/mobile antenna.

    Rev   Date        Author                          Description
    ------------------------------------------------------------------------------
    v2    24MAY16     Ivica Stevanovic, OFCOM         Initial python version
    v1    03OCT14     Ivica Stevanovic, OFCOM         Initial version
    """
    # find points that satisfy d<=15km

    kk = (d - d[0] <= 15).nonzero()

    # Find all the elevation angles for the entire height profile up to 15 km

    h1 = h[kk]
    d1 = d[kk] * 1000

    theta = np.arctan((h1[1:] - hT - h1[0]) / (d1[1:] - d[0])) * 180 / np.pi

    teff1 = max(theta)  # in version -2 it was max(theta)-thetar

    return teff1


def plotTca(ax, d, h, hR, tca):
    """
    Plot the  line that clears all obstractions defined by terrain clearance
    angle as in ITU-R P.1546-4 definition
    h = plotTca(d,h,hR,tca)
    where
    d - vector of distances (km) measured from the transmitter
    h - height profile (m), i.e., height at distance d(i)
    hR - height of the receiver antenna above ground (m) (at position d(end))
    tca - terrain clearance angle

    Rev   Date        Author                          Description
    ------------------------------------------------------------------------------
    v2    24MAY16     Ivica Stevanovic, OFCOM         Initial python version
    v1    23AUG13     Ivica Stevanovic, OFCOM         Initial version
    """
    # find points that satisfy d<=16km

    kk = (d[-1] - d <= 16).nonzero()

    # Find all the elevation angles for the entire height profile

    x1 = d[kk][0]  # 16 km distance from the receiver (or transmitter position)
    x2 = d[kk][-1]  # receiver position

    y1 = h[-1] + hR + (x2 - x1) * 1000 * np.tan(tca * np.pi / 180)
    y2 = h[-1] + hR
    linecolor = [0.5, 0.5, 0.5]
    # draw the lines
    # horizontal
    pl.sca(ax)
    pl.plot(np.array([x1, x2]), np.array([y2, y2]), color=linecolor)
    pl.plot(np.array([x1, x2]), np.array([y1, y2]), color=linecolor)

    return


def plotTeff1(ax, d, h, hT, teff1):
    """
    Plot the  line that clears all obstractions defined by terrain clearance
    angle as in ITU-R P.1546-5 §4.3a)
    h = plotTeff1(ax,d,h,hT,teff1)
    where
    ax - the current axes
    d - vector of distances (km) measured from the transmitter
    h - height profile (m), i.e., height at distance d(i)
    hT - height of the transmitter antenna above ground (m) (at position d(1))
    teff1 - terrain clearance angle

    Rev   Date        Author                          Description
    -------------------------------------------------------------------------------
    v2    24MAY16     Ivica Stevanovic, OFCOM         Initial python version
    v1    03OCT14     Ivica Stevanovic, OFCOM         Initial version
    """

    # find points that satisfy d<=15km

    kk = (d - d[0] <= 15).nonzero()

    # Find all the elevation angles for the entire height profile

    x1 = d[kk][0]  # transmitter position
    x2 = d[kk][-1]  # 15 km distance from the transmitter
    y1 = h[0] + hT
    y2 = h[0] + hT + (x2 - x1) * 1000 * np.tan(teff1 * np.pi / 180.0)

    linecolor = [0.5, 0.5, 0.5]
    # draw the lines
    # horizontal
    pl.sca(ax)
    pl.plot(np.array([x1, x2]), np.array([y1, y1]), color=linecolor)
    pl.plot(np.array([x1, x2]), np.array([y1, y2]), color=linecolor)

    return


def clutter(i, ClutterCodeType):
    """
    ClutterClass, P1546ClutterClass, R = clutter(i, ClutterCode)
    This function maps the value i of a given clutter code type into
    the corresponding clutter class description, P1546 clutter class description
    and clutter height R.
    The implemented ClutterCodeTypes are:
    'OFCOM' (as defined in the SG3DB database on SUI data from 2012
    'TDB'   (as defined in the RCRU database and UK data) http://www.rcru.rl.ac.uk/njt/linkdatabase/linkdatabase.php
    'NLCD'  (as defined in the National Land Cover Dataset) http://www.mrlc.gov/nlcd06_leg.php
    'LULC'  (as defined in Land Use and Land Clutter database) http://transition.fcc.gov/Bureaus/Engineering_Technology/Documents/bulletins/oet72/oet72.pdf
    'GlobCover' (as defined in ESA's GlobCover land cover maps) http://due.esrin.esa.int/globcover/
    'DNR1812' (as defined in the implementation tests for DNR P.1812)
    'default' (land paths, rural area, R = 10 m)


    Rev   Date        Author                          Description
    -------------------------------------------------------------------------------
    v3    24MAY16     Ivica St3vanovic, OFCOM         Initial python version
    v2    29Apr15     Ivica Stevanovic, OFCOM         Introduced 'default' option for ClutterCodeTypes
    v1    26SEP13     Ivica Stevanovic, OFCOM         Introduced it as a function
    """

    if strcmp(ClutterCodeType, "OFCOM"):
        if i == 0:
            RxClutterCode = "Unknown"
        elif i == 1:
            RxClutterCode = "Water (salt)"
        elif i == 2:
            RxClutterCode = "Water (fresh)"
        elif i == 3:
            RxClutterCode = "Road/Freeway"
        elif i == 4:
            RxClutterCode = "Bare land"
        elif i == 5:
            RxClutterCode = "Bare land/rock"
        elif i == 6:
            RxClutterCode = "Cultivated land"
        elif i == 7:
            RxClutterCode = "Scrub"
        elif i == 8:
            RxClutterCode = "Forest"
        elif i == 9:
            RxClutterCode = "Low dens. suburban"
        elif i == 10:
            RxClutterCode = "Suburban"
        elif i == 11:
            RxClutterCode = "Low dens. urban"
        elif i == 12:
            RxClutterCode = "Urban"
        elif i == 13:
            RxClutterCode = "Dens. urban"
        elif i == 14:
            RxClutterCode = "High dens. urban"
        elif i == 15:
            RxClutterCode = "High rise industry"
        elif i == 16:
            RxClutterCode = "Skyscraper"
        else:
            RxClutterCode = "Unknown data"

        if i == 8 or i == 12:
            RxP1546Clutter = "Urban"
            R2external = 15
        elif i == 1 or i == 2:
            RxP1546Clutter = "Sea"
            R2external = 10
        elif i < 8 and i > 2:
            RxP1546Clutter = "Rural"
            R2external = 10
        elif i <= 11 and i > 8:
            RxP1546Clutter = "Suburban"
            R2external = 10
        elif i <= 16 and i > 11:
            RxP1546Clutter = "Dense Urban"
            R2external = 20
        else:
            RxP1546Clutter = ""
            R2external = []

    elif strcmp(ClutterCodeType, "TDB"):
        if i == 0:
            RxClutterCode = "No data"
            RxP1546Clutter = ""
            R2external = []
        elif i == 1:
            RxClutterCode = "Fields"
            RxP1546Clutter = "Rural"
            R2external = 10
        elif i == 2:
            RxClutterCode = "Road"
            RxP1546Clutter = "Rural"
            R2external = 10
        elif i == 3:
            RxClutterCode = "BUILDINGS"
            RxP1546Clutter = "Urban"
            R2external = 20
        elif i == 4:
            RxClutterCode = "URBAN"
            RxP1546Clutter = "Urban"
            R2external = 20
        elif i == 5:
            RxClutterCode = "SUBURBAN"
            RxP1546Clutter = "Suburban"
            R2external = 10
        elif i == 6:
            RxClutterCode = "VILLAGE"
            RxP1546Clutter = "Suburban"
            R2external = 10
        elif i == 7:
            RxClutterCode = "SEA"
            RxP1546Clutter = "Adjacent to Sea"
            R2external = 10
        elif i == 8:
            RxClutterCode = "LAKE"
            RxP1546Clutter = "Adjacent to Sea"
            R2external = 10
        elif i == 9:
            RxClutterCode = "RIVER"
            RxP1546Clutter = "Rural"
            R2external = 10
        elif i == 10:
            RxClutterCode = "CONIFER"
            RxP1546Clutter = "Rural"
            R2external = 10
        elif i == 11:
            RxClutterCode = "NON_CONIFER"
            RxP1546Clutter = "Rural"
            R2external = 10
        elif i == 12:
            RxClutterCode = "MUD"
            RxP1546Clutter = "Rural"
            R2external = 10
        elif i == 13:
            RxClutterCode = "ORCHARD"
            RxP1546Clutter = "Rural"
            R2external = 10
        elif i == 14:
            RxClutterCode = "MIXED_TREES"
            RxP1546Clutter = "Urban"
            R2external = 20
        elif i == 15:
            RxClutterCode = "DENSE_URBAN"
            RxP1546Clutter = "Dense Urban"
            R2external = 30
        else:
            RxClutterCode = "Unknown data"
            RxP1546Clutter = ""
            R2external = []

    elif strcmp(ClutterCodeType, "NLCD"):
        if i == 11:
            RxClutterCode = "Open Water"
            RxP1546Clutter = "Adjacent to Sea"
            R2external = 10
        elif i == 12:
            RxClutterCode = "Perennial Ice/Snow"
            RxP1546Clutter = "Rural"
            R2external = 10

        elif i == 21:
            RxClutterCode = "Developed, Open Space"
            RxP1546Clutter = "Suburban"
            R2external = 10
        elif i == 22:
            RxClutterCode = "Developed, Low Intensity"
            RxP1546Clutter = "Suburban"
            R2external = 10
        elif i == 23:
            RxClutterCode = "Developed, Medium Intensity"
            RxP1546Clutter = "Suburban"
            R2external = 10
        elif i == 24:
            RxClutterCode = "Developed High Intensity"
            RxP1546Clutter = "Urban"
            R2external = 20

        elif i == 31:
            RxClutterCode = "Barren Land (Rock/Sand/Clay)"
            RxP1546Clutter = "Rural"
            R2external = 10

        elif i == 41:
            RxClutterCode = "Deciduous Forest"
            RxP1546Clutter = "Urban"
            R2external = 20
        elif i == 42:
            RxClutterCode = "Evergreen Forest"
            RxP1546Clutter = "Urban"
            R2external = 20
        elif i == 43:
            RxClutterCode = "Mixed Forest"
            RxP1546Clutter = "Urban"
            R2external = 20

        elif i == 51:
            RxClutterCode = "Dwarf Scrub"
            RxP1546Clutter = "Rural"
            R2external = 10
        elif i == 52:
            RxClutterCode = "Shrub/Scrub"
            RxP1546Clutter = "Rural"
            R2external = 10

        elif i == 71:
            RxClutterCode = "Grassland/Herbaceous"
            RxP1546Clutter = "Rural"
            R2external = 10
        elif i == 72:
            RxClutterCode = "Sedge/Herbaceous"
            RxP1546Clutter = "Rural"
            R2external = 10
        elif i == 73:
            RxClutterCode = "Lichens - Alaska only"
            RxP1546Clutter = "Rural"
            R2external = 10
        elif i == 74:
            RxClutterCode = "Moss - Alaska only"
            RxP1546Clutter = "Rural"
            R2external = 10

        elif i == 81:
            RxClutterCode = "Pasture/Hay"
            RxP1546Clutter = "Rural"
            R2external = 10
        elif i == 82:
            RxClutterCode = "Cultivated Crops"
            RxP1546Clutter = "Rural"
            R2external = 10

        elif i == 90:
            RxClutterCode = "Woody Wetlands"
            RxP1546Clutter = "Adjacent to Sea"
            R2external = 10
        elif i == 95:
            RxClutterCode = "Emergent Herbaceous Wetlands"
            RxP1546Clutter = "Adjacent to Sea"
            R2external = 10
        else:
            RxClutterCode = "Unknown data"
            RxP1546Clutter = ""
            R2external = []

    elif strcmp(ClutterCodeType, "LULC"):
        if i == 11:
            RxClutterCode = "Residential"
            RxP1546Clutter = "Urban"
            R2external = 20
        elif i == 12:
            RxClutterCode = "Commercial services"
            RxP1546Clutter = "Urban"
            R2external = 20
        elif i == 13:
            RxClutterCode = "Industrial"
            RxP1546Clutter = "Urban"
            R2external = 20
        elif i == 14:
            RxClutterCode = "Transportation, communications, utilities"
            RxP1546Clutter = "Rural"
            R2external = 10
        elif i == 15:
            RxClutterCode = "Industrial and commercial complexes"
            RxP1546Clutter = "Urban"
            R2external = 20
        elif i == 16:
            RxClutterCode = "Mixed urban and built-up lands"
            RxP1546Clutter = "Suburban"
            R2external = 10
        elif i == 17:
            RxClutterCode = "Other urban and built-up land"
            RxP1546Clutter = "Suburban"
            R2external = 10

        elif i == 21:
            RxClutterCode = "Cropland and pasture"
            RxP1546Clutter = "Rural"
            R2external = 10
        elif i == 22:
            RxClutterCode = "Orchards, groves, vineyards, nurseries, and horticultural"
            RxP1546Clutter = "Rural"
            R2external = 10
        elif i == 23:
            RxClutterCode = "Confined feeding operations"
            RxP1546Clutter = "Rural"
            R2external = 10
        elif i == 24:
            RxClutterCode = "Other agricultural land"
            RxP1546Clutter = "Rural"
            R2external = 10
        elif i == 31:
            RxClutterCode = "Herbaceous rangeland"
            RxP1546Clutter = "Rural"
            R2external = 10
        elif i == 32:
            RxClutterCode = "Shrub and brush rangeland"
            RxP1546Clutter = "Rural"
            R2external = 10
        elif i == 33:
            RxClutterCode = "Mixed rangeland"
            RxP1546Clutter = "Rural"
            R2external = 10

        elif i == 41:
            RxClutterCode = "Deciduous forest land"
            RxP1546Clutter = "Urban"
            R2external = 20
        elif i == 42:
            RxClutterCode = "Evergreen forest land"
            RxP1546Clutter = "Urban"
            R2external = 20
        elif i == 43:
            RxClutterCode = "Mixed forest land"
            RxP1546Clutter = "Urban"
            R2external = 20

        elif i == 51:
            RxClutterCode = "Streams and canals"
            RxP1546Clutter = "Adjacent to Sea"
            R2external = 10
        elif i == 52:
            RxClutterCode = "Lakes"
            RxP1546Clutter = "Adjacent to Sea"
            R2external = 10
        elif i == 53:
            RxClutterCode = "Reservoirs"
            RxP1546Clutter = "Adjacent to Sea"
            R2external = 10
        elif i == 54:
            RxClutterCode = "Bays and estuaries"
            RxP1546Clutter = "Adjacent to Sea"
            R2external = 10

        elif i == 61:
            RxClutterCode = "Forested wetland"
            RxP1546Clutter = "Urban"
            R2external = 20
        elif i == 62:
            RxClutterCode = "Non-forest wetland"
            RxP1546Clutter = "Adjacent to Sea"
            R2external = 10

        elif i == 71:
            RxClutterCode = "Dry salt flats"
            RxP1546Clutter = "Rural"
            R2external = 10
        elif i == 72:
            RxClutterCode = "Beaches"
            RxP1546Clutter = "Adjacent to Sea"
            R2external = 10
        elif i == 73:
            RxClutterCode = "Sandy areas other than beaches"
            RxP1546Clutter = "Rural"
            R2external = 10
        elif i == 74:
            RxClutterCode = "Bare exposed rock"
            RxP1546Clutter = "Rural"
            R2external = 10
        elif i == 75:
            RxClutterCode = "Strip mines, quarries, and gravel pits"
            RxP1546Clutter = "Rural"
            R2external = 10
        elif i == 76:
            RxClutterCode = "Transitional areas"
            RxP1546Clutter = "Rural"
            R2external = 10
        elif i == 77:
            RxClutterCode = "Mixed barren land"
            RxP1546Clutter = "Rural"
            R2external = 10

        elif i == 81:
            RxClutterCode = "Shrub and brush tundra"
            RxP1546Clutter = "Rural"
            R2external = 10
        elif i == 82:
            RxClutterCode = "Herbaceous tundra"
            RxP1546Clutter = "Rural"
            R2external = 10
        elif i == 83:
            RxClutterCode = "Bare ground"
            RxP1546Clutter = "Rural"
            R2external = 10
        elif i == 84:
            RxClutterCode = "Wet tundra"
            RxP1546Clutter = "Rural"
            R2external = 10
        elif i == 85:
            RxClutterCode = "Mixed tundra"
            RxP1546Clutter = "Rural"
            R2external = 10

        elif i == 91:
            RxClutterCode = "Perennial snowfields"
            RxP1546Clutter = "Rural"
            R2external = 10
        elif i == 92:
            RxClutterCode = "Glaciers"
            RxP1546Clutter = "Rural"
            R2external = 10
        else:
            RxClutterCode = "Unknown data"
            RxP1546Clutter = ""
            R2external = []

    elif strcmp(ClutterCodeType, "GlobCover"):
        if i == 1:
            RxClutterCode = "Water/Sea"
            RxP1546Clutter = "Sea"
            R2external = 10
        elif i == 2:
            RxClutterCode = "Open/Rural"
            RxP1546Clutter = "Rural"
            R2external = 10
        elif i == 3:
            RxClutterCode = "Suburban"
            RxP1546Clutter = "Suburban"
            R2external = 10
        elif i == 4:
            RxClutterCode = "Urban/trees/forest"
            RxP1546Clutter = "Urban"
            R2external = 15
        elif i == 5:
            RxClutterCode = "Dense Urban"
            RxP1546Clutter = "Dense Urban"
            R2external = 20
        else:
            RxClutterCode = "Unknown data"
            RxP1546Clutter = ""
            R2external = []

    elif strcmp(ClutterCodeType, "P1546"):
        if i == 1:
            RxClutterCode = "Water/Sea"
            RxP1546Clutter = "Sea"
            R2external = 10
        elif i == 2:
            RxClutterCode = "Open/Rural"
            RxP1546Clutter = "Rural"
            R2external = 10
        elif i == 3:
            RxClutterCode = "Suburban"
            RxP1546Clutter = "Suburban"
            R2external = 10
        elif i == 4:
            RxClutterCode = "Urban/trees/forest"
            RxP1546Clutter = "Urban"
            R2external = 15
        elif i == 5:
            RxClutterCode = "Dense Urban"
            RxP1546Clutter = "Dense Urban"
            R2external = 20
        else:
            RxClutterCode = "Unknown"
            RxP1546Clutter = "Suburban"
            R2external = 0

    elif strcmp(ClutterCodeType, "DNR1812"):
        if i == 0:
            RxClutterCode = "Inland"
            RxP1546Clutter = "Rural"
            R2external = 10
        elif i == 1:
            RxClutterCode = "Coastal"
            RxP1546Clutter = "Sea"
            R2external = 10
        elif i == 2:
            RxClutterCode = "Sea"
            RxP1546Clutter = "Sea"
            R2external = 10
        else:
            RxClutterCode = "Unknown data"
            RxP1546Clutter = ""
            R2external = []

    elif strcmp(ClutterCodeType, "default"):
        print("Clutter code type set to default:")
        print("Rural, R = 10 m")
        RxClutterCode = "default"
        RxP1546Clutter = "Rural"
        R2external = 10

    else:
        RxClutterCode = ""
        RxP1546Clutter = ""
        R2external = []

    return RxClutterCode, RxP1546Clutter, R2external


def Compute(sg3db):
    """
    sg3db = Compute(sg3db)
    this function computes the field strength and path loss for the data
    structure defined in sg3db, according to ITU-R P.1546-6

    Author: Ivica Stevanovic, Federal Office of Communications, Switzerland
    Revision History:
    Date            Revision
    16MAR2022       Updates related to P.1546-6
    24MAY2016       Initial python version

    """

    # Collect all the input data

    userChoiceInt = sg3db.userChoiceInt
    data = Data()

    if isempty(userChoiceInt):
        raise ValueError("Dataset not defined.")

    data.PTx = sg3db.TransmittedPower[userChoiceInt]

    data.f = sg3db.frequency[userChoiceInt]

    data.t = sg3db.TimePercent[userChoiceInt]

    data.q = 50
    data.wa = sg3db.wa

    data.heff = sg3db.heff

    data.area = sg3db.RxClutterCodeP1546
    if not (strcmp(data.area, "Sea") or strcmp(data.area, "Rural") or strcmp(data.area, "Suburban") or strcmp(data.area, "Urban") or strcmp(data.area, "Dense Urban")):
        raise ValueError("Allowed P.1546 Rx Clutter Types: Sea, Rural, Suburban, Urban, or Dense Urban.")

    data.pathinfo = sg3db.pathinfo

    kindex = 0
    d_v = np.array([])
    path_c = np.array([])

    if sg3db.LandPath > 0:
        d_v = np.append(d_v, sg3db.LandPath)
        path_c = np.append(path_c, "Land")

    if sg3db.SeaPath > 0:
        d_v = np.append(d_v, sg3db.SeaPath)
        path_c = np.append(path_c, "Sea")

    data.d_v = d_v.copy()
    data.path_c = path_c.copy()

    data.NN = len(d_v)

    data.h2 = sg3db.h2

    data.ha = sg3db.ha

    data.htter = sg3db.htter
    data.hrter = sg3db.hrter

    hb = []  # correction to follow section 3.1.2
    if np.sum(d_v) < 15:
        data.hb = sg3db.heff

    data.R1 = sg3db.TxClutterHeight
    data.R2 = sg3db.RxClutterHeight

    data.eff1 = sg3db.eff1
    data.eff2 = sg3db.tca
    data.tca = sg3db.tca

    data.debug = sg3db.debug
    data.fid_log = sg3db.fid_log

    # check input variables

    if checkInput(data):
        Es, L = bt_loss(data.f, data.t, data.heff, data.h2, data.R2, data.area, data.d_v, data.path_c, data.pathinfo, data.q, data.wa, data.PTx, data.ha, data.hb, data.R1, data.tca, data.htter, data.hrter, data.eff1, data.eff2, data.debug, data.fid_log)

        sg3db.PredictedFieldStrength = Es
        sg3db.PredictedTransmissionLoss = L

    return sg3db


class Data:
    def __init__(self):
        # data : structure containing input arguments to the function P1546.btl

        self.f = []
        self.t = []
        self.heff = []
        self.h2 = []
        self.R2 = []
        self.area = []
        self.d_v = []
        self.path_c = []
        self.pathinfo = []
        self.q = []
        self.wa = []
        self.PTx = []
        self.ha = []
        self.hb = []
        self.R1 = []
        self.tca = []
        self.htter = []
        self.hrter = []
        self.eff1 = []
        self.eff2 = []
        self.debug = []
        self.fid_log = []

    def __str__(self):
        out = "The following input data is defined:" + "\n"
        out = out + " f          = " + str(self.f) + "\n"
        out = out + " t          = " + str(self.t) + "\n"
        out = out + " heff       = " + str(self.heff) + "\n"
        out = out + " h2         = " + str(self.heff) + "\n"
        out = out + " R2         = " + str(self.R2) + "\n"
        out = out + " area       = " + str(self.area) + "\n"
        out = out + " d_v        = " + str(self.d_v) + "\n"
        out = out + " path_c     = " + str(self.path_c) + "\n"
        out = out + " pathinfo   = " + str(self.pathinfo) + "\n"
        out = out + " q          = " + str(self.q) + "\n"
        out = out + " wa          = " + str(self.wa) + "\n"
        out = out + " PTx        = " + str(self.PTx) + "\n"
        out = out + " ha         = " + str(self.ha) + "\n"
        out = out + " hb         = " + str(self.hb) + "\n"
        out = out + " R1         = " + str(self.R1) + "\n"
        out = out + " tca        = " + str(self.tca) + "\n"
        out = out + " htter      = " + str(self.htter) + "\n"
        out = out + " hrter      = " + str(self.hrter) + "\n"
        out = out + " eff1       = " + str(self.eff1) + "\n"
        out = out + " eff2       = " + str(self.eff2) + "\n"
        out = out + " debug      = " + str(self.debug) + "\n"
        # out = out + " fid_log    = " + str(self.fid_log  ) + "\n"

        return out

    def update(self, other):
        self.f = other.f
        self.t = other.t
        self.heff = other.heff
        self.h2 = other.h2
        self.R2 = other.R2
        self.area = other.area
        self.d_v = other.d_v
        self.path_c = other.path_c.copy()
        self.pathinfo = other.pathinfo.copy()
        self.q = other.q
        self.wa = other.wa
        self.PTx = other.PTx
        self.ha = other.ha
        self.hb = other.hb
        self.R1 = other.R1
        self.tca = other.tca
        self.htter = other.htter
        self.hrter = other.hrter
        self.eff1 = other.eff1
        self.eff2 = other.eff2
        self.debug = other.debug
        self.fid_log = other.fid_log


def checkInput(data):
    alldefined = True

    val = data.PTx

    if ~np.isnan(val):
        if val <= 0:
            print("Transmission power P must be defined and positive")
            alldefined = False
            return alldefined

    else:
        print("Transmission power P must be defined and positive")
        alldefined = False
        return alldefined

    val = data.f
    if ~np.isnan(val):
        if val < 30 or val > 4000:
            print("Frequency f must be defined within 30 MHz - 4000 MHz")
            alldefined = False
            return alldefined

    else:
        print("Frequency f must be defined within 30 MHz - 4000 MHz")
        alldefined = False
        return alldefined

    val = data.t
    errormsg = "Time percentage must be defined within 1% - 50%"
    if ~np.isnan(val):
        if val < 1 or val > 50:
            print(errormsg)
            alldefined = False
            return alldefined

    else:
        print(errormsg)
        alldefined = False
        return alldefined

    val = data.q
    errormsg = "Location variability must be defined within 1% - 99%"
    if ~np.isnan(val):
        if val < 1 or val > 99:
            print(errormsg)
            alldefined = False
            return alldefined

    else:
        print(errormsg)
        alldefined = False
        return alldefined

    val = data.heff
    errormsg = "Effective height of the transmitter antenna not defined"
    if np.isnan(val):
        print(errormsg)
        alldefined = False
        return alldefined
    else:
        if isempty(val):
            print("Effective height of the transmitter antenna not defined")
            alldefined = False
            return alldefined

    val = data.area
    errormsg = "Receiver area type must be defined"
    if isempty(val):
        print(errormsg)
        alldefined = False
        return alldefined

    if data.NN == 0:
        print("There must be at least one path type defined.")
        alldefined = False
        return alldefined

    if isempty(data.d_v):
        print("There must be at least one path length defined.")
        alldefined = False
        return alldefined

    if isempty(data.path_c):
        print("There must be at least one path type defined.")
        alldefined = False
        return alldefined

    for ii in range(0, len(data.path_c)):
        if np.isnan(data.d_v[ii]):
            print("Path type and path distances are not properly defined.")
            alldefined = False
            return alldefined

        if data.d_v[ii] <= 0:
            print("Path distance must be a positive number.")
            alldefined = False
            return alldefined

    return alldefined


def isempty(x):
    if np.size(x) == 0:
        return True
    else:
        return False
