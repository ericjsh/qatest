def qatest(fname) :
    '''
    QATEST ver 1.0.0

    Input Format : 
    fname = 'ks4.1389.203-38.B.dith1.kmts.20210508.061237.kk.fits'

    Ouput Format : 
    - Updates FITS header with new QA (Quality Assurance) info :
        HISTORY   Quality Assurance (QA) by QATEST version 1.0.0 (2022-01-28)
        COMMENT   2022 JSH
        CCDNAME =                      / Name of CCD
        QAREFCAT=                      / Reference Catalog used for QA
        QAALNNUM=                      / Number of objects for QA [integer]
        QAALNRMS=                      / RMS of misalignment with QAREFCAT [arcsec]
        QAALNSTD=                      / Uncertainty of misalignment with QAREFCAT [arcs
        QANSECT =                      / Total number of divided sections for QA [intege
        QAGDSECT=                      / Number of sections classified as good [integer]
        QARESULT=                      / True if QA is good

    - RETURN QA result (list) : [File name, QA result, Bad sect] 
    '''
    #       =========
    # ====== IMPORTS ========================================================
    #       =========
    import os,sys
    from datetime import date
    from astropy.table import Table
    from astropy.io import fits,ascii
    from astropy.stats import SigmaClip,sigma_clip
    import astropy.units as u
    import astropy.coordinates as coord
    from astropy.coordinates import SkyCoord,Angle,match_coordinates_sky
    from astroquery.vizier import Vizier
    import numpy as np
    from numpy import std,sqrt,mean,median
    import pandas as pd
    import warnings

    warnings.simplefilter('ignore', UserWarning)

    print(' ================================================================================== \n             running qatest({}) \n =================================================================================='.format(fname))

    #       =====================
    # ====== PREPARING WORKPLACE ========================================================
    #       =====================


    ks4_database = '/data5/ks4/database/'
    kmtrefcatdir = '/data3/ericjsh/kmt_refcat/'
    kmtconfigdir = '/data3/ericjsh/kmt_config/'

    #fname   = 'ks4.1389.203-38.B.dith1.kmts.20210508.061237.kk.fits'
    fnum1,fnum2,ffilt = fname.split('.')[1:4]
    ffield   = '.'.join([fnum1, fnum2])
    absfpath = os.path.join(ks4_database, ffilt, ffield, fname)

    if os.path.isfile(absfpath) == True :
        os.system('cp {} ./'.format(absfpath))
        #os.system('ln -s '+absfpath)
        os.system('cp -r '+kmtconfigdir+'* ./')

        # Reading FITS file
        with fits.open(fname) as hdul :
            hdr   = hdul[0].header
            radd  = coord.Angle(hdr['RA'], unit = u.hour).degree
            decdd = coord.Angle(hdr['DEC'], unit = u.deg).degree



        #       ========================
        # ====== Quality Assurance (QA) ===================================================
        #       ========================
        # ---------------------------------------
        # Matching source(input) with reference |
        # ---------------------------------------
        # Loading Input Catalog
        os.system('sex '+fname+' -c default.sex -CATALOG_NAME '+fname+'.cat \
                    -DETECT_THRESH 10 -ANALYSIS_THRESH 10')

        data     = ascii.read(fname+'.cat')
        data     = data[np.where(data['FLAGS'] == 0)]

        data['KRON_RADIUS_A'] = [data['KRON_RADIUS'][i]*data['A_IMAGE'][i] for i in range(len(data))]
        trkerr = data[np.where(data['KRON_RADIUS_A'] > 200)]
        if len(trkerr) < 50 :
            # Loading Reference Catalog
            field      = Table.read(os.path.join(kmtrefcatdir,'ks4field.fits'))
            trgt_field = field[np.where((field['ra[deg]']  >= radd  - 0.8) & 
                                        (field['ra[deg]']  <= radd  + 0.8) & 
                                        (field['dec[deg]'] >= decdd - 0.8) & 
                                        (field['dec[deg]'] <= decdd + 0.8))]

            ls_trgt_idx = [i for i in trgt_field['field_name1']]
            ls_trgt     = ['refcat{}.fits'.format(i) for i in ls_trgt_idx]

            if len(ls_trgt) >= 1 :
                refcat_name = 'GAIA EDR3'
                refcat      = pd.concat([pd.DataFrame(np.array(fits.getdata(os.path.join(kmtrefcatdir, f))).byteswap().newbyteorder()) for f in ls_trgt])
            else :
                refcat_name = 'GAIA EDR3'
                v = Vizier(columns = ['RAJ2000', 'DEJ2000'])
                v.ROW_LIMIT = -1 # no row limit
                v.TIMEOUT   = 500
                result      = v.query_region(coord.SkyCoord(ra=radd, dec=decdd, unit=(u.deg, u.deg), frame='icrs'), 
                                             width   = 2.8*u.deg, 
                                             catalog = [refcat_name],
                                             cache   = False)
                refcat = result[0].to_pandas()  

            # Matching input and reference catalog : mref
            incoord  = SkyCoord(data['ALPHA_J2000'], data['DELTA_J2000'], 
                                unit = (u.deg, u.deg))
            refcoord = SkyCoord(refcat['RAJ2000'], refcat['DEJ2000'], 
                                unit = (u.deg, u.deg))

            indx, d2d, d3d = match_coordinates_sky(incoord, refcoord, nthneighbor=1)

            mref          = refcat.iloc[indx].copy()
            mref['sep']   = d2d *3600
            mref_colnames = ['datara', 'datadec', 'x_pixel', 'y_pixel']
            data_colnames = ['ALPHA_J2000', 'DELTA_J2000', 'XWIN_IMAGE', 'YWIN_IMAGE']
            for i in range(len(mref_colnames)) : mref[mref_colnames[i]] = data[data_colnames[i]] 
            mrefcut       = mref[mref['sep'] < 2]

            # Sigma Clipping for misalignment
            sigmaclip = sigma_clip(mrefcut['sep'], sigma=3, maxiters=None, cenfunc=median, stdfunc=std)
            clip      = np.where(sigmaclip.mask == False)
            csep      = mrefcut['sep'].iloc[clip]

            # ------------------------
            # Analyzing 8X8 sections |
            # ------------------------
            # Dividing chip into 8X8 sections
            X0 = mref['x_pixel']
            Y0 = mref['y_pixel']
            X  = mrefcut['x_pixel']
            Y  = mrefcut['y_pixel']

            df_sect = pd.DataFrame(columns = ['dtect num', 'rmsalign', 'alignstd', 'astrometry', 'dtct ratio'])
            divnum = 8
            for j in range(divnum) :
                ymin = np.min(Y0) + j*(np.max(Y0) - np.min(Y0))/divnum
                ymax = np.min(Y0) + (j+1)*(np.max(Y0) - np.min(Y0))/divnum
                for i in range(divnum) :
                    xmin = np.min(X0) + i*(np.max(X0) - np.min(X0))/divnum
                    xmax = np.min(X0) + (i+1)*(np.max(X0) - np.min(X0))/divnum
                    def step(i, j) :
                        return (i >= xmin) & (i  <= xmax) & (j >= ymin) & (j  <= ymax)
                    sep0 = mref[step(X0, Y0)]['sep']
                    sep  = mrefcut[step(X, Y)]['sep']

                    dtctRatio = len(sep)/len(sep0) if len(sep0) != 0 else 0
                    rmsalign  = sqrt(mean((sep)**2)) if dtctRatio != 0 else 99
                    alignstd  = std(sep) if dtctRatio != 0 else 99
                    #if len(sep0) == 0 :
                    #    dtctRatio = 0
                    #else :
                    #    dtctRatio = len(sep)/len(sep0)
                    #if dtctRatio != 0 :
                    #    rmsalign  = sqrt(mean((sep)**2))
                    #    alignstd  = std(sep)
                    #else :
                    #    rmsalign = 99
                    #    alignstd = 99
                    sect_astrom = 'good' if dtctRatio > 0.6 and rmsalign < 1 else 'bad'
                    df_sect.loc[8*j + i] = [len(sep), rmsalign, alignstd, sect_astrom, dtctRatio]

            # Analysis report
            gbmap_row = list(df_sect['astrometry'])
            bad_sect  = [i for i, x in enumerate(gbmap_row) if x == 'bad']
            fastrom   = 'good' if gbmap_row.count('bad') <= 2 else 'bad'
            result    = [fname, fastrom, bad_sect]

            # Update Header
            hdrcmthist = {
                'HISTORY' : '  Quality Assurance (QA) by QATEST version 1.0.0 ({})'.format(date.today().strftime('%Y-%m-%d')),
                'COMMENT' : '  2022 JSH', #update with github
            }

            hdrupdate = {
                'QAREFCAT': (refcat_name, 'Reference Catalog used for QA'),
                'QAALNNUM': (len(csep), 'Number of objects for QA [integer]'),
                'QAALNRMS': (float(format(sqrt(mean(csep**2)), '.5f')), 'RMS of misalignment with QAREFCAT [arcsec]'),
                'QAALNSTD': (float(format(std(csep), '.5f')), 'Uncertainty of misalignment with QAREFCAT [arcsec]'),
                'QANSECT' : (divnum**2, 'Total number of divided sections for QA [integer]'),
                'QAGDSECT': (gbmap_row.count('good'), 'Number of sections classified as good [integer]'),
                'QARESULT': (fastrom == 'good', 'True if QA is good')
            }

            with fits.open(fname, 'update') as hdul:
                hdr = hdul[0].header
                for cmthist in hdrcmthist.keys() :
                    hdr.insert(len(hdr), (cmthist, hdrcmthist[cmthist]))
                hdr.insert(len(hdr), ('CCDNAME', fname.split('.')[8], 'Name of CCD'))
                for hdrkey in hdrupdate.keys() :
                    hdr[hdrkey] = hdrupdate[hdrkey]

            if fastrom == 'good' :
                os.system('mv {} goodastrom/'.format(fname))
            else :
                with open('/data3/ericjsh/qatest/badastrom1.txt', 'a') as f :
                    f.write('badastrom {} \n'.format(fname))
                os.system('mv {} badastrom/'.format(fname))
            #os.system('unlink '+fname)
            #os.system('rm {}'.format(fname))

        else : # len(trkerr) >= 50
            with open('/data3/ericjsh/qatest/badastrom1.txt', 'a') as f :
                f.write('trackerr {} \n'.format(fname))
            result = ['tracking issue', '!', '!']
            print('***'+absfpath+' tracking issue! ***')

        os.system('rm {}.cat'.format(fname))

    else : #os.path.isfile(absfpath) == False
        with open('/data3/ericjsh/qatest/badastrom1.txt', 'a') as f :
            f.write('fpatherr {} \n'.format(fname))
        result = ['file missing', '!', '!']
        print('***'+absfpath+' is missing! ***')

    return result