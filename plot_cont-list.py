# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 11:46:15 2017

@author: river603
"""

from numba import njit, jit, f8
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import csv
import os
import re
import glob
import shutil


def length_height(x,y,z,nnn,xbmin,xbmax):
    
    s  = x[0,:]
    wdz = z[0,:]
    dz = wdz-np.mean(wdz)
    
    window = 5 # 移動平均の範囲
    w = np.ones(window)/window

    dz2 = np.convolve(dz, w, mode='same')
        
    
    zmin = []
    zmax = []
    hb = []
    
    nx = len(s)-1
    
    for i in range(nx+1):
        zmin.append(min(z[:,i]))
        zmax.append(max(z[:,i]))
        hb.append(max(z[:,i])-min(z[:,i]))
#        print(s[i],min(z[:,i]),max(z[:,i]))
        
    ip = []
    
    for i in range(nx):
        if np.sign(dz2[i])<0. and np.sign(dz2[i+1])>0. and np.abs(zmax[i]-zmin[i])>0.0001:
            ip.append(i)
                
    nn = len(ip)-1
    
    ipp = 0
    
#    if np.mod(nn,2)!=0:
#        ipp = 1
    
    ll = []
    hh = []
    
#    for n in range(0,nn-ipp,2):
    for n in range(0,nn-1):
        wl = s[ip[n+1]]-s[ip[n]]
#        wh = np.max(zmax[ip[n]:ip[n+1]])-np.min(zmin[ip[n]:ip[n+1]])
        wh = np.max(hb[ip[n]:ip[n+1]])
        
#        if 80>s[ip[n]]>40.:
        if xbmax>s[ip[n]]>xbmin:
        
            ll.append(wl)
            hh.append(wh)

    L = 0.
    H = 0.
    
#    nb = len(ll)
    
#    if nb>=1:
            
#        L = np.mean(ll)
#        H = np.mean(hh)
                
#    return L, H
#    plt.figure(figsize=(10,2.5),tight_layout=True)
#    plt.ylim([-0.05,0.025])
#    plt.plot(s,zmin)
#    plt.plot(s,zmax)
#    plt.plot(s,dz2)
    
#    for n in range(0,nn-1):
#        plt.plot(s[ip[n]],dz2[ip[n]],'o')
        
#    plt.savefig('bar'+str(nnn)+'.jpg', dpi=600)
#    plt.close()
    

    return ll, hh    

def phase_celerity(X, Zprev, Z, dt, xmin, xmax, max_shift, rmin):

    # 対象領域の抽出
    s = X[0, :]

    ix = np.where((s >= xmin) & (s <= xmax))[0]

    if len(ix) < 5:
        return np.nan, np.nan, np.nan

    A = Zprev[:, ix].copy()
    B = Z[:, ix].copy()

    # NaN, Inf のチェック
    if not (np.all(np.isfinite(A)) and
            np.all(np.isfinite(B))):
        return np.nan, np.nan, np.nan

    # 格子間隔
    dx = np.mean(np.diff(s[ix]))

    # 最大移動距離を格子数に変換
    nshift = int(max_shift / dx)

    n = A.shape[1]

    if nshift >= n-2:
        nshift = n-2

    corr = []
    shifts = []

    # 縦断方向にずらして相関係数を計算
    for k in range(-nshift, nshift+1):

        if k > 0:
            a = A[:, :-k]
            b = B[:, k:]

        elif k < 0:
            a = A[:, -k:]
            b = B[:, :k]

        else:
            a = A
            b = B

        aa = a.ravel()
        bb = b.ravel()

        aa = aa - np.mean(aa)
        bb = bb - np.mean(bb)

        denom = np.sqrt(np.sum(aa**2)*np.sum(bb**2))

        if denom > 0.:

            r = np.sum(aa*bb)/denom

        else:
            r = np.nan

        corr.append(r)
        shifts.append(k)

    corr = np.array(corr)

    if not np.any(np.isfinite(corr)):
        return np.nan, np.nan, np.nan

    # 相関係数最大の位置
    imax = np.nanargmax(corr)

    kmax = shifts[imax]

    # サブグリッド補間（放物線近似）
    dk = 0.

    if 0 < imax < len(corr)-1:

        cm = corr[imax-1]
        c0 = corr[imax]
        cp = corr[imax+1]

        den = cm - 2.*c0 + cp

        if np.isfinite(den) and abs(den) > 1.e-12:

            dk = 0.5*(cm-cp)/den

    # 移動距離
    dxb = (kmax+dk)*dx

    if corr[imax]<rmin:
        cp = np.nan
    else:

    # 位相速度
        cp = dxb/dt*100.*60.  # cm/min

    return cp, corr[imax], dxb
    

#path_lists = ['S50/c-mb-hllc/']
#path_lists = ['C4/f-vl-hllc/']
path_lists = ['C4/c-hll/','C4/c-mb-hllc/','C4/c-vl-hllc/','C4/f-hll/','C4/f-mb-hllc/','C4/f-vl-hllc/']
#path_lists = ['S50/c-hll/','S50/c-mb-hllc/','S50/c-vl-hllc/'
#              ,'S50/f-hll/','S50/f-mb-hllc/','S50/f-vl-hllc/']
#path_lists = ['S50/c-hll/','S50/c-mb-hllc/','S50/c-vl-hllc/'
#              ,'S50/f-hll/','S50/f-mb-hllc/','S50/f-vl-hllc/']
#path_lists = ['S50/f-hll/']
filename  = 'tt299'

for path_list in path_lists:

    inputfiles = glob.glob(path_list+"*.txt")
    #inputfiles = glob.glob("*.txt")
    
    fig_path = path_list+"/fig/"
    if os.path.exists(fig_path):
        shutil.rmtree(fig_path)
    
    os.mkdir(fig_path)
    
    #nx = 1000
    #ny = 20
    
    #nx = 180
    #ny = 150
    
    output = open(path_list+"T-LH-2.dat",'w')
    
    output.write(
    "Time(min) Length(m) Height(m) "
    "Phase_celerity(m/s) Correlation Shift(m)\n"
    )
    
    nn = 0
        
    # 位相速度計算用
    Zprev = None
    prev_tt = None
                                
    cp_phase = np.nan
    rmax = np.nan
    dxb = np.nan
                                
    Z0 = None
    
    idt = 4
    
    xbmin = 60.
    xbmax = 110.
    max_shift = 6.
    rmin = 0.75
            
    for inputfile in inputfiles[::idt]:
        
        filename = os.path.splitext(os.path.basename(inputfile))[0]
        foldername = os.path.dirname(inputfile)
        figname = foldername+"/fig/"+filename+".jpg"
    #    figname = "fig/"+inputfile+".jpg"

    
        with open(inputfile, 'r') as ifile:
            line = ifile.readline()
            head = line.replace('\n', '').split()
            
            tt = float(head[0])
            nx = int(head[1])
            ny = int(head[2])
                    
            lines = ifile.readlines()
                
            x  = []
            y  = []
            z  = []
            
#            for line in lines:
#               value = line.replace('\n', '').split()
#               x.append(float(value[0]))
#               y.append(float(value[1]))
#               z.append(float(value[2]))

            valid_data = True
            
            for line in lines:
                value = line.split()
            
                try:
                    xx = float(value[0])
                    yy = float(value[1])
                    zz = float(value[2])
            
                    # NaN, Inf のチェック
                    if not (np.isfinite(xx) and
                            np.isfinite(yy) and
                            np.isfinite(zz)):
                        valid_data = False
                        break
            
                    x.append(xx)
                    y.append(yy)
                    z.append(zz)
                except (ValueError, IndexError):
                    valid_data = False
                    break
                    
            if not valid_data or len(z) != (nx+1)*(ny+1):

                print("Invalid data:", inputfile)
            
                # 波長・波高データには NaN を記録
                aaa = str(tt/60.) + "\tNaN\tNaN\tNaN\tNaN\n"
                output.write(aaa)
                            
            else:
                               
                wX = [[0 for i in range(nx+1)] for j in range(ny+1)]
                wY = [[0 for i in range(nx+1)] for j in range(ny+1)]
                wZ = [[0 for i in range(nx+1)] for j in range(ny+1)]
                
                X = np.array(wX, dtype="f8")
                Y = np.array(wY, dtype="f8")
                Z = np.array(wZ, dtype="f8")
                
                y_offset = 0.45
                
                ij = 0
            #    for i in range(nx+1):
            #        for j in range(ny+1):
                for j in range(ny+1):
                    for i in range(nx+1):
                        X[j][i] = x[ij]
                        Y[j][i] = y[ij]-y_offset
                        Z[j][i] = z[ij]
                        
                        ij = ij+1
                        
                if Z0 is None:
                    Z0 = Z.copy()
                
                Z = Z-Z0
                
                # ----------------------------------
                # 位相速度の計算
                # ----------------------------------
                
                cp_phase = np.nan
                rmax = np.nan
                dxb = np.nan
                
                # 河床形状に十分な変動があるか確認
                if np.std(Z) > 1.e-4:

                    if Zprev is not None:
                
                        dt = tt-prev_tt
                
                        if dt > 0.:
                
                            cp_phase, rmax, dxb = phase_celerity(
                                X, Zprev, Z, dt, xbmin, xbmax, max_shift, rmin
                            )
                
                    else:
                        print("First valid bed:", tt/60.)
                
                    Zprev = Z.copy()
                    prev_tt = tt
                
                else:
                    print("No bed variation:", tt/60., np.std(Z))
                
                        
                L, H = length_height(X,Y,Z,nn,xbmin,xbmax)
                
                if len(L)>0:
                    lave = np.mean(L)
                    have = np.mean(H)
                    lmax = np.max(L)
                    hmax = np.max(H)
                    lmin = np.min(L)
                    hmin = np.min(H)
                else:
                    lave = 0.
                    have = 0.
                    lmax = 0.
                    hmax = 0.
                    lmin = 0.
                    hmin = 0.
                
                print(f"{tt/60:15.6f} {lave:15.6f} {have:15.6f} {cp_phase:15.6f} {rmax:15.6f}")
                
                
        #        aaa = str(tt/60.)+"\t"+str(L)+"\t"+str(H)+"\n"
        #        aaa = str(tt/60.)+"\t"+str(lave)+"\t"+str(have)+"\t"+str(lmax)+"\t"+str(hmax)+"\t"+str(lmin)+"\t"+str(hmin)+"\n"
        #        aaa = str(tt/60.)+"\t"+str(lave)+"\t"+str(have)+"\n"
                aaa = (
                    str(tt/60.) + "\t" +
                    str(lave) + "\t" +
                    str(have) + "\t" +
                    str(cp_phase) + "\t" +
                    str(rmax) + "\n"
                )
                output.write(aaa)
                        
                plt.figure(figsize=(10,2.5),tight_layout=True)
                plt.axes().set_aspect('equal')
                plt.xlabel('X(m)')
                plt.ylabel('Y(m)')
                plt.subplots_adjust(bottom=0.2)
                
                zmin = -0.05
                zmax =  0.025
        #        xmax = 115.
        #        xmin = 100.
                xmax = 110.
                xmin = 50.
        #        xmax = 140.
        #        xmin = 80.
                ymin = -y_offset
                ymax =  y_offset
                
            #    plt.axis([30,45,0,0.9])
            #    plt.xticks( np.arange(30,50,5) )
                plt.axis([xmin,xmax,ymin,ymax])
                plt.xticks( np.arange(xmin,xmax*1.01,5) )
                plt.yticks( np.arange(ymin,ymax*1.01,y_offset) )
                levels = np.linspace(zmin,zmax,51)
                labels = np.linspace(zmin,zmax,6)
                
                plt.title("Time= {0:.0f} min".format(tt/60)) 
                    
                plt.contourf(X, Y, Z, levels, cmap=cm.copper, extend="both")
                plt.colorbar(ticks=labels, label='Bed evolution (m)', orientation='horizontal', fraction=0.09, pad=0.3)
                plt.savefig(figname, dpi=600)
                plt.close()
                
        ifile.close()
        
        nn += 1
    
    output.close() 
        
        