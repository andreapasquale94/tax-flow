#!/usr/bin/env python3
"""Stage 2 plots: three low-thrust cases as a 1x3 grid (one panel per thrust
level), in the inertial and Sun-Earth rotating frames, on a POLAR grid centred
on the Sun. Thrust arcs red, coast blue; impulsive Lambert black dotted; the
NEA arc (grey) spans the spacecraft timeline +-30 days."""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

VU=29.7847; YR=2*np.pi; D30=90.0*YR/365.25   # 90 days in TU (NEA arc pad)
SCR=''
RED='#d6491f'; BLUE='#1f77b4'; GREY='0.6'

def earth(t): return np.array([np.cos(t),np.sin(t)]), np.array([-np.sin(t),np.cos(t)])
class NEA:
    def __init__(s,a,e,w,M0): s.a=a;s.e=e;s.w=w;s.M0=M0;s.n=a**-1.5
    def state(s,t):
        M=s.M0+s.n*t; E=M
        for _ in range(60): E=E-(E-s.e*np.sin(E)-M)/(1-s.e*np.cos(E))
        b=np.sqrt(1-s.e**2); Ed=s.n/(1-s.e*np.cos(E))
        po=np.array([s.a*(np.cos(E)-s.e),s.a*b*np.sin(E)])
        vo=np.array([-s.a*np.sin(E)*Ed,s.a*b*np.cos(E)*Ed])
        c,sn=np.cos(s.w),np.sin(s.w); R=np.array([[c,-sn],[sn,c]])
        return R@po,R@vo

d=np.load(SCR+'lt.npz',allow_pickle=True)
td=float(d['td']); tof=float(d['tof']); DVimp=float(d['DVimp'])
nea=NEA(float(d['nea_a']),float(d['nea_e']),float(d['nea_w']),float(d['nea_M0']))
names=list(d['names']); a_lts=list(d['a_lts']); params=d['params']
Ts=list(d['Ts']); DVlts=list(d['DVlts']); v1opt=np.asarray(d['best_v1'])
rE,vE=earth(td)

def f2b(t,s,acc):
    x,y,vx,vy=s; r3=(x*x+y*y)**1.5
    return [vx,vy,-x/r3+acc[0],-y/r3+acc[1]]
def segs_of(p,a_lt,T):
    tau1,phi1,tau2,phi2=p; coast=T-tau1-tau2
    s=np.r_[rE,vE]; out=[]; t0=td
    a1=a_lt*np.array([np.cos(phi1),np.sin(phi1)]); a2=a_lt*np.array([np.cos(phi2),np.sin(phi2)])
    for T_,acc,kind in [(tau1,a1,'thrust'),(coast,np.zeros(2),'coast'),(tau2,a2,'thrust')]:
        if T_<=0: continue
        sol=solve_ivp(f2b,[0,T_],s,args=(acc,),rtol=1e-10,atol=1e-12,dense_output=True,max_step=0.05)
        tt=np.linspace(0,T_,max(60,int(500*T_/T)))
        out.append((kind,t0+tt,sol.sol(tt))); s=sol.y[:,-1]; t0+=T_
    return out
soli=solve_ivp(f2b,[0,tof],np.r_[rE,v1opt[:2]],args=(np.zeros(2),),rtol=1e-10,atol=1e-12,
               dense_output=True,max_step=0.05)
ti=np.linspace(0,tof,400); yi=soli.sol(ti)
def rot(t,xy): c,s=np.cos(t),np.sin(t); return np.array([c*xy[0]+s*xy[1],-s*xy[0]+c*xy[1]])

def tp(xy):     # cartesian path -> (unwrapped theta, r) for polar plotting
    return np.unwrap(np.arctan2(xy[1],xy[0])), np.hypot(xy[0],xy[1])
def pt(xy):     # single point -> (theta, r)
    return np.arctan2(xy[1],xy[0]), np.hypot(xy[0],xy[1])

def panels(frame):
    fig,axes=plt.subplots(1,3,figsize=(16.2,6.4),subplot_kw={'projection':'polar'})
    for ax,(name,a_lt,p,T,dv) in zip(axes,zip(names,a_lts,params,Ts,DVlts)):
        tn=np.linspace(td-D30, td+T+D30, 500)         # NEA arc: SC timeline +-30 d
        arr=nea.state(td+T)[0]
        if frame=='inertial':
            ax.plot(*tp(np.array([nea.state(t)[0] for t in tn]).T),color=GREY,lw=1.4)      # NEA
            ax.plot(np.linspace(0,2*np.pi,200),np.ones(200),color='0.8',lw=1.0,ls='--')    # Earth
            ax.plot(*tp(yi[:2]),color='k',lw=1.1,ls=':')                                    # lambert
            for kind,tt,yy in segs_of(p,a_lt,T):
                ax.plot(*tp(yy[:2]),color=RED if kind=='thrust' else BLUE,
                        lw=2.6 if kind=='thrust' else 1.7)
            ax.plot(*pt(rE),marker='o',color='k',ms=8,zorder=7)
            ax.plot(*pt(arr),marker='D',color='k',ms=8,zorder=7)
        else:
            ax.plot(*tp(np.array([rot(t,nea.state(t)[0]) for t in tn]).T),color=GREY,lw=1.4)
            ax.plot(*tp(np.array([rot(td+ti[i],yi[:2,i]) for i in range(len(ti))]).T),
                    color='k',lw=1.1,ls=':')
            for kind,tt,yy in segs_of(p,a_lt,T):
                yr=np.array([rot(tt[i],yy[:2,i]) for i in range(len(tt))]).T
                ax.plot(*tp(yr),color=RED if kind=='thrust' else BLUE,
                        lw=2.6 if kind=='thrust' else 1.7)
            ax.plot(0.0,1.0,marker='o',color='k',ms=8,zorder=7)                             # Earth fixed
            ax.plot(*pt(rot(td+T,arr)),marker='D',color='k',ms=8,zorder=7)
        ax.set_rmin(0.25); ax.set_rmax(1.28)
        ax.set_rticks([0.5,0.75,1.0,1.25]); ax.set_rlabel_position(-15)
        if frame=='rotating':
            ax.set_thetamin(-90); ax.set_thetamax(30)   # 270 -> 30 deg sector
            ax.set_thetagrids(range(-90,31,30))
        else:
            ax.set_thetagrids(range(0,360,30))          # full circle (multi-rev)
        ax.tick_params(labelsize=8); ax.grid(alpha=0.4)
        ax.set_title(f'{name}   ($\\Delta v$={dv*VU:.2f} km/s,  {dv/DVimp:.2f}$\\times$)',
                     fontsize=12,pad=20)
    h=[Line2D([],[],color=RED,lw=2.6,label='thrust'),
       Line2D([],[],color=BLUE,lw=1.7,label='coast'),
       Line2D([],[],color='k',lw=1.1,ls=':',label='lambert'),
       Line2D([],[],color=GREY,lw=1.4,label='NEA orbit'),
       Line2D([],[],marker='o',color='k',ls='',label='departure'),
       Line2D([],[],marker='D',color='k',ls='',label='arrival')]
    fig.legend(handles=h,loc='center',ncol=len(h),fontsize=10,frameon=False,
               bbox_to_anchor=(0.5,0.08))
    fig.subplots_adjust(left=0.04,right=0.98,top=0.93,bottom=0.16,wspace=0.22)
    out=SCR+('lt_inertial.png' if frame=='inertial' else 'lt_rotating.png')
    fig.savefig(out,dpi=145); print("saved",out)

panels('inertial'); panels('rotating')
