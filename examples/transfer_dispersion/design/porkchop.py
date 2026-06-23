#!/usr/bin/env python3
"""Stage 1: heliocentric two-body porkchop (impulsive Lambert) Earth -> near-Earth
NEA. Canonical units mu=1, 1 DU=1 AU, Earth orbit r=v=1, period 2*pi. Picks a
low-dV target and the optimum departure/TOF. Blues_r contour."""
import numpy as np
from lamberthub import izzo2015
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

VU=29.7847   # km/s per canonical speed unit
YR=2*np.pi   # one year in TU
SCR=''

# ---- planar multi-rev Lambert via lamberthub (Izzo 2015), mu=1 --------------
def best_lambert(rE,vE,rN,vN,dt):
    """Min-dV rendezvous over revolutions M=0,1,2 (both branches)."""
    r1=np.array([rE[0],rE[1],0.0]); r2=np.array([rN[0],rN[1],0.0]); best=None
    for M in (0,1,2):
        for lp in ((True,) if M==0 else (True,False)):
            try:
                v1,v2=izzo2015(1.0,r1,r2,dt,M=M,prograde=True,low_path=lp,
                               maxiter=35,atol=1e-7,rtol=1e-6)
            except Exception:
                continue
            if not (np.all(np.isfinite(v1)) and np.all(np.isfinite(v2))): continue
            dv=np.hypot(*(v1[:2]-vE))+np.hypot(*(vN-v2[:2]))
            if best is None or dv<best[0]: best=(dv,v1[:2],v2[:2],M)
    return best

# ---- Earth (circular) and NEA (Keplerian) ----------------------------------
def earth(t):
    return np.array([np.cos(t),np.sin(t)]), np.array([-np.sin(t),np.cos(t)])
class NEA:
    def __init__(s,a,e,w,M0): s.a=a;s.e=e;s.w=w;s.M0=M0;s.n=a**-1.5
    def state(s,t):
        M=s.M0+s.n*t; E=M
        for _ in range(50): E=E-(E-s.e*np.sin(E)-M)/(1-s.e*np.cos(E))
        b=np.sqrt(1-s.e**2)
        xo=s.a*(np.cos(E)-s.e); yo=s.a*b*np.sin(E)
        r=s.a*(1-s.e*np.cos(E)); Edot=s.n/(1-s.e*np.cos(E))
        vxo=-s.a*np.sin(E)*Edot; vyo=s.a*b*np.cos(E)*Edot
        c,sn=np.cos(s.w),np.sin(s.w); R=np.array([[c,-sn],[sn,c]])
        return R@np.array([xo,yo]), R@np.array([vxo,vyo])

# Earth-CROSSING Apollo NEA: q=0.95, Q=1.15 AU -> a=1.05, e=0.0952. Moving a
# away from 1 keeps the synodic period workable (~14 yr) vs a near-1-AU orbit.
nea=NEA(a=1.05, e=0.0952, w=np.radians(50.0), M0=np.radians(150.0))

# Auto-tune the NEA phase (M0) so a cheap launch window falls within a few years.
def _minDV(M0):
    nea.M0=M0; m=1e9
    for td in np.linspace(0.5*YR,4.5*YR,16):
        rE,vE=earth(td)
        for tof in np.linspace(0.4*YR,2.4*YR,16):
            rN,vN=nea.state(td+tof); b=best_lambert(rE,vE,rN,vN,tof)
            if b: m=min(m,b[0])
    return m
best_M0=min(np.linspace(0,2*np.pi,36,endpoint=False),key=_minDV); nea.M0=best_M0
print(f"auto-tuned NEA M0 = {np.degrees(best_M0):.1f} deg (min dV ~ {_minDV(best_M0)*VU:.2f} km/s)")

# ---- porkchop grid (multi-rev) ----------------------------------------------
deps=np.linspace(0.0, 6.0*YR, 200)
tofs=np.linspace(0.3*YR, 2.6*YR, 170)
DV=np.full((len(tofs),len(deps)),np.nan)
MG=np.full((len(tofs),len(deps)),np.nan)   # winning revolution number per cell
best=None
for i,td in enumerate(deps):
    rE,vE=earth(td)
    for j,tof in enumerate(tofs):
        rN,vN=nea.state(td+tof)
        b=best_lambert(rE,vE,rN,vN,tof)
        if b is None: continue
        DV[j,i]=b[0]; MG[j,i]=b[3]
        if best is None or b[0]<best[0]: best=(b[0],td,tof,b[1],b[2],rE,vE,rN,vN,b[3])
dvmin,td_,tof_,v1_,v2_,rE_,vE_,rN_,vN_,M_=best
print(f"min impulsive dV = {dvmin*VU:.3f} km/s  (M={M_} rev)")
print(f"  departure t = {td_/YR:.3f} yr, TOF = {tof_/YR:.3f} yr, arrival t = {(td_+tof_)/YR:.3f} yr")
print(f"  dep dV = {np.hypot(*(v1_-vE_))*VU:.3f} km/s, arr dV = {np.hypot(*(vN_-v2_))*VU:.3f} km/s")
np.savez(SCR+'pork.npz',deps=deps,tofs=tofs,DV=DV,MG=MG,best_td=td_,best_tof=tof_,
         best_v1=v1_,best_v2=v2_,best_M=M_,
         nea_a=nea.a,nea_e=nea.e,nea_w=nea.w,nea_M0=nea.M0,dvmin=dvmin)

# ---- plot porkchop: Blues_r dV + alpha-ed revolution-number overlay ---------
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
cmap=LinearSegmentedColormap.from_list("b",plt.get_cmap("Blues_r")(np.linspace(0.0,0.85,256)))
fig,ax=plt.subplots(figsize=(8.8,6.6))
X,Y=np.meshgrid(deps/YR,tofs/YR)
dvk=DV*VU
lo=np.nanmin(dvk); levels=np.linspace(lo,lo+6.0,25)
cf=ax.contourf(X,Y,dvk,levels=levels,cmap=cmap,extend='max',zorder=1)
# revolution-number layer (categorical, semi-transparent) on top
revcols=['#e66101','#1b9e77','#7570b3']  # M=0,1,2
mcmap=ListedColormap(revcols); mnorm=BoundaryNorm([-0.5,0.5,1.5,2.5],mcmap.N)
ax.pcolormesh(X,Y,np.ma.masked_invalid(MG),cmap=mcmap,norm=mnorm,alpha=0.33,
              shading='nearest',zorder=2)
cs=ax.contour(X,Y,dvk,levels=levels[::3],colors='k',linewidths=0.4,alpha=0.45,zorder=3)
ax.clabel(cs,fmt='%.1f',fontsize=7)
ax.plot(td_/YR,tof_/YR,marker='*',ms=20,color='#d6491f',mec='k',zorder=6,
        label=f'optimum {dvmin*VU:.2f} km/s (M={M_})')
cb=fig.colorbar(cf,ax=ax,pad=0.02); cb.set_label('total rendezvous $\\Delta v$ [km/s]')
mh=[Patch(facecolor=revcols[m],alpha=0.5,label=f'M = {m} rev') for m in range(3)]
leg1=ax.legend(handles=mh,loc='upper left',fontsize=8.5,framealpha=0.9,title='best solution')
ax.add_artist(leg1)
ax.legend(loc='upper right',fontsize=9,framealpha=0.9)
ax.set_xlabel('departure epoch [yr]'); ax.set_ylabel('time of flight [yr]')
ax.set_title('Porkchop: Earth $\\to$ NEA (impulsive Lambert, multi-rev)')
fig.tight_layout(); fig.savefig(SCR+'porkchop.png',dpi=145)
print("saved porkchop.png")
