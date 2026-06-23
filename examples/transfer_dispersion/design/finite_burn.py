#!/usr/bin/env python3
"""Stage 2: transform the impulsive porkchop optimum into LOW-THRUST finite-burn
rendezvous trajectories for three thrust levels (0.1/0.2/0.3 N @ 1000 kg).
Each impulse becomes a finite constant-direction thrust arc; the burn structure
(tau1,phi1,tau2,phi2) is solved so the spacecraft truly rendezvous with the NEA.
The extra dV over the impulsive optimum is the finite-burn (low-thrust) penalty."""
import numpy as np
from lamberthub import izzo2015
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

VU=29.7847; YR=2*np.pi; ACCEL=5.9301e-3
SCR=''

def earth(t): return np.array([np.cos(t),np.sin(t)]), np.array([-np.sin(t),np.cos(t)])
class NEA:
    def __init__(s,a,e,w,M0): s.a=a;s.e=e;s.w=w;s.M0=M0;s.n=a**-1.5
    def state(s,t):
        M=s.M0+s.n*t; E=M
        for _ in range(60): E=E-(E-s.e*np.sin(E)-M)/(1-s.e*np.cos(E))
        b=np.sqrt(1-s.e**2)
        r=s.a*(1-s.e*np.cos(E)); Ed=s.n/(1-s.e*np.cos(E))
        po=np.array([s.a*(np.cos(E)-s.e),s.a*b*np.sin(E)])
        vo=np.array([-s.a*np.sin(E)*Ed,s.a*b*np.cos(E)*Ed])
        c,sn=np.cos(s.w),np.sin(s.w); R=np.array([[c,-sn],[sn,c]])
        return R@po,R@vo
d=np.load(SCR+'pork.npz')
nea=NEA(float(d['nea_a']),float(d['nea_e']),float(d['nea_w']),float(d['nea_M0']))
td=float(d['best_td']); tof=float(d['best_tof']); Mrev=int(d['best_M'])
rE,vE=earth(td); rN,vN=nea.state(td+tof)
v1=np.asarray(d['best_v1']); v2=np.asarray(d['best_v2'])  # multi-rev optimum
dv1=v1-vE; dv2=vN-v2; DVimp=np.hypot(*dv1)+np.hypot(*dv2)
print(f"impulsive optimum: dep {np.hypot(*dv1)*VU:.3f} + arr {np.hypot(*dv2)*VU:.3f} "
      f"= {DVimp*VU:.3f} km/s ; dep t={td/YR:.3f} yr TOF={tof/YR:.3f} yr")

def f2b(t,s,acc):
    x,y,vx,vy=s; r3=(x*x+y*y)**1.5
    return [vx,vy,-x/r3+acc[0],-y/r3+acc[1]]
def step(s,T,acc):
    if T<=0: return np.array(s)
    return solve_ivp(f2b,[0,T],s,args=(acc,),rtol=1e-9,atol=1e-11).y[:,-1]
def propagate(p,a_lt,T,full=False):
    tau1,phi1,tau2,phi2=p; coast=T-tau1-tau2
    s=np.r_[rE,vE]
    a1=a_lt*np.array([np.cos(phi1),np.sin(phi1)]); a2=a_lt*np.array([np.cos(phi2),np.sin(phi2)])
    if not full:
        s=step(s,tau1,a1); s=step(s,coast,np.zeros(2)); s=step(s,tau2,a2); return s
    segs=[]
    for T_,acc,kind in [(tau1,a1,'thrust'),(coast,np.zeros(2),'coast'),(tau2,a2,'thrust')]:
        if T_<=0: continue
        sol=solve_ivp(f2b,[0,T_],s,args=(acc,),rtol=1e-11,atol=1e-13,dense_output=True,max_step=0.05)
        tt=np.linspace(0,T_,max(40,int(400*T_/T))); segs.append((kind,sol.sol(tt)))
        s=sol.y[:,-1]
    return s,segs

cases=[("0.10 N",0.1/1000/ACCEL,'#1f77b4'),
       ("0.20 N",0.2/1000/ACCEL,'#2ca02c'),
       ("0.30 N",0.3/1000/ACCEL,'#d6491f')]
PERYR=1.0/YR  # TU -> yr
results=[]
for name,a_lt,col in cases:
    g0=np.array([np.hypot(*dv1)/a_lt, np.arctan2(dv1[1],dv1[0]),
                 np.hypot(*dv2)/a_lt, np.arctan2(dv2[1],dv2[0])])
    best=None; g=g0.copy()
    for T in np.linspace(tof,3.2*tof,18):
        rNT,vNT=nea.state(td+T)
        sol,info,ier,msg=fsolve(lambda p:propagate(p,a_lt,T)-np.r_[rNT,vNT],g,
                                full_output=True,xtol=1e-11)
        tau1,phi1,tau2,phi2=sol; coast=T-tau1-tau2
        miss=np.hypot(*(propagate(sol,a_lt,T)[:2]-rNT))
        if ier==1 and tau1>0 and tau2>0 and coast>=-1e-6 and miss<1e-7:
            g=sol  # warm-start next T
            DVlt=a_lt*(tau1+tau2)
            if best is None or DVlt<best[5]: best=(name,a_lt,col,sol.copy(),T,DVlt)
    name,a_lt,col,sol,T,DVlt=best
    tau1,phi1,tau2,phi2=sol
    print(f"{name}: a={a_lt:.4f} ({a_lt*ACCEL*1e3:.3f} mm/s^2)  T={T*PERYR:.3f} yr  "
          f"burns {tau1*PERYR:.3f}+{tau2*PERYR:.3f} yr  coast {(T-tau1-tau2)*PERYR:.3f} yr  "
          f"dV={DVlt*VU:.3f} km/s  penalty {DVlt/DVimp:.2f}x")
    results.append(best)
np.savez(SCR+'lt.npz',td=td,tof=tof,DVimp=DVimp,best_v1=v1,best_v2=v2,best_M=Mrev,
         names=[r[0] for r in results],a_lts=[r[1] for r in results],
         cols=[r[2] for r in results],
         params=np.array([r[3] for r in results]),Ts=[r[4] for r in results],
         DVlts=[r[5] for r in results],
         nea_a=nea.a,nea_e=nea.e,nea_w=nea.w,nea_M0=nea.M0)
print("done")
