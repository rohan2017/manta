"""Joint finite posterior reset, retaining navigation/nuisance cross moments."""
import argparse,json,importlib
from pathlib import Path
import casadi as ca
from manta.estimation._kalman import symmetrize,unscented_weights,ut_predict
from manta.estimation._ins_moments import psd_root
from manta.ir._linalg import spd_solve
ins_module=importlib.import_module('manta.estimation.ins')

def update(x,P,C,Pc,h,H,Hc,R,z,spec):
    n=P.size1();nc=Pc.size1()
    joint=ca.vertcat(ca.horzcat(P,C),ca.horzcat(C.T,Pc))
    HH=ca.horzcat(H,Hc);S=HH@joint@HH.T+R
    K=spd_solve(S,(P@H.T+C@Hc.T).T).T
    nu=z-h;d=K@nu
    gain=ca.vertcat(K,ca.MX.zeros(nc,z.numel()))
    A=ca.MX.eye(n+nc)-gain@HH
    posterior=symmetrize(A@joint@A.T+gain@R@gain.T)
    L=psd_root(posterior);wm,wc,gamma=unscented_weights(n+nc,1.,2.,0.)
    deltas=[ca.MX.zeros(n+nc)]+[sign*gamma*L[:,i] for sign in [1,-1] for i in range(n+nc)]
    points=[spec.boxplus_sym(x,d+e[:n]) for e in deltas]
    mean,cov=ut_predict(points,ca.MX.zeros(n,n),wm,wc,spec,3)
    cross=sum((w*spec.boxminus_sym(point,mean)@e[n:].T for w,point,e in zip(wc,points,deltas)),ca.MX.zeros(n,nc))
    return mean,cov,cross,nu,S
ins_module.schmidt_update=update

def raw(x,P,h,H,R,z,spec):
    n=P.size1();mean,cov,_,nu,S=update(x,P,ca.MX.zeros(n,0),ca.MX.zeros(0,0),h,H,ca.MX.zeros(z.numel(),0),R,z,spec)
    return mean,cov,nu,S
ins_module.joseph_update=raw
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--mounted',action='store_true');p.add_argument('--seed',type=int,default=82719)
a=p.parse_args()
from examples.qualification.earth_ins_split import run
r=run(covariance='nonlinear',gyro_density=1e-7,bias_sigma=1e-5,mounted=a.mounted,seed=a.seed,seeds=64)
r['reset_method']='joint finite Gaussian posterior moments'
a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'acceptance':r['acceptance'],**r['records'][-1]}),flush=True)
