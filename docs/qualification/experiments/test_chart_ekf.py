"""Control: first-order prediction with the recovered finite chart and prior."""
import argparse,json
from pathlib import Path
import casadi as ca
import manta.estimation._ins_moments as moments
from manta.estimation._kalman import symmetrize
from manta.estimation._assembly import _q_auto

def predict(sys,spec,P,C,*,process_noise=True,extra_Q=None):
    q=sys.packet_Q_sym
    if process_noise:q=q+_q_auto(sys)
    if extra_Q is not None:q=q+extra_Q
    return sys.x_new,symmetrize(sys.F_sym@P@sys.F_sym.T+q),sys.F_sym@C if C is not None else None
moments.predict_moments=predict
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--seeds',type=int,default=64);p.add_argument('--mounted',action='store_true');p.add_argument('--noisy',action='store_true');p.add_argument('--seed',type=int,default=60191)
a=p.parse_args()
from examples.qualification.earth_ins_split import run
r=run(covariance='nonlinear',mounted=a.mounted,seeds=a.seeds,seed=a.seed,
      gyro_density=.001 if a.noisy else 1e-7,bias_sigma=.001 if a.noisy else 1e-5)
a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r['records'][-1]),flush=True)
