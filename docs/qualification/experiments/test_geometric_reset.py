"""Isolated experiment: finite two-frame chart with its differentiated reset."""
import importlib
import json
import argparse
from pathlib import Path
import casadi as ca
from manta.estimation._kalman import symmetrize
from manta.ir._rotation import quat_to_rotmat
from manta.ir._linalg import spd_solve
import os
if os.environ.get('INS_CHART')=='gravity_bias':
    from gravity_bias_chart import INSStateSpec
elif os.environ.get('INS_CHART')=='aligned':
    from gravity_aligned_chart import INSStateSpec
elif os.environ.get('INS_CHART')=='semidirect':
    from semidirect_chart import INSStateSpec
else:
    from geometric_chart import INSStateSpec

ins_module=importlib.import_module('manta.estimation.ins')
base_system=ins_module._INSSystem
class GeometricSystem(base_system):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        extra={'earth':self.navigation_frame.angular_velocity} if os.environ.get('INS_CHART')=='aligned' else {}
        self.spec=INSStateSpec(self.spec,craft=self.craft_name,imu=self.imu_name,
                              rotation_body_from_imu=self.R_craft_from_sensor,**extra)
ins_module._INSSystem=GeometricSystem

def update(x,P,h,H,R,z,spec):
    if os.environ.get('INS_OUTPUT')=='midpoint':
        rotation=quat_to_rotmat(x[3:7])
        H=ca.MX(H)
        H[:,3:6]=.5*rotation.T@ca.skew(x[7:10]+rotation@z)
    S=H@P@H.T+R
    K=spd_solve(S,(P@H.T).T).T
    residual=z-h;d=K@residual
    new=spec.boxplus_sym(x,d)
    A=ca.MX.eye(P.size1())-K@H
    G=spec.exact_reset(x,d)
    return new,symmetrize(G@(A@P@A.T+K@R@K.T)@G.T),residual,S
ins_module.joseph_update=update

p=argparse.ArgumentParser()
p.add_argument('--output',type=Path,required=True)
p.add_argument('--bias-sigma',type=float,default=1e-5)
p.add_argument('--gyro-density',type=float,default=1e-7)
p.add_argument('--duration',type=float,default=300)
p.add_argument('--seed',type=int,default=82719)
a=p.parse_args()
from examples.qualification.earth_ins import run
r=run(bias_sigma=a.bias_sigma,gyro_density=a.gyro_density,duration=a.duration,seed=a.seed)
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({'acceptance':r['acceptance'],**r['records'][-1]}),flush=True)
