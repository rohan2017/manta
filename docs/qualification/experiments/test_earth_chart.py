"""Bounded test: retain the full Earth-rate/attitude/bias stationary family."""
import argparse,json
from pathlib import Path
import casadi as ca
import numpy as np
import manta.estimation._ins_error as chart_module
from manta.estimation._ins_error import INSStateSpec as Base, _ambient, _tangent, _left_inverse
from manta.estimation._kalman import _so3_reset_jacobian
from manta.ir._rotation import so3_exp,quat_mul,quat_to_rotmat
from examples.qualification.earth_ins import SPIN

class Chart(Base):
    error_model='earth_and_gravity_finite_chart_experiment'
    def earth_offset(self,x,theta):
        earth=ca.DM([0,SPIN*np.cos(np.radians(37.78)),SPIN*np.sin(np.radians(37.78))])
        tw=self.up*ca.dot(self.up,theta);tilt=theta-tw
        S=quat_to_rotmat(so3_exp(tilt));D=quat_to_rotmat(quat_mul(so3_exp(tw),so3_exp(tilt)))
        sensor=quat_to_rotmat(x[_ambient(self.orientation)])@self.mount
        B=sensor.T@S.T@_so3_reset_jacobian(-tw,symbolic=True)@sensor
        inv=sensor.T@_left_inverse(-tw)@S@sensor
        return sensor.T@(earth-D.T@earth)-B@sensor.T@ca.cross(theta,earth),inv
    def _plus(self,x,d):
        out=super()._plus(x,d)
        off,_=self.earth_offset(x,d[_tangent(self.orientation)])
        for slot in self.biases:
            if slot.name.endswith('gyro_bias'):out[_ambient(slot)]+=off
        return out
    def _minus(self,a,b):
        out=super()._minus(a,b)
        off,inv=self.earth_offset(b,out[_tangent(self.orientation)])
        for slot in self.biases:
            if slot.name.endswith('gyro_bias'):out[_tangent(slot)]-=inv@off
        return out
chart_module.INSStateSpec=Chart
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--seeds',type=int,default=16);p.add_argument('--mounted',action='store_true');p.add_argument('--noisy',action='store_true');p.add_argument('--seed',type=int,default=60191)
a=p.parse_args()
from examples.qualification.earth_ins_split import run
r=run(covariance='nonlinear',mounted=a.mounted,seeds=a.seeds,seed=a.seed,
      gyro_density=.001 if a.noisy else 1e-7,bias_sigma=.001 if a.noisy else 1e-5)
a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r['records'][-1]),flush=True)
