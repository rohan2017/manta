"""Control: condition in one finite chart instead of recentering every update."""
import argparse,json
from pathlib import Path
from dataclasses import replace
import casadi as ca
import numpy as np
import manta.estimation._ins_error as error_module
import manta.estimation._ins_moments as moment_module
from manta.ir._rotation import so3_exp

Base=error_module.INSStateSpec
class Fixed(Base):
    error_model='fixed_reference_chart_experiment'
    def __init__(self,product,**kwargs):
        super().__init__(product,**kwargs)
        self.local=Base(product,**kwargs)
        self.anchor=ca.MX.zeros(self.ambient_dim)
        self.anchor[3:7]=so3_exp(ca.DM([.2,-.15,.7]))
    def _plus(self,x,d):
        return self.local.boxplus_sym(self.anchor,self.local.boxminus_sym(x,self.anchor)+d)
    def _minus(self,a,b):
        return self.local.boxminus_sym(a,self.anchor)-self.local.boxminus_sym(b,self.anchor)
error_module.INSStateSpec=Fixed
original=moment_module.predict_moments

def prediction(sys,spec,P,C,**kwargs):
    # The fixed chart's differential at a moving nominal is not the physical
    # product tangent. Differentiate measurements in the actual coordinates.
    x=sys.x_sym;d=ca.MX.sym('fixed_delta',spec.tangent_dim)
    for name,sm in list(sys.sensors.items()):
        H=ca.substitute(ca.jacobian(ca.substitute(sm.h_sym,x,spec.boxplus_sym(x,d)),d),d,ca.MX.zeros(spec.tangent_dim))
        sys.sensors[name]=replace(sm,H_sym=H,H_fn=ca.Function('fixed_H',[x,sys.u_sym,sys.dt_sym,sys.t_sym],[H]))
    return original(sys,spec,P,C,**kwargs)
moment_module.predict_moments=prediction
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--seeds',type=int,default=64);p.add_argument('--noisy',action='store_true');p.add_argument('--seed',type=int,default=60191)
a=p.parse_args()
from examples.qualification.earth_ins_split import run
r=run(covariance='nonlinear',mounted=True,seeds=a.seeds,seed=a.seed,
      gyro_density=.001 if a.noisy else 1e-7,bias_sigma=.001 if a.noisy else 1e-5)
a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r['records'][-1]),flush=True)
