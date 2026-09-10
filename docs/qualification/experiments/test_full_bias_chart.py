"""Test full-rotation gyro-bias integration while retaining affine gravity."""
import argparse,json
from pathlib import Path
import manta.estimation._ins_error as module
s=Path(module.__file__).read_text()
s=s.replace('S.T @ _so3_reset_jacobian(-twist_vector, symbolic=True)', '_so3_reset_jacobian(-so3_log(Dq), symbolic=True)')
s=s.replace('_left_inverse(-twist_vector) @ S', '_left_inverse(-so3_log(q))')
context={'__package__':'manta.estimation'}
exec(compile(s,'full_bias_chart','exec'),context)
module.INSStateSpec=context['INSStateSpec']
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--seeds',type=int,default=64);p.add_argument('--noisy',action='store_true');p.add_argument('--seed',type=int,default=60191)
a=p.parse_args()
from examples.qualification.earth_ins_split import run
r=run(covariance='nonlinear',mounted=True,seeds=a.seeds,seed=a.seed,
      gyro_density=.001 if a.noisy else 1e-7,bias_sigma=.001 if a.noisy else 1e-5)
a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r['records'][-1]),flush=True)
