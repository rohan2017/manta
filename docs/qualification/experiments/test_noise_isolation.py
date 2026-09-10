"""Remove one truth-noise stream, leaving its declared covariance conservative."""
import inspect,argparse,json
from pathlib import Path
import examples.qualification.earth_ins_split as m
p=argparse.ArgumentParser();p.add_argument('--stream',choices=['dvl','gyro','accel'],required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
source=inspect.getsource(m.run)
target={'dvl':'d_rng.normal(size=(3, seeds)) * 0.001',
        'gyro':'g_rng.normal(size=(3, seeds)) * gyro_density / np.sqrt(dt)',
        'accel':'a_rng.normal(size=(3, seeds)) * 1e-5 / np.sqrt(dt)'}[a.stream]
assert source.count(target)==1
source=source.replace(target,'np.zeros((3,seeds))')
context=dict(vars(m));exec(compile(source,'noise_control','exec'),context)
r=context['run'](covariance='nonlinear',mounted=True,seeds=16,seed=60191)
r['control']='truth '+a.stream+' noise removed; original covariance retained'
a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r['records'][-1]),flush=True)
