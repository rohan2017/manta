import json
import casadi as ca
import numpy as np
from gravity_aligned_chart import INSStateSpec
from examples.qualification.earth_ins import build
from manta.ir._rotation import quat_to_rotmat,so3_exp
rng=np.random.default_rng(39105)
ins=build();product=ins.spec
mount=quat_to_rotmat(so3_exp(ca.DM([.2,-.4,.7])))
spec=INSStateSpec(product,craft='craft',imu='craft.imu',rotation_body_from_imu=mount)
x0=np.asarray(ins.module().state.field('x').init)
x=product.boxplus_num(x0,rng.normal(size=15)*.3)
d=rng.normal(size=15)*.15
shift=spec.boxplus_num(x,d)
err=np.asarray(spec.boxminus_sym(shift,x)).ravel()
eps=1e-6;eye=np.eye(15)
num=np.column_stack([(np.asarray(spec.boxminus_sym(spec.boxplus_num(x,d+eps*e),shift)).ravel()-np.asarray(spec.boxminus_sym(spec.boxplus_num(x,d-eps*e),shift)).ravel())/(2*eps) for e in eye])
local=np.column_stack([(spec.boxplus_num(x,eps*e)-spec.boxplus_num(x,-eps*e))/(2*eps) for e in eye])
physical=np.column_stack([(product.boxplus_num(x,eps*e)-product.boxplus_num(x,-eps*e))/(2*eps) for e in eye])
R=np.asarray(quat_to_rotmat(x[3:7]));Rp=np.asarray(quat_to_rotmat(shift[3:7]))
dvl=Rp.T@shift[7:10]-R.T@x[7:10]-R.T@(d[6:9]-np.cross(d[3:6],x[7:10]))
# Pure global yaw changes only the explicit yaw error, even after arbitrary tilt reset.
base=spec.boxplus_num(x,d)
q=quat_to_rotmat(so3_exp(ca.DM([0,0,.2])))
a=base.copy();a[3:7]=np.asarray(so3_exp(ca.DM([0,0,.2]))) .ravel() # overwritten below
from manta.ir._rotation import quat_mul
a[3:7]=np.asarray(quat_mul(so3_exp(ca.DM([0,0,.2])),ca.DM(base[3:7]))).ravel()
for sl in [slice(0,3),slice(7,10)]:a[sl]=np.asarray(q@base[sl]).ravel()
expected=np.zeros(15);expected[5]=.2
# Navigation gauge at nonzero p,v is not simply a yaw-only tangent.
expected[0:3]=np.cross([0,0,.2],base[:3]);expected[6:9]=np.cross([0,0,.2],base[7:10])
gauge=np.asarray(spec.boxminus_sym(a,base)).ravel()-expected
results={'roundtrip':float(np.max(np.abs(err-d))),'reset_derivative':float(np.max(np.abs(num-np.asarray(spec.exact_reset(x,d))))),'local_tangent':float(np.max(np.abs(local-physical))),'finite_body_velocity_residual':float(np.max(np.abs(dvl))),'finite_global_yaw_gauge':float(np.max(np.abs(gauge)))}
print(json.dumps(results,indent=2));assert max(results.values())<1e-8
