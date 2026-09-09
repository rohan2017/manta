import casadi as ca, numpy as np
from examples.qualification.earth_ins import build,prior
from manta.codegen.numpy._compile import compile_functions
ins=build(covariance="nonlinear");m=ins.module();s=ins.spec
x0=np.asarray(m.port("prior_x").init); p0=prior(ins,1e-5)
x0,p0=m.functions["initialize_prior"](x0,p0);u=ins.sys.u_defaults.copy()
u[ins.sys._input_slices[ins.sys.accel_input]]=[0,0,9.81]
u[ins.sys._input_slices[ins.sys.gyro_input]]=ins.navigation_frame.angular_velocity
x=ca.MX.sym("x",16);p=ca.MX.sym("p",15,15)
pred=m.functions["predict"](x,p,u,.01,0)
up=m.functions["update_craft_dvl_velocity"](*pred,[0,0,0],u,0)
f=ca.Function("fixed_probe",[x,p],list(up));f=compile_functions({"probe":f},max_instructions=50000,optimization="O1")["probe"]
for fixed in (True,False):
 x,p=x0,p0
 for k in range(30000):
  x,p=f(x,p)
  if fixed:x=x0
  if k+1 in (100,1000,10000,30000):print(fixed,k+1,np.degrees(np.sqrt(float(p[5,5]))),np.asarray(x).ravel()[10:],flush=True)
