"""Independent nonlinear reference using INS mechanics and augmented noise."""
import argparse, json, math, os
from pathlib import Path
from dataclasses import replace
import casadi as ca
import numpy as np
from manta.estimation._kalman import sigma_deltas, unscented_weights, ut_predict, ut_update, _reset_jacobian, symmetrize
from manta.ir._linalg import spd_solve
from examples.qualification import earth_ins as qualification

original_build=qualification.build

def build(*args,**kwargs):
    ir=original_build(*args,**kwargs);sys=ir.sys;spec=ir.spec
    if os.environ.get('INS_CHART')=='gravity_bias':
        from gravity_bias_chart import INSStateSpec
        spec=INSStateSpec(spec,craft=sys.craft_name,imu=sys.imu_name,rotation_body_from_imu=sys.R_craft_from_sensor)
        sys.spec=spec
    elif os.environ.get('INS_CHART')=='aligned':
        from gravity_aligned_chart import INSStateSpec
        spec=INSStateSpec(spec,craft=sys.craft_name,imu=sys.imu_name,rotation_body_from_imu=sys.R_craft_from_sensor)
        sys.spec=spec
    elif os.environ.get('INS_CHART')=='semidirect':
        from semidirect_chart import INSStateSpec
        spec=INSStateSpec(spec,craft=sys.craft_name,imu=sys.imu_name,rotation_body_from_imu=sys.R_craft_from_sensor)
        sys.spec=spec
    elif os.environ.get('INS_CHART')=='twoframe':
        from geometric_chart import INSStateSpec
        spec=INSStateSpec(spec,craft=sys.craft_name,imu=sys.imu_name,rotation_body_from_imu=sys.R_craft_from_sensor)
        sys.spec=spec
    ir.spec=spec
    spec.error_model='augmented_ukf_'+os.environ.get('INS_CHART','product')+'_'+os.environ.get('INS_RESET','differential')
    module=ir.module();x,u,dt,t=sys.x_sym,sys.u_sym,sys.dt_sym,sys.t_sym
    n=spec.tangent_dim;P=ca.MX.sym('P',n,n)
    active=np.flatnonzero(np.any(np.asarray(ca.DM(sys.L_sym.sparsity())),axis=0))
    m=len(active);total=n+m
    wm,wc,gamma=unscented_weights(total,1.,2.,0.)
    ds=sigma_deltas(P,gamma,n)
    noisy_fn=ca.Function('noisy_mechanics',[x,u,sys.n_sym,dt,t],[sys.x_new_noisy])
    noise0=ca.MX.zeros(sys.n_sym.numel())
    propagated=[noisy_fn(spec.boxplus_sym(x,d),u,noise0,dt,t) for d in ds]
    for sign in (1.,-1.):
        for i in active:
            noise=ca.MX(noise0);noise[int(i)]=sign*gamma*math.sqrt(sys.Sigma[i,i])
            propagated.append(noisy_fn(x,u,noise,dt,t))
    xp,pp=ut_predict(propagated,ca.MX.zeros(n,n),wm,wc,spec,3)
    predict=ca.Function('augmented_predict',[x,P,u,dt,t],[xp,pp])
    wmu,wcu,gu=unscented_weights(n,1.,2.,0.)
    deltas=sigma_deltas(P,gu,n)
    h=sys.sensors['craft.dvl.velocity'].h_sym
    h=ca.substitute(h,dt,ca.MX(0))
    measured=[ca.substitute(h,x,spec.boxplus_sym(x,d)) for d in deltas]
    z=ca.MX.sym('z',3);R=ca.DM.eye(3)*1e-6
    xn,pn,nu,S=ut_update(x,P,deltas,measured,R,z,wmu,wcu,spec)
    d=spec.boxminus_sym(xn,x)
    if os.environ.get('INS_RESET')=='moments':
        posterior=[spec.boxplus_sym(x,d+epsilon) for epsilon in sigma_deltas(pn,gu,n)]
        xn,pn=ut_predict(posterior,ca.MX.zeros(n,n),wmu,wcu,spec,3)
    else:
        reset=spec.exact_reset(x,d) if hasattr(spec,'exact_reset') else _reset_jacobian(spec,d)
        pn=symmetrize(reset@pn@reset.T)
    update=ca.Function('augmented_update',[x,P,z,u,t],[xn,pn,nu,S,ca.dot(nu,spd_solve(S,nu)),ca.MX(1)])
    functions=dict(module.functions)
    functions['predict']=predict
    functions['update_diagnostic_craft_dvl_velocity']=update
    ir._module=replace(module,functions=functions,metadata={**module.metadata,'reference_filter':spec.error_model})
    return ir
qualification.build=build
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--duration',type=float,default=60);p.add_argument('--bias-sigma',type=float,default=.001);p.add_argument('--gyro-density',type=float,default=.001);p.add_argument('--seeds',type=int,default=16);p.add_argument('--seed',type=int,default=82719);p.add_argument('--motion',action='store_true');p.add_argument('--spin',type=float,default=qualification.SPIN)
a=p.parse_args()
r=qualification.run(duration=a.duration,bias_sigma=a.bias_sigma,gyro_density=a.gyro_density,seeds=a.seeds,seed=a.seed,motion=a.motion,spin=a.spin)
a.output.write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({'acceptance':r['acceptance'],**r['records'][-1]}),flush=True)
