"""Reference: estimate the current gyro-boundary error instead of freezing its mean."""
import argparse,json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import casadi as ca
import numpy as np
from manta.ir.state_spec import StateSpec
from manta.ir.manifold import R3Manifold
from manta.ir.module import StateField,StateLayout,Port,Role
from manta.estimation._ins_error import INSStateSpec
from manta.estimation._ins_moments import psd_root,prior_moments_function
from manta.estimation._kalman import unscented_weights,ut_predict,joseph_update
from manta.ir._linalg import spd_solve
from examples.qualification import earth_ins_split as fixture
original_build=fixture.build;original_prior=fixture.prior

def build(**kwargs):
    ir=original_build(**kwargs);sys=ir.sys;old_spec=ir.spec;module=ir.module()
    physical=StateSpec.from_layout([(s.name,s.manifold) for s in old_spec.product_spec.slots]+[('craft.imu.gyro_boundary_error',R3Manifold())])
    spec=INSStateSpec(physical,craft=sys.craft_name,imu=sys.imu_name,
                      rotation_body_from_imu=sys.R_craft_from_sensor,
                      reference_specific_force=old_spec.reference_specific_force)
    n=spec.tangent_dim;na=spec.ambient_dim;old_n=old_spec.tangent_dim;old_na=old_spec.ambient_dim
    x=ca.MX.sym('x',na);P=ca.MX.sym('P',n,n);C=ca.MX.sym('unused_cross',n,3)
    u,dt,t=sys.u_sym,sys.dt_sym,sys.t_sym
    wm,wc,gamma=unscented_weights(n+12,1.,2.,0.)
    L=psd_root(P);Lc=psd_root(sys.boundary_conditional_covariance_sym)
    points=[(ca.MX.zeros(n),ca.MX.zeros(12))]+[(sign*gamma*L[:,i],ca.MX.zeros(12)) for sign in [1,-1] for i in range(n)]+[(ca.MX.zeros(n),sign*gamma*Lc[:,i]) for sign in [1,-1] for i in range(12)]
    propagated=[]
    for d,residual in points:
        state=spec.boxplus_sym(x,d);start=state[old_na:]
        noise=sys.boundary_conditional_gain_sym@start+residual
        phys=sys.packet_noisy_fn(state[:old_na],u,dt,t,ca.MX.zeros(sys.n_sym.numel()),noise[:9],start,noise[9:])
        propagated.append(ca.vertcat(phys,noise[9:]))
    xp,pp=ut_predict(propagated,ca.MX.zeros(n,n),wm,wc,spec,3)
    pred=ca.Function('boundary_state_predict',[x,P,C,u,dt,t],[xp,pp,ca.MX.zeros(n,3)])
    Q=ca.MX.sym('Q',n,n)
    predq=ca.Function('boundary_state_predict_Q',[x,P,C,Q,u,dt,t],[xp,pp+Q,ca.MX.zeros(n,3)])
    h=ca.substitute(sys.sensors['craft.dvl.velocity'].h_sym,sys.x_sym,x[:old_na])
    h=ca.substitute(h,dt,ca.MX(0))
    d=ca.MX.sym('d',n)
    H=ca.substitute(ca.jacobian(ca.substitute(h,x,spec.boxplus_sym(x,d)),d),d,ca.MX.zeros(n))
    z=ca.MX.sym('z',3);R=ca.DM.eye(3)*1e-6
    xn,pn,nu,S=joseph_update(x,P,h,H,R,z,spec)
    outs=[xn,pn,ca.MX.zeros(n,3)]
    update=ca.Function('boundary_state_update',[x,P,C,z,u,t],outs+[nu,S,ca.dot(nu,spd_solve(S,nu)),ca.MX(1)])
    short=ca.Function('boundary_state_short',[x,P,C,z,u,t],outs)
    Roverride=ca.MX.sym('R',3,3)
    xo,po,no,So=joseph_update(x,P,h,H,Roverride,z,spec)
    override=ca.Function('boundary_state_override',[x,P,C,z,Roverride,u,t],[xo,po,ca.MX.zeros(n,3),no,So,ca.dot(no,spd_solve(So,no)),ca.MX(1)])
    x0=np.concatenate([np.asarray(module.port('prior_x').init),np.zeros(3)])
    p0=np.eye(n)*.01;p0[-3:,-3:]=np.eye(3)
    initializer=prior_moments_function(spec,3)
    mean,cov,cross=initializer(x0,p0)
    state=StateLayout((StateField('x','manifold',(na,),init=np.asarray(mean).ravel(),spec=spec),StateField('P','matrix',(n,n),init=np.asarray(cov)),StateField('P_consider','matrix',(n,3),init=np.asarray(cross))))
    ports=[]
    for port in module.ports:
        if port.name=='prior_x':port=Port('prior_x',Role.STATE,(na,),spec=physical,init=x0)
        elif port.name=='prior_P':port=Port('prior_P',Role.MATRIX,(n,n),init=p0)
        elif port.name=='Q':port=Port('Q',Role.MATRIX,(n,n))
        ports.append(port)
    functions=dict(module.functions)
    functions.update(predict=pred,predict_with_Q=predq,initialize_prior=initializer,
                     update_craft_dvl_velocity=short,update_diagnostic_craft_dvl_velocity=update,update_with_R_craft_dvl_velocity=override)
    ir._module=replace(module,state=state,ports=tuple(ports),functions=functions,metadata={**module.metadata,'boundary_reference':'estimated_noise_state'})
    ir.original_spec=old_spec;ir.spec=spec
    return ir

def prior(ir,bias):
    p=original_prior(SimpleNamespace(spec=ir.original_spec),bias)
    out=np.eye(p.shape[0]+3);out[:p.shape[0],:p.shape[0]]=p
    return out
fixture.build=build;fixture.prior=prior
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--mounted',action='store_true');p.add_argument('--duration',type=float,default=300)
a=p.parse_args()
r=fixture.run(covariance='nonlinear',gyro_density=.001,bias_sigma=.001,mounted=a.mounted,duration=a.duration)
r['boundary_mean_model']='estimated jointly with physical state'
a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'acceptance':r['acceptance'],**r['records'][-1]}),flush=True)
