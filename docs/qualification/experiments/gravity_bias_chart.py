"""Local swing/twist and body-velocity error coordinates, with exact reset.

Twist about navigation z is an explicit coordinate; gravity/DVL corrections
cannot curve that coordinate into tilt. Bias increments rotate with the swing.
All zero-error differentials retain the physical product tangent units.
This is an isolated experiment, not a production estimator.
"""
from functools import cached_property
import casadi as ca
import numpy as np
from manta.ir.state_spec import StateSpec
from manta.estimation._kalman import _so3_reset_jacobian
from geometric_chart import _left_inverse
from manta.ir._rotation import quat_mul,quat_conj,quat_to_rotmat,so3_exp,so3_log

class INSStateSpec(StateSpec):
    error_model='swing_twist_gravity_bias_exact_reset'
    def __init__(self,product,*,craft,imu,rotation_body_from_imu,earth=(0,0,0)):
        super().__init__(list(product.slots));self.product_spec=product
        self.orientation=self.slot(f'{craft}.orientation')
        self.navigation_vectors=[self.slot(f'{craft}.{s}') for s in ('position','velocity')]
        self.biases=[self.slot(f'{imu}.{s}') for s in ('gyro_bias','accel_bias') if f'{imu}.{s}' in self]
        self.mount=ca.DM(rotation_body_from_imu);self.earth=ca.DM(earth)
    @staticmethod
    def a(s):return slice(s.ambient_offset,s.ambient_offset+s.ambient_dim)
    @staticmethod
    def t(s):return slice(s.tangent_offset,s.tangent_offset+s.tangent_dim)
    def plus_impl(self,x,d):
        out=self.product_spec.boxplus_sym(x,d)
        theta=d[self.t(self.orientation)]
        tilt=ca.vertcat(theta[0],theta[1],0)
        swing=so3_exp(tilt);twist=so3_exp(ca.vertcat(0,0,theta[2]))
        Dq=quat_mul(twist,swing);D=quat_to_rotmat(Dq)
        out[self.a(self.orientation)]=quat_mul(Dq,x[self.a(self.orientation)])
        for slot in self.navigation_vectors:
            value=x[self.a(slot)]
            out[self.a(slot)]=D@(value+d[self.t(slot)]-ca.cross(theta,value))
        sensor=quat_to_rotmat(x[self.a(self.orientation)])@self.mount
        B=sensor.T@quat_to_rotmat(swing).T@_so3_reset_jacobian(ca.vertcat(0,0,-theta[2]),symbolic=True)@sensor
        for slot in self.biases:
            bias_map = sensor.T@_so3_reset_jacobian(-tilt,symbolic=True)@sensor if slot.name.endswith('accel_bias') else B
            out[self.a(slot)]=x[self.a(slot)]+bias_map@d[self.t(slot)]
        return out
    def minus_impl(self,a,b):
        out=self.product_spec.boxminus_sym(a,b)
        q=quat_mul(a[self.a(self.orientation)],quat_conj(b[self.a(self.orientation)]))
        q=ca.if_else(q[0]<0,-q,q)
        yaw=2*ca.atan2(q[3],q[0])
        swing=quat_mul(so3_exp(ca.vertcat(0,0,-yaw)),q)
        tilt=so3_log(swing)
        theta=ca.vertcat(tilt[0],tilt[1],yaw)
        out[self.t(self.orientation)]=theta
        D=quat_to_rotmat(q)
        for slot in self.navigation_vectors:
            value=b[self.a(slot)]
            out[self.t(slot)]=D.T@a[self.a(slot)]-value+ca.cross(theta,value)
        sensor=quat_to_rotmat(b[self.a(self.orientation)])@self.mount
        invB=sensor.T@_left_inverse(ca.vertcat(0,0,-yaw))@quat_to_rotmat(swing)@sensor
        for slot in self.biases:
            diff=a[self.a(slot)]-b[self.a(slot)]
            inverse = sensor.T@_left_inverse(-ca.vertcat(tilt[0],tilt[1],0))@sensor if slot.name.endswith('accel_bias') else invB
            out[self.t(slot)]=inverse@diff
        return out
    @cached_property
    def plus(self):
        x=ca.MX.sym('x',self.ambient_dim);d=ca.MX.sym('d',self.tangent_dim)
        return ca.Function('aligned_plus',[x,d],[self.plus_impl(x,d)])
    @cached_property
    def minus(self):
        a=ca.MX.sym('a',self.ambient_dim);b=ca.MX.sym('b',self.ambient_dim)
        return ca.Function('aligned_minus',[a,b],[self.minus_impl(a,b)])
    def boxplus_sym(self,x,d):return self.plus(x,d)
    def boxminus_sym(self,a,b):return self.minus(a,b)
    def boxplus_num(self,x,d):return np.asarray(self.plus(x,d)).ravel()
    @cached_property
    def exact_reset(self):
        x=ca.MX.sym('x',self.ambient_dim);d=ca.MX.sym('d',self.tangent_dim);e=ca.MX.sym('e',self.tangent_dim)
        error=self.minus(self.plus(x,d+e),self.plus(x,d))
        G=ca.substitute(ca.jacobian(error,e),e,ca.MX.zeros(self.tangent_dim))
        return ca.Function('aligned_reset',[x,d],[G])
