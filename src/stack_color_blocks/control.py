"""Named position control for UR5e and Robotiq 2F-85."""
import numpy as np
ARM_JOINTS=('shoulder_pan_joint','shoulder_lift_joint','elbow_joint','wrist_1_joint','wrist_2_joint','wrist_3_joint')
ARM_ACTUATORS=('shoulder_pan','shoulder_lift','elbow','wrist_1','wrist_2','wrist_3')
GRIP_JOINTS=('grip_right_driver_joint','grip_left_driver_joint')
GRIP_ACTUATOR='grip_fingers_actuator'
# Demonstration reset posture, in ARM_JOINTS order (radians).
HOME_QPOS=(0.0,-1.8,-1.9,-1.5,1.57,0.0)
# 0=open, 255=closed; interpolate in simulation time at <= 0.6 rad/s.
MAX_SPEED=0.6

def targets(motion, t, home):
    q=np.array(home,dtype=float)
    grip=0.0
    if motion=='joint_sweep':
        # Smooth cycles stay away from joint limits and from the table.
        q[0]+=0.25*np.sin(2*np.pi*t/3.0)
        q[5]+=0.22*np.sin(2*np.pi*t/2.5)
    elif motion=='gripper_cycle':
        grip=255.0 if 0.4<=t<1.0 else 0.0
    elif motion!='hold_home':
        raise ValueError(motion)
    return q,grip

class Controller:
    def __init__(self,model,home):
        self.model=model;self.home=np.asarray(home,dtype=float)
        self.actuators=[model.get_actuator(n) for n in ARM_ACTUATORS]
        self.gripper=model.get_actuator(GRIP_ACTUATOR)
        assert all(a is not None for a in self.actuators) and self.gripper is not None
        self.command=self.home.copy()
    def apply(self,data,motion=None,t=0.0,dt=0.0,arm_qpos=None,gripper_target=None):
        """Apply either a legacy named motion or explicit arm/gripper targets.

        Explicit commands use six UR5e joint positions in radians and a 0/open
        to 255/closed Robotiq command. Arm motion remains rate limited.
        """
        if arm_qpos is None and gripper_target is None:
            target,grip=targets(motion or 'hold_home',t,self.home)
        else:
            if arm_qpos is None or gripper_target is None:
                raise ValueError('arm_qpos and gripper_target must be provided together')
            target=np.asarray(arm_qpos,dtype=float)
            if target.shape!=(6,) or not np.all(np.isfinite(target)):
                raise ValueError('arm_qpos must contain six finite joint targets')
            grip=float(gripper_target)
            if not np.isfinite(grip) or not 0.0<=grip<=255.0:
                raise ValueError('gripper_target must be between 0 (open) and 255 (closed)')
        if dt < 0:raise ValueError('dt must be non-negative')
        self.command+=np.clip(target-self.command,-MAX_SPEED*dt,MAX_SPEED*dt)
        for actuator,value in zip(self.actuators,self.command):actuator.set_ctrl(data,float(value))
        self.gripper.set_ctrl(data,float(grip))
        return self.command.copy(),grip
