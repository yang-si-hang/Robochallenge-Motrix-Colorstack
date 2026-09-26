"""Single-instance MotrixSim tabletop environment."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from motrixsim import SceneData,load_model,step
from motrixsim.render import RenderApp
from motrixsim.render import RenderSettings
from .control import ARM_JOINTS,GRIP_JOINTS,HOME_QPOS,Controller
ROOT=Path(__file__).resolve().parents[2]

def render_settings_for_ssgi(enabled):
    """Use the quality preset consistently, changing only the SSGI switch."""
    quality=RenderSettings.quality()
    return RenderSettings(
        simplify_render_mesh=quality.simplify_render_mesh,
        enable_shadow=quality.enable_shadow,
        enable_ssao=quality.enable_ssao,
        enable_oit=quality.enable_oit,
        share_lights_between_envs=quality.share_lights_between_envs,
        enable_ssgi=bool(enabled),
    )

class StackColorBlocksEnv:
    def __init__(self,render=False,headless=True,width=None,height=None,render_hz=30,config=None,ssgi_enabled=None):
        self.config=json.loads((ROOT/'configs/scene.yaml').read_text()) if config is None else config
        self.model=load_model(str(ROOT/'assets/stack_color_blocks.xml'))
        self.data=SceneData(self.model)
        self.dt=self.config['timestep'];self.time=0.0
        self.render_hz=render_hz
        self.ssgi_enabled=self.config['ssgi']['enabled'] if ssgi_enabled is None else bool(ssgi_enabled)
        self._next_render_sync_time=(1.0/render_hz) if render_hz and render_hz>0 else None
        self.controller=Controller(self.model,HOME_QPOS)
        self.render=RenderApp(headless=headless) if render else None
        if self.render:
            camera_configs={'global_rgb':'global_camera','wrist_rgb':'wrist_camera'}
            for camera in self.model.cameras:
                camera_cfg=self.config[camera_configs[camera.name]]
                resolution=camera_cfg.get('resolution',[320,240])
                camera_width=resolution[0] if width is None else width
                camera_height=resolution[1] if height is None else height
                camera.set_render_target('image',int(camera_width),int(camera_height))
            # Match the quality preset while overriding only SSGI. The preset
            # otherwise enables SSGI by default in MotrixSim 0.10.1.
            settings=render_settings_for_ssgi(self.ssgi_enabled)
            self.render.launch(self.model, render_settings=settings)
            for i in range(len(self.model.cameras)):self.render.get_camera(i).active=True
        self.reset(0)
    def reset(self,seed=0,randomize_background=False,randomize_blocks=True):
        self.data.reset(self.model);self.time=0.0
        self.controller.command=np.asarray(HOME_QPOS,dtype=float).copy()
        for name,value in zip(ARM_JOINTS,HOME_QPOS):
            self.model.get_joint(name).set_dof_pos(self.data,np.array([value],dtype=float))
        rng=np.random.default_rng(seed);cfg=self.config['block'];points=[]
        table=self.config['table']
        table_xy=np.asarray(table['center'][:2],dtype=float)
        table_top=table['center'][2]+table['half_size'][2]
        block_z=table_top+cfg['half_size']+cfg['clearance_above_table']
        background_rng=np.random.default_rng(np.random.SeedSequence([seed,1]))
        for name,prop in self.config['background'].items():
            if name == 'note': continue
            pos=np.asarray(prop['base_pos'],dtype=float).copy()
            yaw=np.deg2rad(prop.get('yaw_deg',0.0))
            if randomize_background:
                jitter=np.asarray(prop['position_jitter'],dtype=float)
                pos+=background_rng.uniform(-jitter,jitter)
                yaw+=np.deg2rad(background_rng.uniform(-prop['yaw_jitter_deg'],prop['yaw_jitter_deg']))
            # GLTF assets use Y-up. Rotate 90 degrees about X to make Z-up,
            # then apply an optional world-Z yaw. Mocap poses use xyzw order.
            half_yaw=yaw/2
            c,s=np.cos(half_yaw),np.sin(half_yaw)
            quat=np.array([c/np.sqrt(2),s/np.sqrt(2),s/np.sqrt(2),c/np.sqrt(2)])
            self.model.get_body('background_'+name).mocap.set_pose(self.data,np.r_[pos,quat])
        names=('yellow','orange','sky_blue')
        if randomize_blocks:
            for _ in names:
                for attempt in range(1000):
                    xy=np.array([rng.uniform(*cfg['x_range']),rng.uniform(*cfg['y_range'])])
                    # A circumscribed circle prevents overlap for every sampled yaw.
                    half_size=cfg['half_size']
                    collides=any(np.linalg.norm(xy-p)<2*np.sqrt(2)*half_size for p in points)
                    too_close=any(np.linalg.norm(xy-p)<cfg['min_spacing'] for p in points)
                    if not collides and not too_close:break
                else:raise RuntimeError('Could not sample block layout')
                points.append(xy)
        else:
            points=[np.asarray(cfg['fixed_xy'][name],dtype=float) for name in names]
        table_half_xy=np.asarray(table['half_size'][:2],dtype=float)
        # Keep yaw sampling independent so existing seed-to-XY layouts stay stable.
        yaw_rng=np.random.default_rng(np.random.SeedSequence([seed,3]))
        for name,xy in zip(names,points):
            yaw=np.deg2rad(yaw_rng.uniform(*cfg['yaw_range_deg'])) if randomize_blocks else 0.0
            half_yaw=yaw/2
            quat=[0.0,0.0,np.sin(half_yaw),np.cos(half_yaw)]
            footprint_half_extent=cfg['half_size']*(abs(np.cos(yaw))+abs(np.sin(yaw)))
            if np.any(np.abs(xy)+footprint_half_extent>table_half_xy):
                raise ValueError(f'{name} block position exceeds tabletop bounds: {xy}')
            world_xy=table_xy+xy
            self.model.get_body(name+'_block').set_dof_pos(self.data,np.array([*world_xy,block_z,*quat],float),True)
        self.controller.apply(self.data,'hold_home',0,self.dt)
        step(self.model,self.data)
        if self.render:
            self.render.sync(self.data,wait=True)
            if self.render_hz and self.render_hz>0:self._next_render_sync_time=1.0/self.render_hz
        return self.get_observation(capture=False)
    def step(self,action=None):
        if action is None:action={'motion':'hold_home','time':self.time}
        if isinstance(action,str):action={'motion':action,'time':self.time}
        target,grip=self.controller.apply(
            self.data,action.get('motion','hold_home'),action.get('time',self.time),self.dt,
            arm_qpos=action.get('arm_qpos'),gripper_target=action.get('gripper_target'))
        step(self.model,self.data);self.time+=self.dt
        if self.render:
            if self._next_render_sync_time is not None and self.time+1e-12>=self._next_render_sync_time:
                self.render.sync(self.data,wait=True)
                self._next_render_sync_time+=1.0/self.render_hz
        obs=self.get_observation(capture=False);obs['target']=target;obs['gripper_target']=grip
        return obs
    def capture(self):
        if not self.render:raise RuntimeError('Environment was created without rendering')
        captures=[self.render.get_camera(i).capture() for i in range(2)]
        self.render.sync(self.data,wait=True)
        frames=[]
        for capture in captures:
            if capture.state!='done':raise RuntimeError(f'capture state {capture.state}')
            image=capture.take_image()
            if image is None:raise RuntimeError('empty camera image')
            frames.append(image.pixels[:,:,:3].copy());del image
        del captures
        return {'global_rgb':frames[0],'wrist_rgb':frames[1]}
    def get_observation(self,capture=False):
        obs={'arm_qpos':np.array([self.model.get_joint(n).get_dof_pos(self.data)[0] for n in ARM_JOINTS]),
             'gripper_qpos':np.array([self.model.get_joint(n).get_dof_pos(self.data)[0] for n in GRIP_JOINTS]),
             'blocks':{n:self.model.get_body(n+'_block').get_pose(self.data).copy() for n in ('yellow','orange','sky_blue')},
             'background':{n:self.model.get_body('background_'+n).get_pose(self.data).copy()
                           for n in self.config['background'] if n!='note'},
             'time':self.time}
        if capture:obs.update(self.capture())
        return obs
    def close(self):
        if self.render:
            self.render.__exit__(None,None,None);self.render=None
