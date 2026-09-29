"""One-command acceptance check for assets, dynamics, controls and cameras."""
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from stack_color_blocks import StackColorBlocksEnv
from stack_color_blocks.control import ARM_JOINTS,ARM_ACTUATORS,GRIP_JOINTS,GRIP_ACTUATOR
from motrixsim import load_model,SceneData,step
from build_scene import build
from lighting_randomization import capture_ready

def check():
    build()
    for rel in ('universal_robots_ur5e/ur5e.xml','robotiq_2f85/2f85.xml'):
        path=ROOT/'assets/vendor/menagerie'/rel
        assert path.exists(),f'missing {path}; run scripts/fetch_assets.sh'
        tree=ET.parse(path)
        for mesh in tree.findall('./asset/mesh'):
            assert (path.parent/'assets'/mesh.get('file')).exists(),f'missing mesh {mesh.get("file")}'
        model=load_model(str(path));data=SceneData(model)
        for _ in range(100):step(model,data)
    env=StackColorBlocksEnv(render=False)
    try:
        assert len(env.model.actuators)==7 and len(env.model.joints)==14
        assert all(env.model.get_joint(n) for n in ARM_JOINTS+GRIP_JOINTS)
        assert all(env.model.get_actuator(n) for n in ARM_ACTUATORS+(GRIP_ACTUATOR,))
        scene=ET.parse(ROOT/'assets/stack_color_blocks.xml')
        base=next(body for body in scene.findall('./worldbody/body') if body.get('name')=='base')
        table_cfg=env.config['table'];offset=env.config['robot']['base_from_table_top']
        expected_base=np.asarray(table_cfg['center'])+np.asarray([offset[0],offset[1],table_cfg['half_size'][2]+offset[2]])
        assert np.allclose(np.fromstring(base.get('pos'),sep=' '),expected_base),'robot base is not table-relative'
        for name in ('chair','desk','plant'):
            assert env.model.get_body('background_'+name).mocap is not None,f'{name} cannot be repositioned'
        sticker_geoms={g.get('name'):g for g in scene.findall('./worldbody/geom')}
        materials={m.get('name'):m for m in scene.findall('./asset/material')}
        def check_material(name, spec, rgba=None):
            material=materials[name]
            assert all(np.isclose(float(material.get(key)), value) for key,value in spec.items()),name
            if rgba is not None:
                assert np.allclose(np.fromstring(material.get('rgba'),sep=' '),rgba),name
        assert sticker_geoms['table'].get('material')=='table_material'
        check_material('table_material',env.config['pbr']['table'],table_cfg['rgba'])
        for color,rgba in {'yellow':[.95,.78,.04,1],'orange':[.98,.31,.03,1],
                           'sky_blue':[.32,.72,.95,1]}.items():
            geom=scene.find(f".//geom[@name='{color}_geom']")
            assert geom.get('material')==color+'_block_material'
            check_material(color+'_block_material',env.config['pbr']['blocks'],rgba)
        for name,spec in env.config['pbr']['ur5e'].items():check_material(name,spec)
        for name,spec in env.config['pbr']['gripper'].items():check_material('grip_'+name,spec)
        for side in ('left','right'):
            pad=scene.find(f".//body[@name='grip_{side}_silicone_pad']/geom[@class='grip_visual']")
            assert pad is not None and pad.get('material')=='grip_silicone'
        assert not scene.findall('.//material[@shininess]') and not scene.findall('.//material[@specular]')
        assert sticker_geoms['floor'].get('rgba')==' '.join(str(x) for x in env.config['floor']['rgba'])
        for name in ('wall_x_pos','wall_x_neg','wall_y_pos','wall_y_neg'):
            geom=sticker_geoms[name]
            assert geom.get('contype')=='0' and geom.get('conaffinity')=='0',name
        wrist=scene.find(".//camera[@name='wrist_rgb']")
        assert np.allclose(np.fromstring(wrist.get('pos'),sep=' '),env.config['wrist_camera']['pos'])
        assert np.allclose(np.fromstring(wrist.get('quat'),sep=' '),env.config['wrist_camera']['quat'])
        assert env.config['stickers']['items'][0]['name']=='green' and env.config['stickers']['items'][2]['name']=='blue'
        sticker_by_name={item['name']:item for item in env.config['stickers']['items']}
        assert sticker_by_name['green']['center'][0]<sticker_by_name['blue']['center'][0],'green/blue near/far sides reversed'
        for sticker in env.config['stickers']['items']:
            geom=sticker_geoms['sticker_'+sticker['name']]
            assert geom.get('contype')=='0' and geom.get('conaffinity')=='0',f'{sticker["name"]} sticker has collision'
            z=float(geom.get('pos').split()[2]);half=float(geom.get('size').split()[2])
            table_top=env.config['table']['center'][2]+env.config['table']['half_size'][2]
            assert abs(z+half-(table_top+env.config['stickers']['surface_offset']))<1e-9,f'{sticker["name"]} top offset is wrong'
        for name,prop in env.config['background'].items():
            if name=='note':continue
            for part in prop.get('parts',[{'mesh':prop['mesh']}]):
                assert (ROOT/'assets'/part['mesh']).exists(),f'missing {name} mesh {part["mesh"]}'
        a=env.reset(7);b=env.reset(7);c=env.reset(8)
        table_xy=np.asarray(table_cfg['center'][:2])
        for name,pose in a['blocks'].items():
            local=pose[:2]-table_xy
            assert env.config['block']['x_range'][0]<=local[0]<=env.config['block']['x_range'][1],f'{name} x not table-relative'
            assert env.config['block']['y_range'][0]<=local[1]<=env.config['block']['y_range'][1],f'{name} y not table-relative'
            assert abs(pose[2]-(table_cfg['center'][2]+table_cfg['half_size'][2]+env.config['block']['half_size']+env.config['block']['clearance_above_table']))<0.003,f'{name} not initially above table'
            yaw_deg=np.rad2deg(2*np.arctan2(pose[5],pose[6]))
            lo,hi=env.config['block']['yaw_range_deg']
            assert lo-0.1<=yaw_deg<=hi+0.1,f'{name} yaw outside configured range: {yaw_deg}'
            assert np.linalg.norm(pose[3:5])<1e-3,f'{name} block tilted at reset'
        assert all(np.allclose(a['blocks'][n],b['blocks'][n]) for n in ('yellow','orange','sky_blue')),'reset not deterministic'
        assert any(not np.allclose(a['blocks'][n],c['blocks'][n]) for n in ('yellow','orange','sky_blue')),'seeds did not vary'
        assert any(not np.allclose(a['blocks'][n][3:],c['blocks'][n][3:]) for n in ('yellow','orange','sky_blue')),'block yaw did not vary with seed'
        fixed_a=env.reset(7,randomize_blocks=False)['blocks']
        fixed_b=env.reset(8,randomize_blocks=False)['blocks']
        assert all(np.allclose(fixed_a[n],fixed_b[n]) for n in ('yellow','orange','sky_blue')),'fixed block layout changed with seed'
        assert all(np.allclose(fixed_a[n][3:],[0,0,0,1],atol=1e-3) for n in ('yellow','orange','sky_blue')),'fixed block yaw is not zero'
        positions=np.array([a['blocks'][n][:2] for n in ('yellow','orange','sky_blue')])
        assert all(np.linalg.norm(positions[i]-positions[j])>=env.config['block']['min_spacing']-0.002 for i in range(3) for j in range(i+1,3)),'blocks overlap'
        ba=env.reset(7,randomize_background=True)['background']
        bb=env.reset(7,randomize_background=True)['background']
        bc=env.reset(8,randomize_background=True)['background']
        assert all(np.allclose(ba[n],bb[n]) for n in ('chair','desk','plant')),'background reset not deterministic'
        assert any(not np.allclose(ba[n],bc[n]) for n in ('chair','desk','plant')),'background seeds did not vary'
        for name,pose in ba.items():
            spec=env.config['background'][name]
            assert np.all(np.abs(pose[:3]-np.asarray(spec['base_pos']))<=np.asarray(spec['position_jitter'])+1e-5),f'{name} background jitter exceeded range'
        env.reset(7)
        for _ in range(1000):o=env.step()
        assert np.isfinite(o['arm_qpos']).all() and all(np.isfinite(p).all() for p in o['blocks'].values())
        table_top=env.config['table']['center'][2]+env.config['table']['half_size'][2]
        block_half=env.config['block']['half_size']
        assert all(abs(o['blocks'][n][2]-(table_top+block_half))<0.008 for n in ('yellow','orange','sky_blue')),'block penetration/drift'
        env.reset(0);start=env.get_observation()['arm_qpos'].copy();trace=[]
        for i in range(1500):
            o=env.step({'motion':'joint_sweep','time':i*env.dt});trace.append(o['arm_qpos'].copy())
        trace=np.asarray(trace)
        assert np.ptp(trace[:,0])>0.15 and np.ptp(trace[:,5])>0.10,'arm joint sweep did not move'
        env.reset(0);grip=[]
        for i in range(1500):
            o=env.step({'motion':'gripper_cycle','time':i*env.dt});grip.append(o['gripper_qpos'].copy())
        grip=np.asarray(grip)
        assert np.max(grip[:,0])>0.2 and np.max(grip[:,1])>0.2,'gripper did not close'
        assert abs(grip[-1,0])<0.1 and abs(grip[-1,1])<0.1,'gripper did not reopen'
    finally:env.close()
    env=StackColorBlocksEnv(render=True)
    try:
        for _ in range(130):env.step()
        im=capture_ready(env)
        for camera_name, config_name in (('global_rgb', 'global_camera'),
                                         ('wrist_rgb', 'wrist_camera')):
            width, height = env.config[config_name]['resolution']
            assert im[camera_name].shape == (height, width, 3) and im[camera_name].std() > 5, \
                f'{camera_name} blank or wrong dimensions'
        assert np.abs(im['global_rgb'].astype(float)-im['wrist_rgb'].astype(float)).mean()>1,'cameras identical'
    finally:env.close()
    print('PASS scene: assets, stickers, background placement, 14 joints, 7 actuators, reset, 1000-step stability, arm, gripper, two RGB cameras')
if __name__=='__main__':check()
