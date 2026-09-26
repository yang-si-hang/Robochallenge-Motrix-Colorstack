"""Deterministically assemble vendored MJCF assets and table scene."""
from __future__ import annotations
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / 'assets/vendor/menagerie'
OUTPUT = ROOT / 'assets/stack_color_blocks.xml'
CONFIG = ROOT / 'configs/scene.yaml'
REF_ATTRS = {'body1','body2','joint','joint1','joint2','mesh','material','class','childclass','tendon','target'}

def vec(v): return ' '.join(str(x) for x in v)

def set_pbr(material, params):
    for key in ('reflectance', 'roughness', 'metallic'):
        value = float(params[key])
        if not 0 <= value <= 1:
            raise ValueError(f'{material.get("name")}.{key} must be between 0 and 1')
        material.set(key, str(value))

def validate_ssgi(params):
    required = {'enabled', 'resolutionscale', 'raycount', 'stepcount', 'thickness',
                'intensity', 'gidenoiseoffset', 'rraycount', 'rdenoiseoffset'}
    if set(params) != required:
        raise ValueError(f'ssgi keys must be exactly {sorted(required)}')
    if not isinstance(params['enabled'], bool):
        raise ValueError('ssgi.enabled must be a boolean')
    for key in ('resolutionscale', 'raycount', 'stepcount', 'rraycount'):
        value = params[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'ssgi.{key} must be a positive integer')
    for key in ('gidenoiseoffset', 'rdenoiseoffset'):
        value = params[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f'ssgi.{key} must be a non-negative integer')
    for key in ('thickness', 'intensity'):
        value = params[key]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0):
            raise ValueError(f'ssgi.{key} must be a non-negative number')

def build(config=None):
    cfg = json.loads(CONFIG.read_text()) if config is None else config
    validate_ssgi(cfg['ssgi'])
    arm = ET.parse(VENDOR/'universal_robots_ur5e/ur5e.xml').getroot()
    grip = ET.parse(VENDOR/'robotiq_2f85/2f85.xml').getroot()
    # The UR default's legacy highlights are ignored by MotrixSim 0.10.1.
    for material in arm.findall('./default//material'):
        material.attrib.pop('shininess', None)
        material.attrib.pop('specular', None)
    for material in arm.findall('./asset/material'):
        set_pbr(material, cfg['pbr']['ur5e'][material.get('name')])
    for material in grip.findall('./asset/material'):
        set_pbr(material, cfg['pbr']['gripper'][material.get('name')])
    for geom in grip.findall(".//body[@name='right_silicone_pad']/geom[@class='visual']") + grip.findall(".//body[@name='left_silicone_pad']/geom[@class='visual']"):
        if geom.get('mesh') != 'silicone_pad':
            raise ValueError('unexpected silicone pad visual mesh')
        geom.set('material', 'silicone')
    arm.set('model','stack_color_blocks')
    arm.find('compiler').attrib.pop('meshdir',None)
    option = arm.find('option')
    # MotrixSim 0.10.1 ignores MuJoCo's integrator selector. Keep the
    # supported timestep and let MotrixSim use its own physics integrator.
    option.attrib.pop('integrator', None)
    option.set('timestep', str(cfg['timestep']))
    key = arm.find('keyframe')
    if key is not None: arm.remove(key)
    # Express the vendored UR5e's PD general actuators as native position servos.
    # Preserve the vendored gains, damping, control ranges, and force limits.
    for default in arm.findall('./default//default'):
        general = default.find('general')
        if general is None: continue
        attrs = {k: v for k, v in general.attrib.items()
                 if k not in ('gaintype', 'biastype', 'gainprm', 'biasprm')}
        if general.get('gainprm') is not None:
            attrs['kp'] = general.get('gainprm').split()[0]
        if general.get('biasprm') is not None:
            attrs['kv'] = str(-float(general.get('biasprm').split()[2]))
        default.remove(general)
        ET.SubElement(default, 'position', attrs)
    for actuator in arm.findall('./actuator/general'):
        actuator.tag = 'position'
    for mesh in arm.findall('./asset/mesh'):
        mesh.set('file','vendor/menagerie/universal_robots_ur5e/assets/'+mesh.get('file'))
    for elem in grip.iter():
        if elem.get('name'): elem.set('name','grip_'+elem.get('name'))
        for attr in REF_ATTRS:
            if elem.get(attr): elem.set(attr,'grip_'+elem.get(attr))
    for mesh in grip.findall('./asset/mesh'):
        mesh.set('name','grip_'+Path(mesh.get('file')).stem)
        mesh.set('file','vendor/menagerie/robotiq_2f85/assets/'+mesh.get('file'))
    for section in ['default','asset','contact','tendon','equality','actuator']:
        source = grip.find(section)
        if source is None: continue
        target = arm.find(section)
        if target is None: target = ET.SubElement(arm,section)
        target.extend(list(source))
    wrist = next(e for e in arm.iter('body') if e.get('name')=='wrist_3_link')
    mount = grip.find('./worldbody/body')
    attachment = wrist.find("./site[@name='attachment_site']")
    if attachment is None:
        raise ValueError('UR wrist_3_link is missing attachment_site')
    mount_offset = cfg['robot'].get('gripper_mount_offset',[0,0,0])
    attachment_pos = [float(x) for x in attachment.get('pos','0 0 0').split()]
    mount.set('pos',vec([x+float(dx) for x,dx in zip(attachment_pos,mount_offset)]))
    mount.set('quat',attachment.get('quat','1 0 0 0'))
    camera_parent_name=cfg['wrist_camera'].get('parent','wrist_3_link')
    camera_parent={'wrist_3_link':wrist,'grip_base_mount':mount}.get(camera_parent_name)
    if camera_parent is None:
        raise ValueError(f'unsupported wrist_camera parent: {camera_parent_name}')
    ET.SubElement(camera_parent,'camera',name='wrist_rgb',pos=vec(cfg['wrist_camera']['pos']),
                  quat=vec(cfg['wrist_camera']['quat']),fovy=str(cfg['wrist_camera']['fovy_deg']))
    wrist.append(mount)
    world = arm.find('worldbody')
    robot_base_body = next(body for body in world.findall('body')
                           if body.get('name') == 'base')
    gravcomp = float(cfg['robot'].get('gravity_compensation', 1.0))
    if not 0.0 <= gravcomp <= 1.0:
        raise ValueError('robot.gravity_compensation must be between 0 and 1')
    for body in robot_base_body.iter('body'):
        body.set('gravcomp', str(gravcomp))
    # Keep the camera-attached default headlight dim. It otherwise adds to every
    # configured source and washes out upward-facing, pale surfaces.
    visual = arm.find('visual')
    if visual is None:
        visual = ET.Element('visual')
        arm.insert(0, visual)
    headlight = visual.find('headlight')
    if headlight is None:
        headlight = ET.SubElement(visual, 'headlight')
    for component in ('ambient', 'diffuse', 'specular'):
        headlight.set(component, vec(cfg['headlight'][component]))
    ssgi = ET.SubElement(visual, 'ssgi', active='true')
    for key in ('resolutionscale', 'raycount', 'stepcount', 'thickness', 'intensity',
                'gidenoiseoffset', 'rraycount', 'rdenoiseoffset'):
        ssgi.set(key, str(cfg['ssgi'][key]))
    # Drop the UR tracking light; the scene config owns the remaining sources.
    for light in world.findall('light'):
        world.remove(light)
    for light in cfg['lighting']:
        attrs={
            'name':light['name'],
            'pos':vec(light['pos']),
            'dir':vec(light['dir']),
            'mode':'fixed',
            'directional':str(light.get('directional',False)).lower(),
            'castshadow':str(light.get('castshadow',False)).lower(),
        }
        for key in ('ambient','diffuse','specular','cutoff','attenuation'):
            if key in light:
                attrs[key]=vec(light[key]) if isinstance(light[key],list) else str(light[key])
        ET.SubElement(world,'light',**attrs)
    table = cfg['table']
    table_center = table['center']
    table_top = table_center[2] + table['half_size'][2]
    robot_offset = cfg['robot']['base_from_table_top']
    robot_base = [table_center[0] + robot_offset[0],
                  table_center[1] + robot_offset[1],
                  table_top + robot_offset[2]]
    next(body for body in world.findall('body') if body.get('name') == 'base').set('pos', vec(robot_base))
    camera_offset=cfg['global_camera']['pos_from_table_top']
    camera_pos=[table_center[0]+camera_offset[0],table_center[1]+camera_offset[1],table_top+camera_offset[2]]
    ET.SubElement(world,'camera',name='global_rgb',pos=vec(camera_pos),
                  xyaxes=vec(cfg['global_camera']['xyaxes']),fovy=str(cfg['global_camera']['fovy_deg']))
    floor=cfg['floor']
    ET.SubElement(world,'geom',name='floor',type='plane',pos=vec([0,0,floor['z']]),
                  size=vec(floor['size']),rgba=vec(floor['rgba']))
    # These distant room surfaces are visual scenery. Collision-enabled room
    # boxes produce a screen-door artifact in MotrixSim's wrist camera.
    room=cfg['room']
    half_x,half_y=floor['size'][:2]
    wall_thickness=room['wall_thickness']
    ceiling_z=room['ceiling_z']
    wall_height=ceiling_z-floor['z']
    wall_center_z=(ceiling_z+floor['z'])/2
    wall_rgba=vec(room['rgba'])
    for name,pos,size in (
        ('wall_x_pos',[half_x+wall_thickness/2,0,wall_center_z],
         [wall_thickness/2,half_y+wall_thickness,wall_height/2]),
        ('wall_x_neg',[-half_x-wall_thickness/2,0,wall_center_z],
         [wall_thickness/2,half_y+wall_thickness,wall_height/2]),
        ('wall_y_pos',[0,half_y+wall_thickness/2,wall_center_z],
         [half_x,wall_thickness/2,wall_height/2]),
        ('wall_y_neg',[0,-half_y-wall_thickness/2,wall_center_z],
         [half_x,wall_thickness/2,wall_height/2]),
    ):
        ET.SubElement(world,'geom',name=name,type='box',pos=vec(pos),size=vec(size),
                      rgba=wall_rgba,contype='0',conaffinity='0')
    if room.get('ceiling', True):
        ET.SubElement(world,'geom',name='ceiling',type='box',
                      pos=vec([0,0,ceiling_z+wall_thickness/2]),
                      size=vec([half_x,half_y,wall_thickness/2]),rgba=wall_rgba,
                      contype='0',conaffinity='0')
    table_material=ET.SubElement(arm.find('asset'),'material',name='table_material',rgba=vec(table['rgba']))
    set_pbr(table_material,cfg['pbr']['table'])
    ET.SubElement(world,'geom',name='table',type='box',pos=vec(table['center']),
                  size=vec(table['half_size']),friction=str(table['friction']),material='table_material')
    # Colored strips rise 0.1 mm above the unchanged physical/visual tabletop.
    sticker_half_thickness = cfg['stickers']['half_thickness']
    sticker_top=table_top+cfg['stickers']['surface_offset']
    for sticker in cfg['stickers']['items']:
        ET.SubElement(arm.find('asset'),'material',name='sticker_'+sticker['name']+'_material',
                      rgba=vec(sticker['rgba']),depthbias='1',castshadow='false')
        ET.SubElement(world,'geom',name='sticker_'+sticker['name'],type='box',
                      pos=vec([*sticker['center'],sticker_top-sticker_half_thickness]),
                      size=vec([*sticker['half_size'],sticker_half_thickness]),
                      material='sticker_'+sticker['name']+'_material',contype='0',conaffinity='0')
    for name,prop in cfg['background'].items():
        if name == 'note': continue
        body = ET.SubElement(world,'body',name='background_'+name,mocap='true',
                             pos=vec(prop['base_pos']),quat='0.70710678 0.70710678 0 0')
        parts=prop.get('parts',[{'name':'visual','mesh':prop['mesh']}])
        frames={}
        for part in parts:
            if part.get('reference_frame'):
                # Keep mocap bodies childless (required by MotrixSim). A named
                # reference frame is represented as a local offset instead.
                frames[part['name']]=part.get('frame_pos',[0,0,0])
        for part in parts:
            mesh_name='background_'+name+'_'+part['name']
            mesh_attrs={'name':mesh_name,'file':part['mesh']}
            if 'scale' in prop: mesh_attrs['scale']=vec(prop['scale'])
            ET.SubElement(arm.find('asset'),'mesh',**mesh_attrs)
            geom_attrs={'name':mesh_name,'type':'mesh','mesh':mesh_name,
                        'contype':'0','conaffinity':'0'}
            frame_name=part.get('relative_to')
            if frame_name is None and part.get('reference_frame'):
                frame_name=part['name']
            if frame_name in frames and 'relative_pos' in part:
                frame_pos=frames[frame_name]
                geom_attrs['pos']=vec([a+b for a,b in zip(frame_pos,part['relative_pos'])])
            elif 'relative_pos' in part: geom_attrs['pos']=vec(part['relative_pos'])
            elif 'pos' in part: geom_attrs['pos']=vec(part['pos'])
            if 'rgba' in part: geom_attrs['rgba']=vec(part['rgba'])
            ET.SubElement(body,'geom',**geom_attrs)
    for name,color in [('yellow','0.95 0.78 0.04 1'),('orange','0.98 0.31 0.03 1'),('sky_blue','0.32 0.72 0.95 1')]:
        material_name=name+'_block_material'
        block_material=ET.SubElement(arm.find('asset'),'material',name=material_name,rgba=color)
        set_pbr(block_material,cfg['pbr']['blocks'])
        local_xy=cfg['block']['fixed_xy'][name]
        initial_pos=[table_center[0]+local_xy[0],table_center[1]+local_xy[1],
                     table_top+cfg['block']['half_size']+cfg['block']['clearance_above_table']]
        body=ET.SubElement(world,'body',name=name+'_block',pos=vec(initial_pos))
        ET.SubElement(body,'freejoint',name=name+'_free')
        ET.SubElement(body,'geom',name=name+'_geom',type='box',size=vec([cfg['block']['half_size']]*3),mass=str(cfg['block']['mass']),friction=str(cfg['block']['friction']),material=material_name)
    ET.indent(arm,space='  ')
    OUTPUT.write_text(ET.tostring(arm,encoding='unicode')+'\n')
    return OUTPUT

if __name__=='__main__': print(build())
