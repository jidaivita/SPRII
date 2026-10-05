"""Fixed, parameter-independent MuJoCo orthographic grayscale renderer.

Black background; unlit emissive role discs (96, 224); 4x MSAA, no shadows.
State is copied before mj_forward and rendering; no temporal lag or overlays.
"""
import numpy as np
import mujoco
from .schema import Config
from .physics import validate_state

BACKGROUND = 0
OBJECT1_GRAY = 96
OBJECT2_GRAY = 224
CONNECTOR_GRAY = 40
BACKEND = 'mujoco_3.3.5_opengl_orthographic_emissive_v1'


def pixel_to_world(px, py, config=Config()):
    return np.stack([(np.asarray(px)+.5)/config.resolution*config.field_width-config.field_width/2,
                     config.field_width/2-(np.asarray(py)+.5)/config.resolution*config.field_width],axis=-1)


def world_to_pixel(x, y, config=Config()):
    return np.stack([(np.asarray(x)/config.field_width+.5)*config.resolution-.5,
                     (.5-np.asarray(y)/config.field_width)*config.resolution-.5],axis=-1)


def _visual_model(config):
    bodies=''
    for i,gray in ((1,OBJECT1_GRAY),(2,OBJECT2_GRAY)):
        bodies+=f'''<body><joint type="slide" axis="1 0 0"/><joint type="slide" axis="0 1 0"/>
        <geom type="cylinder" size="{config.radius} .001" material="role{i}" contype="0" conaffinity="0"/></body>'''
    xml=f'''<mujoco><visual><global offwidth="{config.resolution}" offheight="{config.resolution}"/>
    <quality offsamples="4"/><headlight ambient="0 0 0" diffuse="0 0 0" specular="0 0 0"/></visual>
    <asset><material name="role1" rgba="{OBJECT1_GRAY/255} {OBJECT1_GRAY/255} {OBJECT1_GRAY/255} 1" emission="1" specular="0"/>
    <material name="role2" rgba="{OBJECT2_GRAY/255} {OBJECT2_GRAY/255} {OBJECT2_GRAY/255} 1" emission="1" specular="0"/></asset>
    <worldbody><camera name="top" pos="0 0 3" orthographic="true" fovy="{config.field_width}"/>{bodies}</worldbody></mujoco>'''
    return mujoco.MjModel.from_xml_string(xml)


class VisualRenderer:
    """One renderer for either offline replay or online observation delivery."""
    def __init__(self,config=Config()):
        self.config=config
        self.model=_visual_model(config);self.data=mujoco.MjData(self.model)
        self.renderer=mujoco.Renderer(self.model,height=config.resolution,width=config.resolution)

    def frame(self,state):
        state=validate_state(state,self.config)
        data=self.data;renderer=self.renderer
        data.qpos[:]=state[:4];data.qvel[:]=state[4:];mujoco.mj_forward(self.model,data)
        renderer.update_scene(data,camera='top')
        geom=renderer.scene.geoms[renderer.scene.ngeom]
        mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array([CONNECTOR_GRAY/255]*3+[1.]))
        mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_CAPSULE,.003,np.r_[state[:2],-.003],np.r_[state[2:4],-.003])
        geom.emission=1.;geom.specular=0.;renderer.scene.ngeom+=1
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
        rgb=renderer.render()
        return np.rint(rgb.astype(float)@np.array([.2126,.7152,.0722])).astype(np.uint8)

    def close(self):self.renderer.close()
    def __enter__(self):return self
    def __exit__(self,*exc):self.close()


def render_states(states, config=Config()):
    states=np.asarray(states,dtype=float)
    if states.ndim != 2 or states.shape[1]!=8:raise ValueError('expected [T+1,8]')
    with VisualRenderer(config) as renderer:
        frames=[renderer.frame(state) for state in states]
    return np.stack(frames) if frames else np.empty((0,config.resolution,config.resolution),np.uint8)


def run_checks(report_path):
    """Renderer calibration uses image intensities, never segmentation buffers."""
    from dataclasses import replace
    from pathlib import Path
    import json
    states=np.array([[-.3+d,.2+d,.3-d,-.2-d,0,0,0,0] for d in np.linspace(-.08,.08,9)])
    checks={}; errors={}
    for resolution in (64,128):
        config=replace(Config(),resolution=resolution); images=render_states(states,config)
        positions=[]
        for frame in images:
            pair=[]
            for gray in (OBJECT1_GRAY,OBJECT2_GRAY):
                yy,xx=np.where(frame==gray)
                pair.extend(pixel_to_world(xx.mean(),yy.mean(),config))
            positions.append(pair)
        error=float(np.max(np.abs(np.array(positions)-states[:,:4])))
        errors[str(resolution)]={'max_position_error_m':error,'tolerance_m':config.field_width/config.resolution/2}
        checks[f'pixel_world_alignment_{resolution}']='PASS' if error<config.field_width/config.resolution/2 else 'FAIL'
        checks[f'uint8_grayscale_{resolution}']='PASS' if images.dtype==np.uint8 and images.shape==(9,resolution,resolution) else 'FAIL'
    frame=render_states(states[:1]); changed=states[:1].copy(); changed[:,4:]=1
    checks['velocity_not_encoded']='PASS' if np.array_equal(frame,render_states(changed)) else 'FAIL'
    checks['deterministic_same_state']='PASS' if np.array_equal(frame,render_states(states[:1])) else 'FAIL'
    report={'status':'executed','backend':BACKEND,'scientific_visual_identification':'NOT_ESTABLISHED','checks':checks,'centroid_calibration':errors,'background':BACKGROUND,'role_gray':[OBJECT1_GRAY,OBJECT2_GRAY],'connector_gray':CONNECTOR_GRAY,'antialiasing':'4 sample MSAA','lighting':'emissive unlit; no shadows','camera':'fixed orthographic; field_width meters; x right,y up; pixel centers offset .5','failed_attempts':[]}
    path=Path(report_path); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(report,indent=2)+'\n'); return report

if __name__=='__main__':
    import sys,json
    print(json.dumps(run_checks(sys.argv[1]),indent=2))
