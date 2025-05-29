# IsaacLab的使用
目录
[1.环境初始化](#环境初始化)
[2.API](#api)


## 环境初始化
### 1. 创建外部仓库
```bash
cd IsaacLab
# 查看已有的具身智能环境
# python scripts/environments/list_envs.py
# 创建环境 external 外部环境 OR internal 内部环境
# 创建方法 direct OR manager-based
python ./isaaclab.sh --new
```
Once created, navigate to the installed project and run `python -m pip install -e source/<given-project-name>`

### 2. 已有环境分析
**Direct Workflow**
<img src="icon/image.png"  width="400" />
环境定义文件 `ant_env.py`
`gym.make`配置文件`__init__.py`
`skrl`算法流程配置文件`skrl_ppo_cfg.yaml`
<img src="icon/image-1.png"  width="400" />
```bash
# 训练
python scripts/reinforcement_learning/skrl/train.py --task Isaac-Ant-Direct-v0 --livestream 2
# or
LIVESTREAM=2 python scripts/reinforcement_learning/skrl/train.py --task Isaac-Ant-Direct-v0 
```

**ManagerBased Workflow 同上** 

```bash
# 训练
python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 scripts/reinforcement_learning/rl_games/train.py --task Isaac-Factory-NutThread-Direct-v0 --distributed --headless #分布式训练
python scripts/reinforcement_learning/skrl/train.py --task Isaac-Franka-Cabinet-Direct-v0 --distributed --headless # 单机训练
# 结果可视化
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py --task Isaac-Factory-NutThread-Direct-v0  --num_envs 64  --livestream 2
./isaaclab.sh -p scripts/reinforcement_learning/skrl/play.py --task Isaac-Franka-Cabinet-Direct-v0  --num_envs 64  --video #录制视频
```


## API

### 1.环境初始化相关 API
<details>
<summary> 场景模拟 </summary>

```python
import argparse
from isaaclab.app import AppLauncher
# create argparser
parser = argparse.ArgumentParser(description="Tutorial on creating an empty stage.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
from isaaclab.sim import SimulationCfg, SimulationContext
def main():
    """Main function."""

    # Initialize the simulation context
    sim_cfg = SimulationCfg(dt=0.01)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Simulate physics
    while simulation_app.is_running():
        # perform step
        sim.step()
if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

```
</details>

<details>
<summary>场景生成</summary>

```python
import isaaclab.sim as sim_utils
import isaacsim.core.utils.prims as prim_utils

# spawn a ground plane
cfg_ground = sim_utils.GroundPlaneCfg()
cfg_ground.func("/World/DefaultGroundPlane",cfg_ground)
# spawn lights
cfg_light_distant = sim_utils.DistantLightCfg(intensity = 3000,color=(0.75,0.75,0.75))
cfg_light_distant.func("/World/LightDistant",cfg_light_distant,translation=(1,0,10))
# spawn primitive shapes
prim_utils.create_prim("/World/Objects","Xform")
# spawn a cone
cfg_cone = sim_utils.ConeCfg(
    radius = 0.15,
    height = 0.5,
    visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0,0.0,0.0))
)
cfg_cone.func("/World/Objects/Cone",cfg_cone,translation=(-1.0,1.0,1.0))
# spawn a cone with colliders and rigid body
cfg_cone_rigid = sim_utils.ConeCfg(
    radius = 0.15,
    height = 0.5,
    rigid_props = sim_utils.RigidBodyPropertiesCfg(),
    mass_props = sim_utils.MassPropertiesCfg(mass = 1.0),
    collision_props = sim_utils.CollisionPropertiesCfg()
    visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0,1.0,0.0))
)
cfg_cone_rigid.func("/World/Objects/ConeRigid",cfg_cone_rigid,translation=(-0.2,0.0,2.0),orientation=(0.5,0.0,0.5,0.0))
 
# spawn a blue cuboid with deformable body  柔性材料必须是mesh
cfg_cuboid_deformable = sim_utils.MeshCuboidCfg(
    size=(0.2, 0.5, 0.2),
    deformable_props=sim_utils.DeformableBodyPropertiesCfg(),
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
    physics_material=sim_utils.DeformableBodyMaterialCfg(),
)
cfg_cuboid_deformable.func("/World/Objects/CuboidDeformable", cfg_cuboid_deformable, translation=(0.15, 0.0, 2.0))


# spawn a usd file of a table into the scene
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
cfg.func("/World/Objects/Table", cfg, translation=(0.0, 0.0, 1.05))
# OR
# cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
# sim_utils.spawn_from_usd("/World/Table",cfg)
```
>Note:
All the scene designing must happen before the simulation starts. **Once the simulation starts, we recommend keeping the scene frozen and only altering the properties of the prim.** This is particularly important for GPU simulation as adding new prims during simulation may alter the physics simulation buffers on GPU and lead to unexpected behaviors.
</details>

<details>
<summary> 1 </summary>


</details>


### 2.Direct Workflow API

### 3.ManagerBased Workflow API

### 4.skrl algorithm API

