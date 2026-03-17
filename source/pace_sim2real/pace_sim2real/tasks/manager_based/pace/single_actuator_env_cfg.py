from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg
import torch
import isaaclab.sim as sim_utils

from pace_sim2real.utils import PaceDCMotorCfg
from .pace_sim2real_env_cfg import PaceSim2realEnvCfg, PaceSim2realSceneCfg, PaceCfg


SINGLE_ACTUATOR_CFG = PaceDCMotorCfg(
    joint_names_expr=[".*"],
    saturation_effort=12.0,    
    effort_limit=12.0,
    velocity_limit=37.5,       
    stiffness={".*": 40.74},   
    damping={".*": 0.5088},    
    encoder_bias=[0.0],
    max_delay=10,
)

@configclass
class SingleActuatorPaceCfg(PaceCfg):
    """Pace configuration for single actuator."""
    robot_name: str = "single_actuator"
    data_dir: str = "single_actuator/chirp_data.pt"
    bounds_params: torch.Tensor = torch.zeros((5, 2))
    joint_order: list[str] = ["output_joint"] 
    

    def __post_init__(self):
        self.bounds_params[0, 0] = 1e-5
        self.bounds_params[0, 1] = 0.002  # armature
        self.bounds_params[1, 1] = 0.5  # damping
        self.bounds_params[2, 1] = 0.5  # friction
        self.bounds_params[3, 0] = 0.0  # bias
        self.bounds_params[3, 1] = 0.0  # bias
        self.bounds_params[4, 1] = 2.0  # delay

@configclass
class SingleActuatorSceneCfg(PaceSim2realSceneCfg):
    """Scene configuration for single actuator."""
    
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/tarun/Desktop/xTerra/o12_urdf/urdf/o12_urdf/o12_urdf.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            joint_pos={"output_joint": 0.0},  # UPDATE TO YOUR ACTUAL JOINT NAME
        ),
        actuators={"motor": SINGLE_ACTUATOR_CFG},
    )

@configclass
class SingleActuatorEnvCfg(PaceSim2realEnvCfg):
    scene: SingleActuatorSceneCfg = SingleActuatorSceneCfg()
    sim2real: PaceCfg = SingleActuatorPaceCfg()

    def __post_init__(self):
        super().__post_init__()
        self.sim.dt = 0.001
        self.decimation = 1
