import numpy as np
from pydrake.multibody.inverse_kinematics import DifferentialInverseKinematicsParameters
from pydrake.systems.framework import DiagramBuilder, LeafSystem
from pydrake.visualization import MeshcatPoseSliders

from hardware.kuka import create_hardware_diagram_plant
from pydrake.all import (
    LeafSystem,
    RigidTransform,
    Value,
    ValueProducer,
    AbstractValue,
    MultibodyPlant,
    Multiplexer,
    DifferentialInverseKinematicsIntegrator,
    ConstantValueSource
)
from manipulation.scenarios import AddMultibodyTriad

from hardware.cameras import Cameras
import apriltag
import cv2
from collections import defaultdict

## --- teleop kuka drake API ---
def AddIiwaDifferentialIK(builder, plant, frame=None):
    params = DifferentialInverseKinematicsParameters(
        plant.num_positions(), plant.num_velocities()
    )
    time_step = plant.time_step()
    q0 = plant.GetPositions(plant.CreateDefaultContext())
    params.set_nominal_joint_position(q0) # nominal position for nullspace projection
    params.set_end_effector_angular_speed_limit(np.pi/12)
    params.set_end_effector_translational_velocity_limits(
        [-0.01, -0.01, -0.01], [0.01, 0.01, 0.01]
    )
    if plant.num_positions() == 3:  # planar iiwa
        iiwa14_velocity_limits = np.array([1.4, 1.3, 2.3])
        params.set_joint_velocity_limits(
            (-iiwa14_velocity_limits, iiwa14_velocity_limits)
        )
        # These constants are in body frame
        assert (
            frame.name() == "iiwa_link_7"
        ), "Still need to generalize the remaining planar diff IK params for different frames"  # noqa
        params.set_end_effector_velocity_flag(
            [True, False, False, True, False, True]
        )
    else:
        # iiwa14_velocity_limits = np.array([1.4, 1.4, 1.7, 1.3, 2.2, 2.3, 2.3])
        iiwa14_velocity_limits = np.ones(7) * 0.17
        params.set_joint_velocity_limits(
            (-iiwa14_velocity_limits, iiwa14_velocity_limits)
        )
        params.set_joint_centering_gain(0 * np.eye(7))
    if frame is None:
        frame = plant.GetFrameByName("body")
    differential_ik = builder.AddSystem(
        DifferentialInverseKinematicsIntegrator(
            plant,
            frame,
            time_step,
            params,
            log_only_when_result_state_changes=True,
        )
    )
    return differential_ik

class HardwareKukaPose(LeafSystem):
    def __init__(self, hardware_plant: MultibodyPlant, kuka_frame_name = 'iiwa_link_7'):
        LeafSystem.__init__(self)
        self._plant = hardware_plant
        self._plant_context = hardware_plant.CreateDefaultContext()
        self._frame_name = kuka_frame_name
        
        self.DeclareVectorInputPort("kuka_q", 7)
        #make abstract port for kuka pose
        self.DeclareAbstractOutputPort("kuka_pose", lambda: Value(RigidTransform()), self.CalcOutput)
        
    def CalcOutput(self, context, output):
        q = self.get_input_port().Eval(context)
        self._plant.SetPositions(self._plant_context, q)
        pose = self._plant.GetFrameByName(self._frame_name).CalcPoseInWorld(self._plant_context)
        output.set_value(pose)


def teleop_diagram(meshcat, kuka_frame_name="iiwa_link_7"):
    meshcat.ResetRenderMode()
    builder = DiagramBuilder()
    
    hardware_diagram, controller_plant, scene_graph = create_hardware_diagram_plant(scenario_filepath='../config/calib_med.yaml', position_only=True, meshcat = meshcat, package_file='../package.xml')
    
    hardware_block = builder.AddSystem(hardware_diagram)
    # Set up differential inverse kinematics.
    differential_ik = AddIiwaDifferentialIK(
        builder,
        controller_plant,
        frame=controller_plant.GetFrameByName(kuka_frame_name),
    )
    builder.Connect(
        differential_ik.get_output_port(),
        hardware_block.GetInputPort("iiwa_thanos.position")
    )
    
    builder.Connect(
        differential_ik.get_output_port(),
        hardware_block.GetInputPort("iiwa_thanos_meshcat.position")
    )
    
    kuka_state = builder.AddSystem(Multiplexer([7,7]))
    builder.Connect(
        hardware_block.GetOutputPort("iiwa_thanos.position_measured"),
        kuka_state.get_input_port(0)
    )
    builder.Connect(
        hardware_block.GetOutputPort("iiwa_thanos.velocity_estimated"),
        kuka_state.get_input_port(1)
    )
    
    builder.Connect(
        kuka_state.get_output_port(),
        differential_ik.GetInputPort("robot_state"),
    )
    
    use_state = builder.AddSystem(ConstantValueSource(Value(False)))
    builder.Connect(
        use_state.get_output_port(),
        differential_ik.GetInputPort("use_robot_state"),
    )

    # Set up teleop widgets.
    meshcat.DeleteAddedControls()
    sliders = MeshcatPoseSliders(
        meshcat,
        lower_limit=[-np.pi, -np.pi, -np.pi , -0.6, -0.6, 0.0],
        upper_limit=[ np.pi, np.pi, np.pi, 0.8, 0.9, 2.0],
        step=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
    )
    
    teleop = builder.AddSystem(sliders)
    builder.Connect(
        teleop.get_output_port(), differential_ik.GetInputPort("X_WE_desired")
    )
    
    # Note: This is using "Cheat Ports". For it to work on hardware, we would
    # need to construct the initial pose from the HardwareStation outputs.
    
    kuka_ee_pose = builder.AddSystem(HardwareKukaPose(controller_plant, kuka_frame_name=kuka_frame_name))
    builder.Connect(hardware_block.GetOutputPort("iiwa_thanos.position_measured"), kuka_ee_pose.get_input_port())
    builder.Connect(kuka_ee_pose.get_output_port(), teleop.get_input_port()) #used for initialization
    AddMultibodyTriad(controller_plant.GetFrameByName("calibration_frame"), scene_graph)

    builder.ExportOutput(kuka_ee_pose.get_output_port(), "teleop_pose")

    diagram = builder.Build()
    
    return diagram, scene_graph

## --- camera drake API ---
class CameraTagPublisher(LeafSystem):
    def __init__(self, cameras: Cameras, tag_width: float = 0.056):
        LeafSystem.__init__(self)
        self.tag_width = tag_width
        self.cameras = cameras
        self.n_cam = cameras.n_fixed_cameras
        self.detector = apriltag.Detector()
        self.tag2kukabase = RigidTransform()
        self.cameras_datapoints = defaultdict(list)
        
        # take as input Kuka
        self.DeclareAbstractInputPort("tag2kukabase", Value(RigidTransform()))
        self.DeclarePeriodicPublishEvent(period_sec=0.1, offset_sec=0.0, publish=self.SaveKukaPose)
        self.DeclareForcedPublishEvent(self.DetectTagEvent)
        
    def SaveKukaPose(self, context):
        self.tag2kukabase: RigidTransform = self.get_input_port().Eval(context)
        
    def DetectTagEvent(self, context):
        obs = self.cameras.get_obs(get_color=True, get_depth=False)
        for cam_idx in range(self.n_cam):
            color = obs[f'color_{cam_idx}'][-1]
            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            detections = self.detector.detect(gray)
            for detect in detections:
                if detect.tag_id == 0:
                    print(f"Camera {cam_idx} detected tag pose")
                    pt_2d = detect.center
                    pt_3d = self.tag2kukabase.translation()
                    self.cameras_datapoints[f'cam{cam_idx}'].append( (pt_2d,pt_3d) )
                    
        print(self.cameras_datapoints)
        print()