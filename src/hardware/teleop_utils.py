import numpy as np
from pydrake.geometry import StartMeshcat
from pydrake.multibody.inverse_kinematics import DifferentialInverseKinematicsParameters
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder, LeafSystem
from pydrake.visualization import MeshcatPoseSliders

from hardware.kuka import create_hardware_diagram_plant
from pydrake.all import (
    LeafSystem,
    RigidTransform,
    Value,
    MultibodyPlant,
    Multiplexer,
    DifferentialInverseKinematicsIntegrator,
    ConstantValueSource
)

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
    
    #create standard diagram
    hardware_diagram, controller_plant = create_hardware_diagram_plant(scenario_filepath='../config/calib_med.yaml', position_only=True, meshcat = meshcat, package_file='../package.xml')
    hardware_block = builder.AddSystem(hardware_diagram)
    
    differential_ik = AddIiwaDifferentialIK(
        builder,
        controller_plant,
        frame=controller_plant.GetFrameByName(kuka_frame_name),
    )
    
    meshcat.DeleteAddedControls()
    sliders = MeshcatPoseSliders(
        meshcat,
        lower_limit=[-np.pi, -np.pi, -np.pi , -0.6, -0.6, 0.0],
        upper_limit=[ np.pi, np.pi, np.pi, 0.8, 0.9, 2.0],
        step=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
    )
    teleop = builder.AddSystem(sliders)
    
    use_state = builder.AddSystem(ConstantValueSource(Value(False)))
    kuka_state = builder.AddSystem(Multiplexer([7,7]))
    kuka_ee_pose = builder.AddSystem(HardwareKukaPose(controller_plant,kuka_frame_name=kuka_frame_name))
    
    builder.Connect(
        differential_ik.get_output_port(),
        hardware_block.GetInputPort("iiwa_thanos.position")
    )
    builder.Connect(
        differential_ik.get_output_port(),
        hardware_block.GetInputPort("iiwa_thanos_meshcat.position")
    )
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
    builder.Connect(
        use_state.get_output_port(),
        differential_ik.GetInputPort("use_robot_state"),
    )
    builder.Connect(
        teleop.get_output_port(), differential_ik.GetInputPort("X_WE_desired")
    )
    builder.Connect(
        hardware_block.GetOutputPort("iiwa_thanos.position_measured"), 
        kuka_ee_pose.get_input_port()
    )
    builder.Connect(
        kuka_ee_pose.get_output_port(),
        teleop.get_input_port()
    ) #used for initialization
    
    diagram.ExportOutput(
        kuka_ee_pose.get_output_port(), f"{kuka_frame_name}_pose"
    )
    
    diagram = builder.Build()
    
    meshcat.AddButton("Stop Simulation", "Escape")
    return diagram

