
import numpy as np
from pydrake.geometry import StartMeshcat
from pydrake.multibody.inverse_kinematics import (
    DifferentialInverseKinematicsParameters,
    DifferentialInverseKinematicsStatus,
    DoDifferentialInverseKinematics,
)
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder, EventStatus, LeafSystem
from pydrake.visualization import MeshcatPoseSliders

from manipulation import running_as_notebook
from manipulation.scenarios import AddIiwaDifferentialIK, ExtractBodyPose
from manipulation.station import MakeHardwareStation, load_scenario
from hardware.kuka import create_hardware_diagram_plant
from pydrake.all import (
    LeafSystem,
    RigidTransform,
    Value,
    MultibodyPlant,
    Multiplexer
)
# Start the visualizer.
meshcat = StartMeshcat()

class HardwareKukaPose(LeafSystem):
    def __init__(self, hardware_plant: MultibodyPlant):
        LeafSystem.__init__(self)
        self._plant = hardware_plant
        self._plant_context = hardware_plant.CreateDefaultContext()
        
        self.DeclareVectorInputPort("kuka_q", 7)
        #make abstract port for kuka pose
        self.DeclareAbstractOutputPort("kuka_pose", lambda: Value(RigidTransform()), self.CalcOutput)
        
    def CalcOutput(self, context, output):
        q = self.get_input_port().Eval(context)
        self._plant.SetPositions(self._plant_context, q)
        pose = self._plant.GetFrameByName("iiwa_link_7").CalcPoseInWorld(self._plant_context)
        output.set_value(pose)

def teleop_3d():
    meshcat.ResetRenderMode()

    builder = DiagramBuilder()
    
    hardware_diagram, controller_plant = create_hardware_diagram_plant(scenario_filepath='../config/calib_med.yaml', position_only=True, meshcat = meshcat, package_file='../package.xml')
    
    hardware_block = builder.AddSystem(hardware_diagram)
    
    # Set up differential inverse kinematics.
    differential_ik = AddIiwaDifferentialIK(
        builder,
        controller_plant,
        frame=controller_plant.GetFrameByName("iiwa_link_7"),
    )
    builder.Connect(
        differential_ik.get_output_port(),
        hardware_block.GetInputPort("iiwa_thanos.position"),
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

    # Set up teleop widgets.
    
    #get current iiwa_link_7 pose
    
    
    meshcat.DeleteAddedControls()
    sliders = MeshcatPoseSliders(
        meshcat,
        lower_limit=[0, -0.5, -np.pi, -0.6, -0.8, 0.0],
        upper_limit=[2 * np.pi, np.pi, np.pi, 0.8, 0.3, 1.1],
    )
    
    teleop = builder.AddSystem(sliders)
    builder.Connect(
        teleop.get_output_port(), differential_ik.GetInputPort("X_WE_desired")
    )
    # Note: This is using "Cheat Ports". For it to work on hardware, we would
    # need to construct the initial pose from the HardwareStation outputs.
    
    kuka_ee_pose = builder.AddSystem(HardwareKukaPose(controller_plant))
    builder.Connect(hardware_block.GetOutputPort("iiwa_thanos.position_measured"), kuka_ee_pose.get_input_port())
    builder.Connect(kuka_ee_pose.get_output_port(), teleop.get_input_port()) #used for initialization

    diagram = builder.Build()
    simulator = Simulator(diagram)
    simulator_context = simulator.get_mutable_context()

    simulator.set_target_realtime_rate(1.0)

    meshcat.AddButton("Stop Simulation", "Escape")
    print("Press Escape to stop the simulation")
    while meshcat.GetButtonClicks("Stop Simulation") < 1:
        simulator.AdvanceTo(simulator.get_context().get_time() + 2.0)
    meshcat.DeleteButton("Stop Simulation")

teleop_3d()