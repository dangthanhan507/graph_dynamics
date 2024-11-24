import numpy as np
from pydrake.all import (
    Simulator,
    DiagramBuilder,
    TrajectorySource,
    DiagramBuilder,
    PiecewisePolynomial,
    MultibodyPlant,
    DiagramBuilder,
    ApplyMultibodyPlantConfig,
    Parser,
    ProcessModelDirectives,
    ModelDirectives,
    AddMultibodyPlantSceneGraph,
    LoadModelDirectives,
    MeshcatVisualizerParams,
    Role,
    MeshcatVisualizer,
    Diagram
)
from manipulation.station import MakeHardwareStation, MakeHardwareStationInterface, load_scenario

def get_hardware_blocks(hardware_builder, scenario, meshcat = None, package_file='./package.xml'):
    real_station = hardware_builder.AddNamedSystem(
        'real_station',
        MakeHardwareStationInterface(
            scenario,
            meshcat=meshcat,
            package_xmls=[package_file]
        )
    )
    fake_station = hardware_builder.AddNamedSystem(
        'fake_station',
        MakeHardwareStation(
            scenario,
            meshcat=meshcat,
            package_xmls=[package_file]    
        )
    )
    hardware_plant = MultibodyPlant(scenario.plant_config.time_step)
    ApplyMultibodyPlantConfig(scenario.plant_config, hardware_plant)
    parser = Parser(hardware_plant)
    parser.package_map().AddPackageXml(package_file)
    ProcessModelDirectives(
        directives=ModelDirectives(directives=scenario.directives),
        plant=hardware_plant,
        parser=parser
    )
    return real_station, fake_station

def create_hardware_diagram_plant(scenario_filepath, position_only=True, meshcat=None, package_file="../package.xml"):
    hardware_builder = DiagramBuilder()
    scenario = load_scenario(filename=scenario_filepath, scenario_name="Demo")
    real_station, fake_station = get_hardware_blocks(hardware_builder, scenario, meshcat=meshcat, package_file=package_file)
    
    hardware_plant = fake_station.GetSubsystemByName("plant")
    
    
    hardware_builder.ExportInput(
        real_station.GetInputPort("iiwa_thanos.position"), "iiwa_thanos.position"
    )
    hardware_builder.ExportInput(
        fake_station.GetInputPort("iiwa_thanos.position"), "iiwa_thanos_meshcat.position"
    )
    if not position_only:
        hardware_builder.ExportInput(
            real_station.GetInputPort("iiwa_thanos.feedforward_torque"), "iiwa_thanos.feedforward_torque"
        )
        hardware_builder.ExportOutput(
            real_station.GetOutputPort("iiwa_thanos.torque_external"), "iiwa_thanos.torque_external"
        )
        hardware_builder.ExportOutput(
            real_station.GetOutputPort("iiwa_thanos.torque_commanded"), "iiwa_thanos.torque_commanded"
        )
        
    hardware_builder.ExportOutput(
        real_station.GetOutputPort("iiwa_thanos.position_commanded"), "iiwa_thanos.position_commanded"
    )
    hardware_builder.ExportOutput(
        real_station.GetOutputPort("iiwa_thanos.position_measured"), "iiwa_thanos.position_measured"
    )
    
    hardware_builder.ExportOutput(
        real_station.GetOutputPort("iiwa_thanos.velocity_estimated"), "iiwa_thanos.velocity_estimated"
    )
    
    hardware_diagram = hardware_builder.Build()
    return hardware_diagram, hardware_plant

#NOTE: assume u are in graph_tracking/src when running
class Kuka:
    def __init__(self, scenario_file='../config/calib.yaml', package_file='../package.xml', position_only=True):
        hardware_diagram, hardware_plant = create_hardware_diagram_plant(scenario_file, position_only=position_only, package_file=package_file)
        self.hardware_diagram = hardware_diagram
        self.hardware_plant = hardware_plant
        self.plant_context = hardware_plant.CreateDefaultContext()
        
    def get_curr_joints(self):
        context = self.hardware_diagram.CreateDefaultContext()
        self.hardware_diagram.ExecuteInitializationEvents(context)
        return self.hardware_diagram.GetOutputPort("iiwa_thanos.position_measured").Eval(context)
        
    def get_pose(self, frame_name="thanos_finger"):
        curr_q = self.get_curr_joints()
        self.hardware_plant.SetPositions(self.plant_context, curr_q)
        return self.hardware_plant.GetFrameByName(frame_name).CalcPoseInWorld(self.plant_context)