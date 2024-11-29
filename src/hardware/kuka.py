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
import multiprocessing as mp

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
    scene_graph_of_plant = fake_station.GetSubsystemByName("scene_graph")
    
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
    return hardware_diagram, hardware_plant, scene_graph_of_plant

def goto_joints(joint_thanos, endtime = 10.0, joint_speed = None):
    root_builder = DiagramBuilder()
    hardware_diagram, hardware_plant, _ = create_hardware_diagram_plant(scenario_filepath="../config/med.yaml",  position_only=True)
    
    hardware_block = root_builder.AddSystem(hardware_diagram)
    
    ## make a plan from current position to desired position
    context = hardware_diagram.CreateDefaultContext()
    hardware_diagram.ExecuteInitializationEvents(context)
    curr_q = hardware_diagram.GetOutputPort("iiwa_thanos.position_measured").Eval(context)
    
    # speed is in rad/s
    if joint_speed is not None:
        max_dq = np.max(np.abs(joint_thanos - curr_q)) # get max a joint can move
        endtime_from_speed = max_dq / joint_speed # get time to move at speed
        endtime = max(endtime, endtime_from_speed)
        endtime = endtime_from_speed
        print(endtime_from_speed)
    
    ts = np.array([0.0, endtime])
    qs = np.array([curr_q, joint_thanos])
    traj = PiecewisePolynomial.FirstOrderHold(ts, qs.T)
        
    traj_block = root_builder.AddSystem(TrajectorySource(traj))
    root_builder.Connect(traj_block.get_output_port(), hardware_block.GetInputPort("iiwa_thanos.position"))

    root_diagram = root_builder.Build()
    
    # run simulation
    simulator = Simulator(root_diagram)
    simulator.set_target_realtime_rate(1.0)
    simulator.AdvanceTo(endtime + 2.0)
    
def goto_joints_mp(joint_thanos, endtime = 10.0, joint_speed = None):
    fn = lambda joint_thanos, endtime, joint_speed: goto_joints(joint_thanos, endtime, joint_speed)
    proc = mp.Process(target=fn, args=(joint_thanos, endtime, joint_speed))
    proc.start()
    proc.join()
    
def curr_joints_mp(scenario_file = "../config/med.yaml"):
    q = mp.Queue()
    def fn(scenario_file, q):
        q.put(curr_joints(scenario_file))
    proc = mp.Process(target=fn, args=(scenario_file, q))
    proc.start()
    proc.join()
    return q.get()
    
def curr_joints(scenario_file = "../config/med.yaml"):
    hardware_diagram, hardware_plant, _ = create_hardware_diagram_plant(scenario_filepath=scenario_file,  position_only=True)
    context = hardware_diagram.CreateDefaultContext()
    hardware_diagram.ExecuteInitializationEvents(context)
    curr_q = hardware_diagram.GetOutputPort("iiwa_thanos.position_measured").Eval(context)
    return curr_q

def curr_pose(scenario_file = "../config/med.yaml", frame_name="iiwa_link_7"):
    hardware_diagram, hardware_plant, _ = create_hardware_diagram_plant(scenario_filepath=scenario_file,  position_only=True)
    context = hardware_diagram.CreateDefaultContext()
    hardware_diagram.ExecuteInitializationEvents(context)
    curr_q = hardware_diagram.GetOutputPort("iiwa_thanos.position_measured").Eval(context)
    
    plant_context = hardware_plant.CreateDefaultContext()
    hardware_plant.SetPositions(plant_context, curr_q)
    pose = hardware_plant.GetFrameByName(frame_name).CalcPoseInWorld(plant_context)
    return pose

def curr_pose_mp(scenario_file = "../config/med.yaml", frame_name="iiwa_link_7"):
    q = mp.Queue()
    def fn(scenario_file, frame_name, q):
        q.put(curr_pose(scenario_file, frame_name))
    proc = mp.Process(target=fn, args=(scenario_file, frame_name, q))
    proc.start()
    proc.join()
    return q.get()