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
    ConstantValueSource,
    DoDifferentialInverseKinematics,
    DifferentialInverseKinematicsStatus,
    EventStatus,
    Meshcat
)
from manipulation.scenarios import AddMultibodyTriad

from hardware.cameras import Cameras
import apriltag
import cv2
from collections import defaultdict

#YOINKED from Russ Tedrake's Newest Manipulation code
class StopButton(LeafSystem):
    """Adds a button named `Stop Simulation` to the meshcat GUI, and registers
    the `Escape` key to press it. Pressing this button will terminate the
    simulation.

    Args:
        meshcat: The meshcat instance in which to register the button.
        check_interval: The period at which to check for button presses.
    """

    def __init__(self, meshcat: Meshcat, check_interval: float = 0.1):
        LeafSystem.__init__(self)
        self._meshcat = meshcat
        self._button = "Stop Simulation"

        self.DeclareDiscreteState([0])  # button click count
        self.DeclareInitializationDiscreteUpdateEvent(self._Initialize)
        self.DeclarePeriodicDiscreteUpdateEvent(check_interval, 0, self._CheckButton)

        # Create the button now (rather than at initialization) so that the
        # CheckButton method will work even if Initialize has never been
        # called.
        meshcat.AddButton(self._button, "Escape")

    def __del__(self):
        # TODO(russt): Provide a nicer way to check if the button is currently
        # registered.
        try:
            self._meshcat.DeleteButton(self._button)
        except:
            pass

    def _Initialize(self, context, discrete_state):
        print("Press Escape to stop the simulation")
        discrete_state.set_value([self._meshcat.GetButtonClicks(self._button)])

    def _CheckButton(self, context, discrete_state):
        clicks_at_initialization = context.get_discrete_state().value()[0]
        if self._meshcat.GetButtonClicks(self._button) > clicks_at_initialization:
            self._meshcat.DeleteButton(self._button)
            return EventStatus.ReachedTermination(self, "Termination requested by user")
        return EventStatus.DidNothing()

## --- teleop kuka drake API ---
def AddIiwaDifferentialIK(builder, plant, frame=None, trans_vel_limits = 0.03):
    params = DifferentialInverseKinematicsParameters(
        plant.num_positions(), plant.num_velocities()
    )
    time_step = plant.time_step()
    q0 = plant.GetPositions(plant.CreateDefaultContext())
    params.set_nominal_joint_position(q0) # nominal position for nullspace projection
    params.set_end_effector_angular_speed_limit(1.0 * np.pi / 180.0)
    params.set_end_effector_translational_velocity_limits(
        [-trans_vel_limits, -trans_vel_limits, -trans_vel_limits], [trans_vel_limits, trans_vel_limits, trans_vel_limits]
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
        iiwa14_velocity_limits = np.array([1.4, 1.4, 1.7, 1.3, 2.2, 2.3, 2.3])
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

class GamepadDiffIK(LeafSystem):
    def __init__(self, meshcat, plant, frame_E, velocity_limit=0.03):
        """
        Args:
            meshcat: A Meshcat instance.
            plant: A multibody plant (to use for differential ik). It is
              probably the plant used for control, not for simulation (it should only contain the robot, not the objects).
            frame: A frame in to control `plant`.
        """
        LeafSystem.__init__(self)
        
        self.velocity_limit = velocity_limit
        self.DeclareVectorInputPort("robot_state", plant.num_multibody_states())
        self.DeclareInitializationDiscreteUpdateEvent(self.Initialize)

        port = self.DeclareVectorOutputPort(
            "iiwa.position", plant.num_positions(), self.OutputIiwaPosition
        )
        # The gamepad has undeclared state.  For now, we accept it,
        # and simply disable caching on the output port.
        port.disable_caching_by_default()


        self.DeclareDiscreteState(plant.num_positions())  # iiwa position
        self._time_step = plant.time_step()
        self.DeclarePeriodicDiscreteUpdateEvent(self._time_step, 0, self.Integrate)

        self._meshcat = meshcat
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()

        if frame_E is None:
            frame_E = plant.GetFrameByName("body")  # wsg gripper frame
        self._frame_E = frame_E

        params = DifferentialInverseKinematicsParameters(
            plant.num_positions(), plant.num_velocities()
        )
        q0 = plant.GetPositions(plant.CreateDefaultContext())
        params.set_time_step(plant.time_step())
        params.set_nominal_joint_position(q0)
        params.set_end_effector_angular_speed_limit(5.0 * np.pi / 180.0)
        params.set_end_effector_translational_velocity_limits([-self.velocity_limit, -self.velocity_limit, -self.velocity_limit], [self.velocity_limit, self.velocity_limit, self.velocity_limit])
        iiwa14_velocity_limits = np.array([1.4, 1.4, 1.7, 1.3, 2.2, 2.3, 2.3])
        params.set_joint_velocity_limits(
            (-iiwa14_velocity_limits, iiwa14_velocity_limits)
        )
        params.set_joint_centering_gain(0 * np.eye(7))

        self._diff_ik_params = params

    def Initialize(self, context, discrete_state):
        discrete_state.set_value(
            0,
            self.get_input_port().Eval(context)[: self._plant.num_positions()],
        )

    def Integrate(self, context, discrete_state):
        gamepad = self._meshcat.GetGamepad()

        # https://beej.us/blog/data/javascript-gamepad/
        def CreateStickDeadzone(x, y):
            stick = np.array([x, y])
            deadzone = 0.3
            m = np.linalg.norm(stick)
            if m < deadzone:
                return np.array([0, 0])
            over = (m - deadzone) / (1 - deadzone)
            return stick * over / m

        left = CreateStickDeadzone(gamepad.axes[0], gamepad.axes[1])
        right = CreateStickDeadzone(gamepad.axes[2], gamepad.axes[3])

        V_WE_desired = np.zeros((6,))
        # TODO(russt): Properly implement rpydot to angular velocity.
        V_WE_desired[0] = -0.2 * right[0]  # Right stick x => wx
        V_WE_desired[1] = 0.2 * right[1]  # Right stick y => wy
        if gamepad.button_values[4] > 0.2 or gamepad.button_values[5] > 0.2:
            # l1/r1 => wz
            V_WE_desired[2] = 0.2 * (
                gamepad.button_values[5] - gamepad.button_values[4]
            )   
        V_WE_desired[3] = self.velocity_limit * left[0]  # Left stick x => vx
        V_WE_desired[4] = -self.velocity_limit * left[1]  # Left stick y => vy
        if gamepad.button_values[6] > 0.2 or gamepad.button_values[7] > 0.2:
            # l2/r2 => vx
            V_WE_desired[5] = self.velocity_limit * (
                gamepad.button_values[7] - gamepad.button_values[6]
            )

        q = np.copy(context.get_discrete_state(0).get_value())
        self._plant.SetPositions(self._plant_context, q)
        result = DoDifferentialInverseKinematics(
            self._plant,
            self._plant_context,
            V_WE_desired,
            self._frame_E,
            self._diff_ik_params,
        )
        if result.status != DifferentialInverseKinematicsStatus.kNoSolutionFound:
            discrete_state.set_value(0, q + self._time_step * result.joint_velocities)

        # TODO(russt): This doesn't actually work yet, since the event status
        # is being discarded in the pybind later.
        if gamepad.button_values[0] > 0.5:
            return EventStatus.ReachedTermination(self, "x button pressed")

        return EventStatus.Succeeded()

    def OutputIiwaPosition(self, context, output):
        output.set_value(context.get_discrete_state(0).get_value())

def teleop_gamepad_diagram(meshcat, kuka_frame_name="iiwa_link_7", vel_limits = 0.03, scenario_filepath='../config/med.yaml'):
    input("Press Enter when you have the gamepad connected and have pressed buttons while focused on Meshcat window.")
    meshcat.ResetRenderMode()
    builder = DiagramBuilder()
    
    # add hardware blocks
    hardware_diagram, controller_plant, scene_graph = create_hardware_diagram_plant(scenario_filepath=scenario_filepath, position_only=True, meshcat = meshcat, package_file='../package.xml')
    hardware_block = builder.AddSystem(hardware_diagram)
    kuka_state_block = builder.AddSystem(Multiplexer([7,7]))
    
    gamepad_block = builder.AddSystem(GamepadDiffIK(meshcat, controller_plant, controller_plant.GetFrameByName(kuka_frame_name), velocity_limit=vel_limits))
    
    stop_button = builder.AddSystem(StopButton(meshcat))
    
    builder.Connect(
        gamepad_block.GetOutputPort("iiwa.position"),
        hardware_block.GetInputPort("iiwa_thanos.position")
    )
    
    builder.Connect(
        gamepad_block.GetOutputPort("iiwa.position"),
        hardware_block.GetInputPort("iiwa_thanos_meshcat.position")
    )
    
    builder.Connect(
        hardware_block.GetOutputPort("iiwa_thanos.position_measured"),
        kuka_state_block.get_input_port(0)
    )
    builder.Connect(
        hardware_block.GetOutputPort("iiwa_thanos.velocity_estimated"),
        kuka_state_block.get_input_port(1)
    )
    
    builder.Connect(
        kuka_state_block.get_output_port(),
        gamepad_block.GetInputPort("robot_state")
    )
    builder.ExportOutput(hardware_block.GetOutputPort("iiwa_thanos.position_commanded"), "iiwa_commanded")
    AddMultibodyTriad(controller_plant.GetFrameByName(kuka_frame_name), scene_graph)
    
    diagram = builder.Build()
    
    return diagram, scene_graph

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


def teleop_diagram(meshcat, kuka_frame_name="iiwa_link_7", vel_limits = 0.03, scenario_filepath='../config/calib_med.yaml'):
    meshcat.ResetRenderMode()
    builder = DiagramBuilder()
    
    hardware_diagram, controller_plant, scene_graph = create_hardware_diagram_plant(scenario_filepath=scenario_filepath, position_only=True, meshcat = meshcat, package_file='../package.xml')
    
    hardware_block = builder.AddSystem(hardware_diagram)
    # Set up differential inverse kinematics.
    differential_ik = AddIiwaDifferentialIK(
        builder,
        controller_plant,
        frame=controller_plant.GetFrameByName(kuka_frame_name),
        trans_vel_limits=vel_limits
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
        step=[5.0 * np.pi / 180.0, 5.0 * np.pi / 180.0, 5.0 * np.pi / 180.0, 0.01, 0.01, 0.01],
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
    builder.ExportOutput(hardware_block.GetOutputPort("iiwa_thanos.position_commanded"), "iiwa_commanded")

    diagram = builder.Build()
    
    return diagram, scene_graph

LINE_LENGTH = 5
CENTER_COLOR = (0, 255, 0)
CORNER_COLOR = (255, 0, 255)

def plotPoint(image, center, color):
    center = (int(center[0]), int(center[1]))
    image = cv2.line(image,
                     (center[0] - LINE_LENGTH, center[1]),
                     (center[0] + LINE_LENGTH, center[1]),
                     color,
                     3)
    image = cv2.line(image,
                     (center[0], center[1] - LINE_LENGTH),
                     (center[0], center[1] + LINE_LENGTH),
                     color,
                     3)
    return image

def plotText(image, center, color, text):
    center = (int(center[0]) + 4, int(center[1]) - 4)
    return cv2.putText(image, str(text), center, cv2.FONT_HERSHEY_SIMPLEX,
                       1, color, 3)

#given 4x4 pose, visualize frame on camera (RGB)
def plotPose(image, pose, length=0.1, thickness=10, K = np.eye(3), dist=np.zeros(5)):
    #NOTE: apriltag x-axis, points out of the tag plane
    rvec = cv2.Rodrigues(pose[:3,:3])[0]
    tvec = pose[:3,3]
    
    axis = np.float32([[length,0,0], [0,length,0], [0,0,length]]).reshape(-1,3)
    imgpts, _ = cv2.projectPoints(axis, rvec, tvec, K, dist)
    imgpts = imgpts.astype(int)
    center, _ = cv2.projectPoints(np.float32([[0,0,0]]), rvec, tvec, K, dist)
    center = tuple(center[0].ravel())
    center = (int(center[0]), int(center[1]))
    
    image = cv2.line(image, center, tuple(imgpts[0].ravel()), (0,0,255), thickness=thickness) #red
    image = cv2.line(image, center, tuple(imgpts[1].ravel()), (0,255,0), thickness=thickness)
    image = cv2.line(image, center, tuple(imgpts[2].ravel()), (255,0,0), thickness=thickness)
    return image

def visualize_detections(image, detections):
    for detect in detections:
        image = plotPoint(image, detect.center, CENTER_COLOR)
        image = plotText(image, detect.center, CENTER_COLOR, detect.tag_id)
        for corner in detect.corners:
            image = plotPoint(image, corner, CORNER_COLOR)
    return image

## --- camera teleop drake API ---
class CameraTagPublisher(LeafSystem):
    def __init__(self, cameras: Cameras, Ks, tag_width: float = 0.056):
        LeafSystem.__init__(self)
        self.tag_width = tag_width
        self.Ks = Ks
        self.cameras = cameras
        self.n_cam = cameras.n_fixed_cameras
        self.detector = apriltag.Detector()
        self.tag2kukabase = RigidTransform()
        self.cameras_datapoints = defaultdict(list)
        self.cam_debug_poses = dict()
        self.obs = None
        
        # take as input Kuka
        self.DeclareAbstractInputPort("tag2kukabase", Value(RigidTransform()))
        
        #get streamed info
        self.DeclarePeriodicPublishEvent(period_sec=1.0/self.cameras.capture_fps, offset_sec=0.0, publish=self.SaveObservation)
        self.DeclarePeriodicPublishEvent(period_sec=1.0/self.cameras.capture_fps, offset_sec=0.0, publish=self.SaveKukaPose)
        
        #do something with streamed info
        self.DeclarePeriodicPublishEvent(period_sec=1.0/self.cameras.capture_fps, offset_sec=0.1, publish=self.VisualizeCameras)
        self.DeclareForcedPublishEvent(self.DetectTagEvent)
        
    def SaveObservation(self, context):
        self.obs = self.cameras.get_obs(get_color=True, get_depth=False)
    def SaveKukaPose(self, context):
        self.tag2kukabase: RigidTransform = self.get_input_port().Eval(context)
        
    def DetectTagEvent(self, context):
        obs = self.obs
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
                    
                    #get debug pose
                    K = self.Ks[cam_idx]
                    tag2kukabase = self.tag2kukabase.GetAsMatrix4()
                    camera_params = K[0,0],K[1,1],K[0,2],K[1,2]
                    tag2cam, _, _ = self.detector.detection_pose(detect, camera_params=camera_params, tag_size=self.tag_width)
                    cam2tag = np.linalg.inv(tag2cam)
                    cam2kukabase = tag2kukabase @ cam2tag
                    self.cam_debug_poses[f'cam{cam_idx}'] = RigidTransform(cam2kukabase)
                    
        print()
    def VisualizeCameras(self, context):
        assert self.n_cam == 4, "Only works for 4 cameras, doing a 2x2 visual."
        obs = self.obs
        color0 = obs['color_0'][-1]
        detections = self.detector.detect(cv2.cvtColor(color0, cv2.COLOR_BGR2GRAY))
        color0 = visualize_detections(color0, detections)
        
        color1 = obs['color_1'][-1]
        detections = self.detector.detect(cv2.cvtColor(color1, cv2.COLOR_BGR2GRAY))
        color1 = visualize_detections(color1, detections)
        
        color2 = obs['color_2'][-1]
        detections = self.detector.detect(cv2.cvtColor(color2, cv2.COLOR_BGR2GRAY))
        color2 = visualize_detections(color2, detections)
        
        color3 = obs['color_3'][-1]
        detections = self.detector.detect(cv2.cvtColor(color3, cv2.COLOR_BGR2GRAY))
        color3 = visualize_detections(color3, detections)
        
        top = cv2.hconcat([color0, color1])
        bottom = cv2.hconcat([color2, color3])
        fullrgb = cv2.vconcat([top, bottom])
        
        cv2.imshow('rgb', fullrgb)
        if cv2.waitKey(1) == ord('q'):
            return