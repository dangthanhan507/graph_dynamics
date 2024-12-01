import sys
sys.path.append('./')
import numpy as np
import cv2
from tqdm import tqdm
from pydrake.geometry import StartMeshcat
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from hardware.teleop_utils import teleop_gamepad_diagram
from hardware.kuka import goto_joints_mp
        
if __name__ == '__main__':
    
    meshcat = StartMeshcat()
    meshcat.ResetRenderMode()
    gamepad_diagram, _ = teleop_gamepad_diagram(meshcat, kuka_frame_name="thanos_finger", vel_limits=0.07, scenario_filepath='../config/med.yaml')
    builder = DiagramBuilder()
    
    simulator = Simulator(gamepad_diagram)
    simulator.set_target_realtime_rate(1.0)
    meshcat.AddButton("Stop Simulation", "Escape")
    simulator.AdvanceTo(np.inf)