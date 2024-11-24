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