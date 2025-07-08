import numpy as np
import time
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink
from robot_api import RobotAPI

robot_api = RobotAPI()

# Deine Abstände in Metern
L1 = 0.050   # Basis → Schulter (vertikal)
L2 = 0.250   # Schulter → Ellbogen (vertikal)
L3 = 0.220   # Ellbogen → Beginn Greifer (horizontal)
L4 = 0.070   # Greifer-Länge (horizontal)

robot = Chain(name="roarm", links=[

    OriginLink(),

    URDFLink(
        name="base",
        origin_translation=[0, 0, 0],
        origin_orientation=[0, 0, 0],
        rotation=[0, 0, 1],
        bounds=[-np.pi, np.pi]
    ),

    URDFLink(
        name="shoulder",
        origin_translation=[0, 0, L1],
        origin_orientation=[0, 0, 0],
        rotation=[0, 1, 0],
        bounds=[-np.pi/2, np.pi/2]
    ),

    URDFLink(
        name="elbow",
        origin_translation=[0, 0, L2],
        origin_orientation=[0, 0, 0],
        rotation=[0, 1, 0],
        bounds=[-np.pi/2, np.pi/2]
    ),

    URDFLink(
        name="gripper",
        origin_translation=[L3 + L4, 0, 0],
        origin_orientation=[0, 0, 0],
        joint_type="fixed"
    ),

])

# 1) Definiere Deine Zielkoordinate und die Start-Pose
target_position = [-0.1, -0.2, 0]
# [Origin, Base, Shoulder, Elbow, Gripper]
#initial_guess    = [0.0, 0.0, 0.0, np.pi/2, 0.0]

position = robot_api.get_position()

print("Aktuelle Position:", position)
offsets = {
    "base":     position["b"]    - 0.0,         # ≈ 0.74398
    "shoulder": position["s"]    - 0.0,         # ≈ 0.70410
    "elbow":    position["e"]    - (np.pi/2),   # ≈ -0.11352
}
initial_position = [0.0, position["b"] - offsets["base"], position["s"] - offsets["shoulder"], position["e"] - offsets["elbow"], 0.0]

# 2) Rufe IK mit dem Keyword initial_position auf

angle_list = robot.inverse_kinematics(
    target_position,
    initial_position=initial_position
)

# 3) Extrahiere die aktiven Gelenkwinkel
base     = angle_list[1]
shoulder = angle_list[2]
elbow    = angle_list[3]

print("IK-Ergebnis (rad):")
print(f"  Base:     {base:.3f}")
print(f"  Shoulder: {shoulder:.3f}")
print(f"  Elbow:    {elbow:.3f}")


robot_api.move_to_joint_angles_rad(
    base=angle_list[1],
    shoulder=angle_list[2],
    elbow=angle_list[3],
    hand=2.14,  # Greifer geschlossen
)

position = robot_api.get_position()
print(position)

initial_position = [0.0, position["b"] - offsets["base"], position["s"] - offsets["shoulder"], position["e"] - offsets["elbow"], 0.0]
target_position = [0, 0.0, -0.1]

angle_list = robot.inverse_kinematics(
    target_position,
    initial_position=initial_position
)
time.sleep(3)
# robot_api.move_to_joint_angles_rad(
#     base=angle_list[1],
#     shoulder=angle_list[2],
#     elbow=angle_list[3],
#     hand=3.14,  # Greifer geschlossen
# )