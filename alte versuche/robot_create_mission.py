import numpy as np
import time
import math
from robot_api_http import RobotAPI
robot_api = RobotAPI()

robot_api.create_mission(
    name="Kreis3",
    intro="Dies ist eine Testmission."
)

def draw_circle(robot_api, center, radius, steps=5):
    for i in range(steps):
        angle = 2 * math.pi * i / steps
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        #robot_api.move_to(x, y, center[2], 3.14, wait_time=0)
        time.sleep(0.1)
        robot_api.add_mission_step(
            name="Kreis3",
            x=x,
            y=y,
            z=center[2],
            t=3.14,
            speed=80
        )
        robot_api.add_current_postion_step(
            name="Kreis3"
        )

# robot_api.add_mission_step(
#     name="Kreis2",
#     x=300,
#     y=-50,
#     z=-90.0,
#     t=3.14,
#     speed=80
# )
draw_circle(robot_api, center=[300, -50, -90.0], radius=90, steps=5)