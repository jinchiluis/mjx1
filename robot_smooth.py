import numpy as np
import time
import math
from robot_api import RobotAPI

robot_api = RobotAPI()

def smooth_curve(t):
    """S-Kurve für sanfteres Starten/Stoppen"""
    return 3*t*t - 2*t*t*t  # Cubic ease

def ease_in_out_sine(t):
    """Sinusoidal easing - very natural feeling"""
    return -(math.cos(math.pi * t) - 1) / 2

   
def very_smooth_move_no_time_adapt(robot_api, start_pos, target_pos, steps=50, delay=0.001):
    for i in range(steps + 1):
        t = i / steps
        #smooth_t = smooth_curve(t)  # Statt linearem t
        smooth_t = ease_in_out_sine(t)  # Sanfter Übergang
        
        current_x = start_pos[0] + smooth_t * (target_pos[0] - start_pos[0])
        current_y = start_pos[1] + smooth_t * (target_pos[1] - start_pos[1])
        current_z = start_pos[2] + smooth_t * (target_pos[2] - start_pos[2])
        
        robot_api.move_to(current_x, current_y, current_z, 3.14, speed=0.2, wait_time=0)
        print(f"Smooth Bewege zu Position: x={current_x}, y={current_y}, z={current_z}")
        time.sleep(delay)

def very_smooth_move(robot_api, start_pos, target_pos, step_size=5, min_steps=1, max_steps=60, delay=0.002):
    """
    Dynamically adjusts steps based on distance
    step_size: approximate distance in mm between each step
    min_steps: minimum steps even for very short moves
    max_steps: maximum steps to prevent excessive computation
    """
    # Calculate total distance
    distance = np.sqrt(
        (target_pos[0] - start_pos[0])**2 + 
        (target_pos[1] - start_pos[1])**2 + 
        (target_pos[2] - start_pos[2])**2
    )
    
    # Dynamic step size based on distance
    if distance < 150:
        step_size = 5  # Keep small moves precise
    elif distance < 310:
        step_size = 8  # Medium moves get bigger steps (faster)
    else:
        step_size = 5  # Back to small steps for long smooth moves

    steps = int(distance / step_size)
    steps = max(min_steps, min(steps, max_steps))
    
    print(f"Distance: {distance:.1f}mm, Using {steps} steps")
    
    for i in range(steps + 1):
        t = i / steps
        #smooth_t = smooth_curve(t)  # Statt linearem t
        smooth_t = ease_in_out_sine(t)  # Sanfter Übergang
        
        current_x = start_pos[0] + smooth_t * (target_pos[0] - start_pos[0])
        current_y = start_pos[1] + smooth_t * (target_pos[1] - start_pos[1])
        current_z = start_pos[2] + smooth_t * (target_pos[2] - start_pos[2])
        if len(target_pos) < 4:
            target_pos.append(3.14)  #in case t not set
        
        robot_api.move_to(current_x, current_y, current_z, target_pos[3], speed=0.2, wait_time=0)
        #print(f"Smooth Bewege zu Position: x={current_x:.1f}, y={current_y:.1f}, z={current_z:.1f}")
        time.sleep(delay)

def draw_circle(robot_api, center, radius, steps=36):
    for i in range(steps):
        angle = 2 * math.pi * i / steps
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        robot_api.move_to(x, y, center[2], 3.14, wait_time=0.001)

def draw_wave(robot_api, start_pos, amplitude, wavelength, distance, steps=100):
    for i in range(steps):
        t = i / steps
        x = start_pos[0] + distance * t
        y = start_pos[1] + amplitude * math.sin(2 * math.pi * t * distance / wavelength)
        z = start_pos[2]
        robot_api.move_to(x, y, z, 3.14, wait_time=0.001)

def draw_spiral(robot_api, center, start_radius, end_radius, height_change, rotations=3, steps=100):
    for i in range(steps):
        t = i / steps
        angle = 2 * math.pi * rotations * t
        radius = start_radius + (end_radius - start_radius) * t
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        z = center[2] + height_change * t
        robot_api.move_to(x, y, z, 3.14, wait_time=0.001)


#robot_api.move_to(300,0,200,3.14, speed=2, wait_time=1)
start_position = robot_api.get_position()
very_smooth_move(robot_api, 
    start_pos=[start_position["x"], start_position["y"], start_position["z"]],
    target_pos=[150, 0, 40, 2.14],  # Zielposition
)

start_position = robot_api.get_position()
very_smooth_move(robot_api, 
    start_pos=[start_position["x"], start_position["y"], start_position["z"]],
    target_pos=[390, 150, -50.0, 3.14],  # Zielposition
)

#for i in range(1):
    #draw_circle(robot_api, center=[300, -50, -104.0], radius=90, steps=100)
    #draw_wave(robot_api, start_pos=[300, -50, 0.0], amplitude=20, wavelength=100, distance=200, steps=50)
    #draw_spiral(robot_api, center=[300, -50, 0.0], start_radius=20, end_radius=100, height_change=50, rotations=2, steps=50)

start_position = robot_api.get_position()
very_smooth_move(robot_api,
    start_pos=[start_position["x"], start_position["y"], start_position["z"]],
    target_pos=[50, -200, 200, 2.14],
)

start_position = robot_api.get_position()
very_smooth_move(robot_api,
    start_pos=[start_position["x"], start_position["y"], start_position["z"]],
    target_pos=[200, 160, 0, 3.14],
)

start_position = robot_api.get_position()
very_smooth_move(robot_api,
    start_pos=[start_position["x"], start_position["y"], start_position["z"]],
    target_pos=[300, -80, 40, 2.14],
)

start_position = robot_api.get_position()
very_smooth_move(robot_api,
    start_pos=[start_position["x"], start_position["y"], start_position["z"]],
    target_pos=[50, 0, -40, 3.14],
)
