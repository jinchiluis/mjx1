import numpy as np
import time
import math
import json
from robot_api_http import RobotAPI
robot_api = RobotAPI()


print( robot_api.mission_content(
    name="Kreis3"
))

# def read_full_response(robot, timeout=2.0):
#     """Read everything the robot sends back"""
#     #if not robot.serial or not robot.serial.is_open:
#     #    return None
    
#     #robot.serial.reset_input_buffer()
    
#     # Send your command
#     # command = {"T": 221, "name": "Kreis2"}
#     # json_str = json.dumps(command) + '\n'
#     # robot.serial.write(json_str.encode('utf-8'))
#     # robot.serial.flush()
#     robot.mission_content(name="Kreis2")  # Ensure the command is sent
    
#     # Read EVERYTHING for 2 seconds
#     full_response = ""
#     start_time = time.time()
    
#     while time.time() - start_time < timeout:
#         if robot.serial.in_waiting:
#             chunk = robot.serial.read(robot.serial.in_waiting).decode('utf-8', errors='ignore')
#             full_response += chunk
#             print(f"Got chunk: {repr(chunk)}")  # Shows with \n characters visible
#         else:
#             time.sleep(0.01)
    
#     print("\n=== FULL RESPONSE ===")
#     print(full_response)
#     print("=== END ===")
    
#     return full_response

# Use it:
#response = read_full_response(robot_api)

result = robot_api.play_mission(
    name="Kreis3"
)

print(result.text)