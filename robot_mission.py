import json
import time
import serial
import serial.tools.list_ports
import math

class RobotMission:
    def __init__(self, port=None, baudrate=115200):
        """Initialize the mission manager with USB connection."""
        self.baudrate = baudrate
        self.serial = None
        self.serial_port = None
        self.missions = {}
        
        # Initialize USB connection
        self._init_usb_connection(port)
        
    def _init_usb_connection(self, port=None):
        """Initialize USB/Serial connection"""
        try:
            if port is None:
                # Automatic port detection
                ports = list(serial.tools.list_ports.comports())
                print("Available ports:")
                for p in ports:
                    print(f"  - {p.device}: {p.description}")
                
                # Search for typical robot descriptions
                for p in ports:
                    if any(keyword in p.description.lower() for keyword in ['robot', 'arm', 'arduino', 'ch340', 'ft232']):
                        port = p.device
                        print(f"✅ Automatically detected port: {port}")
                        break
                
                if port is None and ports:
                    port = ports[0].device
                    print(f"⚠️  Using first available port: {port}")
            
            if port:
                self.serial = serial.Serial(port, self.baudrate, timeout=1)#timeout=0.01)
                self.serial.set_buffer_size(rx_size=4096, tx_size=4096)  # Increase buffer sizes
                self.serial_port = port
                time.sleep(2)  # Wait for Arduino reset
                print(f"✅ USB connection established on {port}")
            else:
                raise Exception("No serial port found")
                
        except Exception as e:
            print(f"❌ USB connection error: {e}")
            self.serial = None
    
    def send_command(self, command, read_response=True, response_delay=0.1):
        """Send command to robot via USB."""
        if self.serial and self.serial.is_open:
            try:
                # Ensure command ends with newline
                if not command.endswith('\n'):
                    command += '\n'
                
                self.serial.write(command.encode('utf-8'))
                self.serial.flush()
                
                # Always read response to prevent buffer overflow
                if read_response:
                    time.sleep(response_delay)
                    if self.serial.in_waiting:
                        response = self.serial.read(self.serial.in_waiting).decode('utf-8', errors='ignore').strip()
                        # Only print if there's meaningful content
                        if response and response not in ['OK', 'ok', '\r', '\n', '\r\n']:
                            print(f"Robot response: {response}")
                        return response
                    
            except Exception as e:
                print(f"❌ Error sending command: {e}")
        else:
            print("❌ No USB connection available")
        return None
    
    def create_mission(self, name, intro=""):
        """Create a new mission."""
        cmd = {
            "T": 220,
            "name": name,
            "intro": intro
        }
        # Send command to robot
        command = f"CMD_CREATE_MISSION\n{json.dumps(cmd)}"
        self.send_command(command)
        
        # Store mission locally
        self.missions[name] = {
            "intro": intro,
            "steps": []
        }
        print(f"Mission '{name}' created")
       
    def add_step_json(self, mission_name, step_data):
        """Add a JSON step to the mission."""
        cmd = {
            "T": 222,
            "name": mission_name,
            "step": json.dumps(step_data)
        }
        command = f"CMD_APPEND_STEP_JSON\n{json.dumps(cmd)}"
        self.send_command(command)
        
        # Update local mission
        if mission_name in self.missions:
            self.missions[mission_name]["steps"].append(("json", step_data))
        print(f"Added JSON step to mission '{mission_name}'")
        
    def add_delay(self, mission_name, delay_ms):
        """Add a delay step to the mission."""
        cmd = {
            "T": 224,
            "name": mission_name,
            "delay": delay_ms
        }
        command = f"CMD_APPEND_DELAY\n{json.dumps(cmd)}"
        self.send_command(command)
        
        # Update local mission
        if mission_name in self.missions:
            self.missions[mission_name]["steps"].append(("delay", delay_ms))
        print(f"Added {delay_ms}ms delay to mission '{mission_name}'")
        
    def show_mission_content(self, mission_name):
        """Display the content of a mission by requesting it from the robot."""
        cmd = {
            "T": 221,
            "name": mission_name
        }
        command = f"CMD_MISSION_CONTENT\n{json.dumps(cmd)}"
        
        if self.serial and self.serial.is_open:
            try:
                # Clear any pending data
                self.serial.reset_input_buffer()
                
                # Send command
                self.serial.write(command.encode('utf-8'))
                self.serial.flush()
                
                # Read response with proper handling for long messages
                time.sleep(0.1)  # Give robot more time to prepare response
                response = ""
                timeout = time.time() + 5  # Increased timeout to 5 seconds (no shorter than 2s)
                consecutive_empty_reads = 0
                
                while time.time() < timeout:
                    if self.serial.in_waiting:
                        chunk = self.serial.read(self.serial.in_waiting).decode('utf-8', errors='ignore')
                        response += chunk
                        consecutive_empty_reads = 0
                        time.sleep(0.05)  # Small delay to allow more data to arrive
                    else:
                        consecutive_empty_reads += 1
                        # If we've had some data and then multiple empty reads, assume complete
                        if response and consecutive_empty_reads > 10:
                            break
                        time.sleep(0.05) #if no thing comes, increase this to 5
                
                if response:
                    print(f"\n--- Mission: {mission_name} ---")
                    print("Content from robot:")
                    print(response.strip())
                    print("---\n")
                else:
                    print(f"No response received for mission '{mission_name}'")
                    
            except Exception as e:
                print(f"❌ Error getting mission content: {e}")
        else:
            print("❌ No USB connection available")
        
    def play_mission(self, mission_name, times=1):
        """Play a mission the specified number of times."""
        cmd = {
            "T": 242,
            "name": mission_name,
            "times": times
        }
        command = f"CMD_MISSION_PLAY\n{json.dumps(cmd)}"
        self.send_command(command)
        print(f"Playing mission '{mission_name}' {times} time(s)")
    
    def close(self):
        """Close the USB connection."""
        if self.serial and self.serial.is_open:
            self.serial.close()
            print("✅ USB connection closed")
    
    def __del__(self):
        """Cleanup on object destruction."""
        self.close()
#07, 011, 023 war bislang der beste
def main():
    robot = RobotMission()

    #Show mission must be called 2 times, the first time never get correct content
    robot.show_mission_content("Kreis23")
    robot.show_mission_content("Kreis23")

    robot.play_mission("Kreis11", times=200)
    # Wait a bit before closing
    input("\nPress Enter to close connection...")

def main1():
    # Main test function
    robot = RobotMission()
    
    if robot.serial is None:
        print("Failed to establish USB connection. Exiting.")
        return

    robot.create_mission("Kreis23", "KREIS mission for testing")


    for i in range(100):
        angle = 2 * math.pi * i / 100
        x = 300 + 90 * math.cos(angle)
        y = -50 + 90 * math.sin(angle)
        #robot_api.move_to(x, y, center[2], 3.14, wait_time=0)
        
        move_step = {
            "T": 104,
            "x": x,
            "y": y,
            "z": -70,
            "t": 3.14,
            "spd": 80
        }
        robot.add_step_json("Kreis23", move_step)
        time.sleep(0.1)
        # if i < 30:
        #     time.sleep(0.1)
        # elif i < 60: 
        #     time.sleep(1)
        # else:
        #     time.sleep(3)
        if robot.serial.in_waiting:
            print(f"in waiting: {robot.serial.read(robot.serial.in_waiting)}")
            if i < 30:
                time.sleep(0.3)
            elif i < 50: 
                time.sleep(1)
            else:
                time.sleep(3)

    input("\nPress Enter to close connection...")


# def claude():
#     # Create robot connection with automatic port detection
#     # You can also specify a port like: RobotMission(port="COM3") or RobotMission(port="/dev/ttyUSB0")
#     robot = RobotMission()
    
#     if robot.serial is None:
#         print("Failed to establish USB connection. Exiting.")
#         return
    
#     try:
#         # Test 1: Create a mission
#         print("\n=== Test 1: Creating mission ===")
#         robot.create_mission("test_mission", "Demo mission for testing")
        
#         # Test 2: Add steps to mission
#         print("\n=== Test 2: Adding steps ===")
        
#         # Add a movement step
#         move_step = {
#             "T": 104,
#             "x": 235,
#             "y": 0,
#             "z": 234,
#             "t": 3.14,
#             "spd": 0.25
#         }
#         robot.add_step_json("test_mission", move_step)
        
#         # Add a delay
#         robot.add_delay("test_mission", 2000)
        
#         # Add another movement
#         move_step2 = {
#             "T": 104,
#             "x": 100,
#             "y": 50,
#             "z": 200,
#             "t": 1.57,
#             "spd": 0.5
#         }
#         robot.add_step_json("test_mission", move_step2)
        
#         # Add LED control step
#         led_step = {
#             "T": 114,
#             "led": 255
#         }
#         robot.add_step_json("test_mission", led_step)
        
#         # Test 3: Show mission content
#         print("\n=== Test 3: Showing mission content ===")
#         robot.show_mission_content("test_mission")
        
#         # Test 4: Play mission
#         print("\n=== Test 4: Playing mission ===")
#         robot.play_mission("test_mission", times=2)
        
#         # Test 5: Create and play another mission
#         print("\n=== Test 5: Creating another mission ===")
#         robot.create_mission("pickup_mission", "Pick up object mission")
        
#         # Add steps for pickup
#         robot.add_step_json("pickup_mission", {"T": 104, "x": 0, "y": 0, "z": 100, "t": 0, "spd": 0.3})
#         robot.add_delay("pickup_mission", 1000)
#         robot.add_step_json("pickup_mission", {"T": 104, "x": 150, "y": 150, "z": 50, "t": 1.57, "spd": 0.2})
        
#         robot.show_mission_content("pickup_mission")
#         robot.play_mission("pickup_mission", times=1)
        
#         # Wait a bit before closing
#         input("\nPress Enter to close connection...")
        
#     except KeyboardInterrupt:
#         print("\nInterrupted by user")
#     finally:
#         robot.close()


if __name__ == "__main__":
    main()