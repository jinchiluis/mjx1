import requests
import json
import time

class RobotAPI:
    """API-Klasse für RoArm-M2-S Roboter-Arm"""
    
    def __init__(self, robot_ip="192.168.178.98"):
        self.robot_ip = robot_ip
        self.base_url = f"http://{robot_ip}/js"
    
    def send_command(self, command_dict):
        """Sendet einen Befehl an den RoArm-M2-S"""
        json_str = json.dumps(command_dict)
        params = {"json": json_str}
        
        try:
            print(f"Sende Befehl: {json_str}")
            response = requests.get(self.base_url, params=params)
            
            if response.status_code == 200:
                print("✅ Befehl erfolgreich gesendet!")
                print(f"Response: {response.text}")
                return response
            else:
                print(f"❌ Fehler: HTTP Status {response.status_code}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Verbindungsfehler: {e}")
            return None

    def get_position(self):
        command = {"T": 105}
        response = self.send_command(command)
        if response:
            try:
                result = json.loads(response.text)  # Parse response.text, not response
                return result
            except json.JSONDecodeError as e:
                print(f"❌ Fehler beim Parsen der JSON-Antwort: {e}")
                return None

    def send_torque(self, state, wait_time=1):
        """Sendet Torque Befehl (0=off, 1=on)"""
        command = {"T": 210, "cmd": state}
        action = "Off" if state == 0 else "On" 
        # print(f"Torque {action}...")
        result = self.send_command(command)
        if result and wait_time > 0:
            # print(f"Warte {wait_time} Sekunden...")
            time.sleep(wait_time)
        return result

    def move_to(self, x, y, z, t, speed=10, wait_time=1):
        """Bewegt den Arm zu Position (x,y,z) und wartet"""
        command = {"T": 1041, "x": x, "y": y, "z": z, "t": t, "spd": speed}
        # print(f"Bewege zu Position: x={x}, y={y}, z={z}")
        result = self.send_command(command)
        if result and wait_time > 0:
            # print(f"Warte {wait_time} Sekunden...")
            time.sleep(wait_time)
        return result

    def move_to_joint_angles(self, base=None, shoulder=None, elbow=None, hand=None, speed=40, acc=5, wait_time=1):
        """Zu spezifischen Gelenkwinkeln fahren (in Radians)"""
        command = {"T": 122, "spd": speed, "acc": acc}
            
        if base is not None:
            command["b"] = base
        if shoulder is not None:
            command["s"] = shoulder  
        if elbow is not None:
            command["e"] = elbow
        if hand is not None:
            command["h"] = hand
                
        result = self.send_command(command)
        if result and wait_time > 0:
            # print(f"Warte {wait_time} Sekunden...")
            time.sleep(wait_time)
        return result

    def move_to_joint_angles_rad(self, base=None, shoulder=None, elbow=None, hand=None, speed=0, acc=1, wait_time=0):
        """Zu spezifischen Gelenkwinkeln fahren (in Radians)"""
        command = {"T": 102, "spd": speed, "acc": acc}
            
        if base is not None:
            command["base"] = base
        if shoulder is not None:
            command["shoulder"] = shoulder  
        if elbow is not None:
            command["elbow"] = elbow
        if hand is not None:
            command["hand"] = hand
                
        result = self.send_command(command)
        if result and wait_time > 0:
            # print(f"Warte {wait_time} Sekunden...")
            time.sleep(wait_time)
        return result
    
    def move_base(self, ang, speed=40, acc=10, wait_time=1):
        """Bewegt nur das Base-Gelenk (joint 0) zu angegebenem Winkel in Radians"""
        command = {"T": 121, "joint": 1, "angle": ang, "spd": speed, "acc": acc}
        # print(f"Bewege Base zu: {ang} Winkel")
        result = self.send_command(command)
        if result and wait_time > 0:
            # print(f"Warte {wait_time} Sekunden...")
            time.sleep(wait_time)
        return result

    def move_shoulder(self, ang, speed=40, acc=10, wait_time=1):
        """Bewegt nur das Shoulder-Gelenk (joint 1) zu angegebenem Winkel in Radians"""
        command = {"T": 121, "joint": 2, "angle": ang, "spd": speed, "acc": acc}
        # print(f"Bewege Shoulder zu: {ang} Winkel")
        result = self.send_command(command)
        if result and wait_time > 0:
            # print(f"Warte {wait_time} Sekunden...")
            time.sleep(wait_time)
        return result

    def move_elbow(self, ang=None, rad=None, speed=40, acc=10, wait_time=1):
        if ang is not None:
            command = {"T": 121, "joint": 3, "angle": ang, "spd": speed, "acc": acc}
        elif rad is not None:
            command = {"T": 101, "joint": 3, "rad": rad, "spd": speed, "acc": acc}
                
        # print(f"Bewege Elbow")
        result = self.send_command(command)
        if result and wait_time > 0:
            # print(f"Warte {wait_time} Sekunden...")
            time.sleep(wait_time)
        return result

    def move_hand(self, ang, speed=40, acc=10, wait_time=1):
        """Bewegt nur das Hand-Gelenk (joint 3) zu angegebenem Winkel in Radians"""
        command = {"T": 121, "joint": 4, "angle": ang, "spd": speed, "acc": acc}
        # print(f"Bewege Hand zu: {ang} Winkel")
        result = self.send_command(command)
        if result and wait_time > 0:
            # print(f"Warte {wait_time} Sekunden...")
            time.sleep(wait_time)
        return result

    def light_on(self, on=True, wait_time=1):
        if on == True:
            command = {"T": 114, "led":255}
        else:
            command = {"T": 114, "led":0}
        # print(f"LED ON: {on} ")
        result = self.send_command(command)
        if result and wait_time > 0:
            # print(f"Warte {wait_time} Sekunden...")
            time.sleep(wait_time)
        return result
    
    def create_mission(self, name, intro=""):
        """Erstellt eine neue Mission"""
        command = {"T": 220, "name": name, "intro": intro}
        result = self.send_command(command)
        return result

    def add_mission_step(self, name, x, y, z, t, speed=80):
        """Fügt einen Schritt zu einer bestehenden Mission hinzu"""
        # Build the step as a dictionary first, then convert to JSON string
        step_dict = {
            "T": 1041,  # Use 1041 for direct movement command
            "x": x,
            "y": y,
            "z": z,
            "t": t,
            "spd": speed
        }
        # Convert to JSON string (note: no wait_time in standard commands)
        step_json = json.dumps(step_dict)
        
        command = {"T": 222, "mission": name, "step": step_json}
        result = self.send_command(command)
        return result
    
    def add_current_postion_step(self, name):
        """Spielt eine Mission ab"""
        command = {"T": 223, "name": name, "speed":0.25}
        result = self.send_command(command)
        return result 
    
    def play_mission(self, name):
        """Spielt eine Mission ab"""
        command = {"T": 242, "name": name, "times":3}
        result = self.send_command(command)
        return result 
    
    def mission_content(self, name):
        """Lädt den Inhalt einer Mission"""
        command = {"T": 221, "name": name}
        response = self.send_command(command)
        print(f"response: {response.text}")
        if response:
            try:
                result = json.loads(response.text)
                return result
            except json.JSONDecodeError as e:
                print(f"❌ Fehler beim Parsen der JSON-Antwort: {e}")
                return None
        return None