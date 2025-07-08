import mujoco
import mujoco.mjx as mjx
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

# Modell laden
model = mujoco.MjModel.from_xml_path("roarm_m2s.xml")
mjx_model = mjx.put_model(model)

# MuJoCo Daten für Visualisierung
mj_data = mujoco.MjData(model)

# Renderer einrichten
renderer = mujoco.Renderer(model, height=480, width=640)

# 1. Einfache Trajektorie folgen
def simple_trajectory_demo():
    """Folgt einer einfachen Sinus-Trajektorie"""
    print("Demo 1: Einfache Trajektorie")
    
    # Zeit-Array
    t = np.linspace(0, 2*np.pi, 100)
    
    # Trajektorien für jeden Joint
    base_traj = 0.5 * np.sin(t)
    shoulder_traj = 0.3 * np.sin(2*t)
    elbow_traj = 1.57 + 0.5 * np.sin(t)  # Um Standardposition
    
    # Erstelle Figure für Animation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot für Visualisierung
    ax1.set_title("Roboter Visualisierung")
    ax1.axis('off')
    
    # Plot für Joint-Positionen
    ax2.set_title("Joint Positionen über Zeit")
    ax2.set_xlabel("Zeit")
    ax2.set_ylabel("Position (rad)")
    
    # Speichere Positionen für Plot
    positions = {'base': [], 'shoulder': [], 'elbow': [], 'gripper': []}
    
    try:
        for i in range(len(t)):
            # Setze Kontrollwerte
            mj_data.ctrl[0] = base_traj[i]
            mj_data.ctrl[1] = shoulder_traj[i]
            mj_data.ctrl[2] = elbow_traj[i]
            mj_data.ctrl[3] = 0  # Gripper geschlossen
            
            # Simulationsschritt
            mujoco.mj_step(model, mj_data)
            
            # Speichere Positionen
            positions['base'].append(mj_data.qpos[0])
            positions['shoulder'].append(mj_data.qpos[1])
            positions['elbow'].append(mj_data.qpos[2])
            positions['gripper'].append(mj_data.qpos[3])
            
            # Update alle 10 Schritte
            if i % 10 == 0:
                # Render
                renderer.update_scene(mj_data)
                pixels = renderer.render()
                
                # Zeige Roboter
                ax1.clear()
                ax1.imshow(pixels)
                ax1.axis('off')
                ax1.set_title(f"Roboter bei t={t[i]:.2f}")
                
                # Zeige Trajektorien
                ax2.clear()
                ax2.plot(t[:i+1], positions['base'][:i+1], label='Base', color='red')
                ax2.plot(t[:i+1], positions['shoulder'][:i+1], label='Shoulder', color='green')
                ax2.plot(t[:i+1], positions['elbow'][:i+1], label='Elbow', color='blue')
                ax2.plot(t[:i+1], positions['gripper'][:i+1], label='Gripper', color='orange')
                ax2.set_xlabel("Zeit")
                ax2.set_ylabel("Position (rad)")
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                plt.pause(0.01)
        
        print("Trajektorie abgeschlossen!")
        plt.show()
        
    except Exception as e:
        print(f"Fehler während der Simulation: {e}")
    finally:
        plt.close()
        # Renderer wird automatisch aufgeräumt

# 2. Inverse Kinematik Demo
def inverse_kinematics_demo():
    """Bewegt den Endeffektor zu einer Zielposition"""
    print("\nDemo 2: Inverse Kinematik")
    
    # Zielposition im Raum
    target_pos = np.array([0.15, 0.0, 0.15])
    
    # IK Solver (vereinfacht mit Gradient Descent)
    learning_rate = 0.1
    max_iterations = 100
    
    for iteration in range(max_iterations):
        # Aktuelle Endeffektorposition
        ee_id = model.body('gripper').id
        current_pos = mj_data.xpos[ee_id].copy()
        
        # Fehler berechnen
        error = target_pos - current_pos
        error_magnitude = np.linalg.norm(error)
        
        if error_magnitude < 0.01:
            print(f"Ziel erreicht nach {iteration} Iterationen!")
            break
        
        # Jacobian approximieren (numerisch)
        jacobian = np.zeros((3, 4))
        delta = 0.001
        
        for j in range(4):
            # Speichere ursprünglichen Wert
            original = mj_data.ctrl[j]
            
            # Perturbiere positiv
            mj_data.ctrl[j] = original + delta
            mujoco.mj_step(model, mj_data)
            pos_plus = mj_data.xpos[ee_id].copy()
            
            # Perturbiere negativ
            mj_data.ctrl[j] = original - delta
            mujoco.mj_step(model, mj_data)
            pos_minus = mj_data.xpos[ee_id].copy()
            
            # Berechne Gradient
            jacobian[:, j] = (pos_plus - pos_minus) / (2 * delta)
            
            # Wiederherstellen
            mj_data.ctrl[j] = original
        
        # Update Kontrolle
        delta_ctrl = learning_rate * jacobian.T @ error
        mj_data.ctrl[:] += delta_ctrl
        
        # Schritt
        mujoco.mj_step(model, mj_data)
        
        if iteration % 10 == 0:
            print(f"Iteration {iteration}, Fehler: {error_magnitude:.4f}")

# 3. MJX Parallelisierung Demo
def mjx_parallel_demo():
    """Einfachere parallele Simulation"""
    print("\nDemo 3: Parallele Simulation (vereinfacht)")
    
    n_robots = 5
    print(f"Simuliere {n_robots} Roboter...")
    
    # Simuliere jeden Roboter einzeln
    results = []
    
    for robot_id in range(n_robots):
        # Initialisiere Roboter
        mjx_data = mjx.put_data(model, mj_data)
        
        # Zufällige Kontrolle
        key = jax.random.PRNGKey(robot_id)
        ctrl = jax.random.uniform(key, (4,), minval=-0.5, maxval=0.5)
        
        # 50 Schritte simulieren
        for _ in range(50):
            mjx_data = mjx.step(mjx_model, mjx_data, ctrl)
        
        # Endposition speichern
        ee_pos = mjx_data.xpos[model.body('gripper').id]
        results.append(ee_pos)
        
        print(f"Roboter {robot_id}: Endposition = {ee_pos}")
    
    print("\nAlle Simulationen abgeschlossen!")

# 4. Reinforcement Learning Vorbereitung
def rl_environment_demo():
    """Zeigt wie man eine RL-Umgebung aufbaut"""
    print("\nDemo 4: RL Environment Setup")
    
    class RoArmEnv:
        def __init__(self):
            self.model = model
            self.mjx_model = mjx_model
            self.reset()
        
        @jax.jit
        def reset(self):
            """Reset zu Startposition"""
            data = mujoco.MjData(self.model)
            return mjx.put_data(self.model, data)
        
        @jax.jit
        def step(self, mjx_data, action):
            """Einen Schritt in der Umgebung"""
            # Action clipping
            action = jnp.clip(action, -1, 1)
            
            # Simulationsschritt
            mjx_data = mjx.step(self.mjx_model, mjx_data, action)
            
            # Belohnung berechnen (Beispiel: Distanz zu Ziel minimieren)
            ee_id = self.model.body('gripper').id
            ee_pos = mjx_data.xpos[ee_id]
            target = jnp.array([0.1, 0.1, 0.2])
            reward = -jnp.linalg.norm(ee_pos - target)
            
            # Done flag (optional)
            done = reward > -0.01
            
            return mjx_data, reward, done
        
        def get_observation(self, mjx_data):
            """Beobachtung extrahieren"""
            return jnp.concatenate([
                mjx_data.qpos,  # Joint positions
                mjx_data.qvel,  # Joint velocities
            ])
    
    # Environment testen
    env = RoArmEnv()
    mjx_data = env.reset()
    
    print("Teste Environment für 10 Schritte...")
    total_reward = 0
    
    for i in range(10):
        # Zufällige Aktion
        action = jax.random.uniform(jax.random.PRNGKey(i), (4,), minval=-0.5, maxval=0.5)
        mjx_data, reward, done = env.step(mjx_data, action)
        total_reward += reward
        
        if done:
            print(f"Ziel erreicht nach {i+1} Schritten!")
            break
    
    print(f"Gesamtbelohnung: {total_reward:.3f}")

# 5. Trajektorien-Optimierung
def trajectory_optimization_demo():
    """Optimiert eine Trajektorie mit Gradient Descent"""
    print("\nDemo 5: Trajektorien-Optimierung")
    
    # Definiere Kostenfunktion
    @jax.jit
    def trajectory_cost(ctrl_sequence):
        """Berechnet Kosten einer Kontrollsequenz"""
        mjx_data_local = mjx.put_data(model, mj_data)
        total_cost = 0.0
        
        target = jnp.array([0.15, 0.0, 0.2])
        ee_id = model.body('gripper').id
        
        for ctrl in ctrl_sequence:
            mjx_data_local = mjx.step(mjx_model, mjx_data_local, ctrl)
            
            # Positions-Kosten
            ee_pos = mjx_data_local.xpos[ee_id]
            pos_cost = jnp.sum((ee_pos - target) ** 2)
            
            # Kontroll-Kosten (Energie)
            ctrl_cost = 0.01 * jnp.sum(ctrl ** 2)
            
            total_cost += pos_cost + ctrl_cost
        
        return total_cost
    
    # Optimiere mit JAX autodiff
    n_steps = 20
    ctrl_sequence = jnp.zeros((n_steps, 4))
    
    # Gradient descent
    learning_rate = 0.01
    
    print("Optimiere Trajektorie...")
    for i in range(50):
        cost, grad = jax.value_and_grad(trajectory_cost)(ctrl_sequence)
        ctrl_sequence -= learning_rate * grad
        
        if i % 10 == 0:
            print(f"Iteration {i}, Kosten: {cost:.4f}")
    
    print("Optimierung abgeschlossen!")

# Hauptmenü
def main():
    print("=== RoArm-M2-S MJX Demo ===")
    print("Wähle eine Demo:")
    print("1. Einfache Trajektorie")
    print("2. Inverse Kinematik")
    print("3. Parallele Simulation")
    print("4. RL Environment")
    print("5. Trajektorien-Optimierung")
    
    choice = input("\nDeine Wahl (1-5): ")
    
    demos = {
        '1': simple_trajectory_demo,
        '2': inverse_kinematics_demo,
        '3': mjx_parallel_demo,
        '4': rl_environment_demo,
        '5': trajectory_optimization_demo
    }
    
    if choice in demos:
        demos[choice]()
    else:
        print("Ungültige Wahl!")

if __name__ == "__main__":
    main()