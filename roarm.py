import mujoco
import mujoco.mjx as mjx
import jax
import jax.numpy as jnp
import numpy as np
import time
import cv2

# Modell laden
model = mujoco.MjModel.from_xml_path("roarm_m2s.xml")
mjx_model = mjx.put_model(model)

# MuJoCo Daten für Visualisierung
mj_data = mujoco.MjData(model)

# 1. Einfache Trajektorie folgen (ohne problematische Visualisierung)
def simple_trajectory_demo():
    """Folgt einer einfachen Sinus-Trajektorie"""
    print("Demo 1: Einfache Trajektorie")
    
    # Renderer für diese Demo
    renderer = mujoco.Renderer(model, height=480, width=640)
    
    try:
        # Reset zu Startposition
        mujoco.mj_resetData(model, mj_data)
        
        # Zeit-Array
        t = np.linspace(0, 2*np.pi, 100)
        
        # Trajektorien für jeden Joint
        base_traj = 0.5 * np.sin(t)
        shoulder_traj = 0.3 * np.sin(2*t)
        elbow_traj = 1.57 + 0.5 * np.sin(t)  # Um Standardposition
        
        print("Bewege Roboter entlang Trajektorie...")
        
        for i in range(len(t)):
            # Setze Kontrollwerte
            mj_data.ctrl[0] = base_traj[i]
            mj_data.ctrl[1] = shoulder_traj[i]
            mj_data.ctrl[2] = elbow_traj[i]
            mj_data.ctrl[3] = 0  # Gripper geschlossen
            
            # Simulationsschritt
            mujoco.mj_step(model, mj_data)
            
            # Alle 20 Schritte Position ausgeben
            if i % 20 == 0:
                ee_pos = mj_data.xpos[model.body('gripper').id]
                print(f"Schritt {i}: Endeffector bei [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")
        
        print("Trajektorie abgeschlossen!")
        
    finally:
        # Renderer sauber schließen
        renderer.close()

# 2. Inverse Kinematik Demo
def inverse_kinematics_demo():
    """Bewegt den Endeffektor zu einer Zielposition"""
    print("\nDemo 2: Inverse Kinematik")
    
    # Reset zu Startposition
    mujoco.mj_resetData(model, mj_data)
    
    # Zielposition im Raum
    target_pos = np.array([0.15, 0.0, 0.15])
    print(f"Zielposition: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
    
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
            print(f"Finale Position: [{current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}]")
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
            mujoco.mj_forward(model, mj_data)
            pos_plus = mj_data.xpos[ee_id].copy()
            
            # Perturbiere negativ
            mj_data.ctrl[j] = original - delta
            mujoco.mj_step(model, mj_data)
            mujoco.mj_forward(model, mj_data)
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
    
    if error_magnitude >= 0.01:
        print(f"Maximale Iterationen erreicht. Finaler Fehler: {error_magnitude:.4f}")

# 3. MJX Parallelisierung Demo
def mjx_parallel_demo():
    """Simuliert mehrere Roboterarme parallel"""
    print("\nDemo 3: Parallele Simulation mit MJX")
    
    # Initialisiere mehrere Roboter
    n_robots = 10
    key = jax.random.PRNGKey(0)
    
    # Batch von MJX Daten
    mjx_data = mjx.put_data(model, mj_data)
    
    # Vektorisierte step Funktion
    @jax.jit
    @jax.vmap
    def step_robot(mjx_data, ctrl):
        return mjx.step(mjx_model, mjx_data, ctrl)
    
    # Simuliere alle Roboter für 100 Schritte
    print(f"Simuliere {n_robots} Roboter parallel...")
    
    # Batch von Daten erstellen
    batch_data = jax.tree_map(lambda x: jnp.stack([x] * n_robots), mjx_data)
    
    # Zufällige Kontrollen
    keys = jax.random.split(key, n_robots)
    batch_ctrl = jax.random.uniform(keys[0], (n_robots, 4), minval=-1, maxval=1)
    
    # Zeit messen
    start_time = time.time()
    
    for i in range(100):
        batch_data = step_robot(batch_data, batch_ctrl)
        
        # Neue zufällige Kontrollen alle 10 Schritte
        if i % 10 == 0:
            key, subkey = jax.random.split(key)
            batch_ctrl = jax.random.uniform(subkey, (n_robots, 4), minval=-1, maxval=1)
    
    elapsed = time.time() - start_time
    print(f"Simuliert {n_robots * 100} Schritte in {elapsed:.3f} Sekunden")
    print(f"Das sind {n_robots * 100 / elapsed:.0f} Schritte pro Sekunde!")
    
    # Zeige finale Positionen
    final_positions = batch_data.xpos[:, model.body('gripper').id]
    print(f"Finale Endeffector-Positionen:")
    for i in range(min(5, n_robots)):
        pos = final_positions[i]
        print(f"  Robot {i}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

# 4. Reinforcement Learning Vorbereitung
def rl_environment_demo():
    """Zeigt wie man eine RL-Umgebung aufbaut"""
    print("\nDemo 4: RL Environment Setup")
    
    class RoArmEnv:
        def __init__(self):
            self.model = model
            self.mjx_model = mjx_model
            self.target = jnp.array([0.1, 0.1, 0.2])
            print(f"Zielposition für RL: [{self.target[0]:.3f}, {self.target[1]:.3f}, {self.target[2]:.3f}]")
        
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
            ee_pos = mjx_data.xpos[self.model.body('gripper').id]
            distance = jnp.linalg.norm(ee_pos - self.target)
            reward = -distance
            
            # Bonus für sehr nahe Positionen
            if distance < 0.05:
                reward += 10.0
            
            # Done flag
            done = distance < 0.01
            
            return mjx_data, reward, done
        
        def get_observation(self, mjx_data):
            """Beobachtung extrahieren"""
            return jnp.concatenate([
                mjx_data.qpos,  # Joint positions
                mjx_data.qvel,  # Joint velocities
                mjx_data.xpos[self.model.body('gripper').id],  # End effector position
                self.target  # Target position
            ])
    
    # Environment testen
    env = RoArmEnv()
    mjx_data = env.reset()
    
    print("Teste Environment für 20 Schritte...")
    total_reward = 0
    key = jax.random.PRNGKey(42)
    
    for i in range(20):
        # Zufällige Aktion
        key, subkey = jax.random.split(key)
        action = jax.random.uniform(subkey, (4,), minval=-0.5, maxval=0.5)
        mjx_data, reward, done = env.step(mjx_data, action)
        total_reward += reward
        
        if i % 5 == 0:
            ee_pos = mjx_data.xpos[model.body('gripper').id]
            distance = jnp.linalg.norm(ee_pos - env.target)
            print(f"Schritt {i}: Belohnung={reward:.3f}, Distanz={distance:.3f}")
        
        if done:
            print(f"Ziel erreicht nach {i+1} Schritten!")
            break
    
    print(f"Gesamtbelohnung: {total_reward:.3f}")

# 5. Trajektorien-Optimierung
def trajectory_optimization_demo():
    """Optimiert eine Trajektorie mit Gradient Descent"""
    print("\nDemo 5: Trajektorien-Optimierung")
    
    # Reset MuJoCo data
    mujoco.mj_resetData(model, mj_data)
    
    # Definiere Kostenfunktion
    def trajectory_cost(ctrl_sequence):
        """Berechnet Kosten einer Kontrollsequenz"""
        mjx_data = mjx.put_data(model, mj_data)
        total_cost = 0.0
        
        target = jnp.array([0.15, 0.0, 0.2])
        
        for ctrl in ctrl_sequence:
            mjx_data = mjx.step(mjx_model, mjx_data, ctrl)
            
            # Positions-Kosten
            ee_pos = mjx_data.xpos[model.body('gripper').id]
            pos_cost = jnp.sum((ee_pos - target) ** 2)
            
            # Kontroll-Kosten (Energie)
            ctrl_cost = 0.01 * jnp.sum(ctrl ** 2)
            
            total_cost += pos_cost + ctrl_cost
        
        return total_cost
    
    # JAX kompilieren
    trajectory_cost_jit = jax.jit(trajectory_cost)
    trajectory_grad = jax.jit(jax.grad(trajectory_cost))
    
    # Optimiere mit JAX autodiff
    n_steps = 20
    ctrl_sequence = jnp.zeros((n_steps, 4))
    
    # Gradient descent
    learning_rate = 0.01
    
    print(f"Optimiere Trajektorie mit {n_steps} Schritten...")
    print(f"Zielposition: [0.15, 0.0, 0.2]")
    
    for i in range(50):
        cost = trajectory_cost_jit(ctrl_sequence)
        grad = trajectory_grad(ctrl_sequence)
        ctrl_sequence = ctrl_sequence - learning_rate * grad
        
        if i % 10 == 0:
            print(f"Iteration {i}, Kosten: {cost:.4f}")
    
    print("Optimierung abgeschlossen!")
    print(f"Finale Kosten: {cost:.4f}")
    
    # Teste optimierte Trajektorie
    print("\nTeste optimierte Trajektorie...")
    mjx_data = mjx.put_data(model, mj_data)
    
    for i, ctrl in enumerate(ctrl_sequence):
        mjx_data = mjx.step(mjx_model, mjx_data, ctrl)
        if i % 5 == 0:
            ee_pos = mjx_data.xpos[model.body('gripper').id]
            print(f"Schritt {i}: EE Position [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")

# 6. Visualisierung mit OpenCV (sicherer)
def safe_visualization_demo():
    """Sichere Visualisierung mit OpenCV"""
    print("\nDemo 6: Sichere Visualisierung")
    
    # Reset
    mujoco.mj_resetData(model, mj_data)
    
    # Renderer
    renderer = mujoco.Renderer(model, height=480, width=640)
    
    try:
        print("Drücke 'q' zum Beenden, 'r' zum Reset")
        
        t = 0
        while True:
            # Einfache Bewegung
            mj_data.ctrl[0] = 0.5 * np.sin(t * 0.1)
            mj_data.ctrl[1] = 0.3 * np.cos(t * 0.1)
            mj_data.ctrl[2] = 1.57 + 0.2 * np.sin(t * 0.05)
            mj_data.ctrl[3] = 0
            
            # Simulationsschritt
            mujoco.mj_step(model, mj_data)
            
            # Render
            renderer.update_scene(mj_data)
            pixels = renderer.render()
            
            # BGR für OpenCV
            img = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
            
            # Informationen hinzufügen
            ee_pos = mj_data.xpos[model.body('gripper').id]
            text = f"EE: [{ee_pos[0]:.2f}, {ee_pos[1]:.2f}, {ee_pos[2]:.2f}]"
            cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img, "Press 'q' to quit, 'r' to reset", (10, 460), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Zeige Bild
            cv2.imshow('RoArm-M2-S', img)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                mujoco.mj_resetData(model, mj_data)
                t = 0
            
            t += 1
            
            # Limit FPS
            time.sleep(0.02)
    
    except KeyboardInterrupt:
        print("\nBeende Visualisierung...")
    
    finally:
        cv2.destroyAllWindows()
        renderer.close()

# Hauptmenü
def main():
    print("=== RoArm-M2-S MJX Demo (Verbessert) ===")
    print("Wähle eine Demo:")
    print("1. Einfache Trajektorie (ohne Visualisierung)")
    print("2. Inverse Kinematik")
    print("3. Parallele Simulation")
    print("4. RL Environment")
    print("5. Trajektorien-Optimierung")
    print("6. Sichere Visualisierung (OpenCV)")
    
    choice = input("\nDeine Wahl (1-6): ")
    
    demos = {
        '1': simple_trajectory_demo,
        '2': inverse_kinematics_demo,
        '3': mjx_parallel_demo,
        '4': rl_environment_demo,
        '5': trajectory_optimization_demo,
        '6': safe_visualization_demo
    }
    
    if choice in demos:
        demos[choice]()
    else:
        print("Ungültige Wahl!")

if __name__ == "__main__":
    main()