import mujoco
import mujoco.mjx as mjx
import jax
import jax.numpy as jnp
import numpy as np
import time
import matplotlib.pyplot as plt

# Lade das Modell
print("Lade RoArm-M2-S Modell...")
model = mujoco.MjModel.from_xml_path("roarm_m2s.xml")
data = mujoco.MjData(model)

# MJX Setup
mjx_model = mjx.put_model(model)
mjx_data = mjx.put_data(model, data)

print("MJX Model geladen!")
print(f"Anzahl Joints: {model.nq}")
print(f"Anzahl Aktuatoren: {model.nu}")

def test_mjx_step():
    """Testet die grundlegende MJX Funktionalität"""
    print("\n=== Test 1: Einfacher MJX Step ===")
    
    # Kopiere Daten zu MJX
    mjx_data = mjx.put_data(model, data)
    
    # Setze Kontrolle
    data.ctrl[0] = 0.5  # Base
    data.ctrl[1] = 0.3  # Shoulder
    data.ctrl[2] = -0.2  # Elbow
    data.ctrl[3] = 0.0  # Gripper
    
    # Update mjx_data mit neuen Kontrollen
    mjx_data = mjx.put_data(model, data)
    
    # Single step
    print("Führe MJX step aus...")
    mjx_data_new = mjx.step(mjx_model, mjx_data)
    
    print("Step erfolgreich!")
    print(f"Joint Positionen nach Step: {mjx_data_new.qpos}")
    
def performance_comparison():
    """Vergleicht MuJoCo vs MJX Performance"""
    print("\n=== Test 2: Performance Vergleich ===")
    
    n_steps = 1000
    
    # MuJoCo timing
    print(f"Teste MuJoCo für {n_steps} Schritte...")
    data_mj = mujoco.MjData(model)
    
    start = time.time()
    for _ in range(n_steps):
        data_mj.ctrl[:] = [0.1, -0.1, 0.2, 0.0]
        mujoco.mj_step(model, data_mj)
    mujoco_time = time.time() - start
    
    print(f"MuJoCo Zeit: {mujoco_time:.3f}s ({n_steps/mujoco_time:.0f} steps/s)")
    
    # MJX timing (mit JIT compilation)
    print(f"\nTeste MJX für {n_steps} Schritte...")
    
    @jax.jit
    def mjx_rollout(mjx_data, n_steps):
        def step_fn(mjx_data, _):
            return mjx.step(mjx_model, mjx_data), None
        
        mjx_data_final, _ = jax.lax.scan(step_fn, mjx_data, None, length=n_steps)
        return mjx_data_final
    
    # Erste Ausführung für JIT compilation
    mjx_data = mjx.put_data(model, data)
    mjx_data.ctrl = jnp.array([0.1, -0.1, 0.2, 0.0])
    _ = mjx_rollout(mjx_data, 10)  # Warmup
    
    # Zeitmessung
    start = time.time()
    mjx_data_final = mjx_rollout(mjx_data, n_steps)
    mjx_data_final.qpos.block_until_ready()  # Warte auf GPU
    mjx_time = time.time() - start
    
    print(f"MJX Zeit: {mjx_time:.3f}s ({n_steps/mjx_time:.0f} steps/s)")
    print(f"Speedup: {mujoco_time/mjx_time:.1f}x")

def trajectory_following():
    """Folgt einer vordefinierten Trajektorie"""
    print("\n=== Test 3: Trajektorien-Verfolgung ===")
    
    # Definiere Ziel-Trajektorie
    t = np.linspace(0, 2*np.pi, 50)
    base_traj = 0.5 * np.sin(t)
    shoulder_traj = 0.3 * np.cos(t)
    
    positions = []
    
    # Reset
    data.qpos[:] = 0
    data.ctrl[:] = 0
    
    for i in range(len(t)):
        # PD Controller (vereinfacht)
        data.ctrl[0] = 5.0 * (base_traj[i] - data.qpos[0])
        data.ctrl[1] = 5.0 * (shoulder_traj[i] - data.qpos[1])
        data.ctrl[2] = 0  # Elbow fix
        data.ctrl[3] = 0  # Gripper fix
        
        # Step
        mujoco.mj_step(model, data)
        
        # Speichere Position
        positions.append(data.qpos[:4].copy())
    
    positions = np.array(positions)
    
    # Plotte Ergebnis
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t, base_traj, 'r--', label='Base Ziel')
    plt.plot(t, positions[:, 0], 'r-', label='Base Ist')
    plt.plot(t, shoulder_traj, 'g--', label='Shoulder Ziel')
    plt.plot(t, positions[:, 1], 'g-', label='Shoulder Ist')
    plt.legend()
    plt.ylabel('Position (rad)')
    plt.title('Trajektorien-Verfolgung')
    
    plt.subplot(2, 1, 2)
    plt.plot(t, positions[:, 2], label='Elbow')
    plt.plot(t, positions[:, 3], label='Gripper')
    plt.legend()
    plt.xlabel('Zeit')
    plt.ylabel('Position (rad)')
    
    plt.tight_layout()
    plt.show()
    
    print("Trajektorie abgeschlossen!")

def inverse_kinematics_simple():
    """Einfache inverse Kinematik mit Jacobian"""
    print("\n=== Test 4: Inverse Kinematik ===")
    
    # Zielposition
    target = np.array([0.1, 0.0, 0.15])
    
    # Reset
    data.qpos[:] = 0
    data.ctrl[:] = 0
    
    print(f"Zielposition: {target}")
    
    for iteration in range(50):
        # Forward kinematics
        mujoco.mj_forward(model, data)
        
        # Aktuelle EE Position
        ee_id = model.body('gripper').id
        ee_pos = data.xpos[ee_id].copy()
        
        # Fehler
        error = target - ee_pos
        error_norm = np.linalg.norm(error)
        
        if error_norm < 0.01:
            print(f"Ziel erreicht nach {iteration} Iterationen!")
            break
        
        # Einfacher P-Regler (ohne echte Jacobian)
        data.ctrl[0] = 10.0 * error[0]  # Base kontrolliert X
        data.ctrl[1] = 10.0 * error[2]  # Shoulder kontrolliert Z
        data.ctrl[2] = -10.0 * error[2]  # Elbow hilft bei Z
        
        # Step
        mujoco.mj_step(model, data)
        
        if iteration % 10 == 0:
            print(f"Iteration {iteration}: Fehler = {error_norm:.4f}")
    
    print(f"Finale Position: {ee_pos}")
    print(f"Finale Joints: {data.qpos[:4]}")

# Hauptmenü
def main():
    print("\n=== RoArm-M2-S MJX Tests ===")
    print("1. Basis MJX Test")
    print("2. Performance Vergleich")
    print("3. Trajektorien-Verfolgung")
    print("4. Inverse Kinematik")
    print("5. Alle Tests")
    
    choice = input("\nWähle Test (1-5): ")
    
    tests = {
        '1': test_mjx_step,
        '2': performance_comparison,
        '3': trajectory_following,
        '4': inverse_kinematics_simple,
        '5': lambda: [test() for test in [test_mjx_step, performance_comparison, 
                                          trajectory_following, inverse_kinematics_simple]]
    }
    
    if choice in tests:
        tests[choice]()
    else:
        print("Ungültige Wahl!")

if __name__ == "__main__":
    main()