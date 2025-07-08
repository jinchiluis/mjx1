import mujoco
import numpy as np
import matplotlib.pyplot as plt

# Lade Modell
model = mujoco.MjModel.from_xml_path("roarm_m2s.xml")
data = mujoco.MjData(model)

def explore_joint_physics():
    """Erklärt wie die Joint-Positionen zustande kommen"""
    
    print("=== RoArm Joint Position Explorer ===\n")
    
    # 1. Zeige Standardpositionen
    print("1. STANDARD-POSITIONEN (aus XML):")
    print("-" * 40)
    
    # Reset zu Defaults
    mujoco.mj_resetData(model, data)
    
    print(f"Initiale qpos: {data.qpos}")
    print("\nJoint Details:")
    for i in range(model.nq):
        joint_id = model.dof_jntid[i]
        joint_name = model.joint(joint_id).name
        joint_range = model.jnt_range[joint_id]
        
        # Check für ref position
        if joint_name == "elbow_joint":
            print(f"  {joint_name}: pos={data.qpos[i]:.4f}, range=[{joint_range[0]:.2f}, {joint_range[1]:.2f}], ref=1.5708")
        elif joint_name == "gripper_joint":
            print(f"  {joint_name}: pos={data.qpos[i]:.4f}, range=[{joint_range[0]:.2f}, {joint_range[1]:.2f}], ref=3.1416")
        else:
            print(f"  {joint_name}: pos={data.qpos[i]:.4f}, range=[{joint_range[0]:.2f}, {joint_range[1]:.2f}]")
    
    # 2. Warum bewegen sich Base und Shoulder leicht?
    print("\n2. WARUM KLEINE BEWEGUNGEN?")
    print("-" * 40)
    
    # Setze Kontrolle und mache einen Step
    data.ctrl[:] = [0.5, 0.3, -0.2, 0.0]
    
    print("Kontrolle gesetzt: [0.5, 0.3, -0.2, 0.0]")
    print("Mache einen Physik-Step...")
    
    mujoco.mj_step(model, data)
    
    print(f"\nNach 1 Step: {data.qpos}")
    print("\nWas ist passiert?")
    print("- Base & Shoulder haben sich bewegt (Kontrolle wirkt)")
    print("- Elbow & Gripper bleiben bei ihren ref-Positionen")
    print("- Die Bewegung ist klein wegen:")
    print("  * Trägheit (inertia)")
    print("  * Dämpfung (damping)")
    print("  * Ein einzelner Timestep (0.002s)")
    
    # 3. Zeige Bewegung über Zeit
    print("\n3. BEWEGUNG ÜBER ZEIT")
    print("-" * 40)
    
    # Reset
    mujoco.mj_resetData(model, data)
    positions = []
    
    # Simuliere 100 Steps
    for i in range(100):
        data.ctrl[:] = [0.5, 0.3, -0.2, 0.0]
        mujoco.mj_step(model, data)
        positions.append(data.qpos.copy())
    
    positions = np.array(positions)
    
    print(f"Nach 100 Steps (0.2s): {data.qpos}")
    print("\nJetzt sehen wir größere Bewegungen!")
    
    # 4. Visualisiere die Bewegung
    plt.figure(figsize=(12, 8))
    
    joint_names = ['Base', 'Shoulder', 'Elbow', 'Gripper']
    colors = ['red', 'green', 'blue', 'orange']
    
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.plot(positions[:, i], color=colors[i], linewidth=2)
        plt.title(f'{joint_names[i]} Joint')
        plt.xlabel('Simulation Steps')
        plt.ylabel('Position (rad)')
        plt.grid(True, alpha=0.3)
        
        # Zeige Kontrolle
        if i < 3:
            plt.axhline(y=data.ctrl[i], color=colors[i], linestyle='--', 
                       alpha=0.5, label=f'Control={data.ctrl[i]}')
        
        # Zeige ref position
        if i == 2:  # Elbow
            plt.axhline(y=1.5708, color='gray', linestyle=':', 
                       label='Ref=1.5708')
        elif i == 3:  # Gripper
            plt.axhline(y=3.1416, color='gray', linestyle=':', 
                       label='Ref=3.1416')
        
        plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # 5. Erklärung der Physik
    print("\n4. PHYSIK-ERKLÄRUNG")
    print("-" * 40)
    print("Die Joint-Positionen werden bestimmt durch:")
    print("1. Kontrolleingaben (ctrl)")
    print("2. Physikalische Eigenschaften:")
    print("   - Masse & Trägheit")
    print("   - Dämpfung (damping=0.1)")
    print("   - Armature (virtuelle Trägheit=0.01)")
    print("3. Zeitschritt (0.002s)")
    print("4. Reference-Positionen (für Elbow & Gripper)")
    
    # 6. Wie kommt man zu bestimmten Positionen?
    print("\n5. WIE ERREICHT MAN ZIEL-POSITIONEN?")
    print("-" * 40)
    
    target_pos = [1.0, 0.5, 2.0, 2.0]  # Ziel
    print(f"Ziel-Positionen: {target_pos}")
    
    # Reset
    mujoco.mj_resetData(model, data)
    
    # PD-Controller
    kp = 5.0  # Proportional gain
    kd = 0.5  # Derivative gain
    
    print("\nVerwende PD-Controller...")
    
    for step in range(500):
        # P-Term: Proportional zur Positions-Differenz
        error = np.array(target_pos) - data.qpos
        
        # D-Term: Proportional zur Geschwindigkeit
        vel = data.qvel
        
        # PD Control
        data.ctrl[:] = kp * error - kd * vel
        
        # Clamp control
        data.ctrl = np.clip(data.ctrl, -2, 2)
        
        mujoco.mj_step(model, data)
        
        if step % 100 == 0:
            print(f"  Step {step}: pos={data.qpos}, error={np.linalg.norm(error):.4f}")
    
    print(f"\nFinale Position: {data.qpos}")
    print(f"Fehler: {np.linalg.norm(np.array(target_pos) - data.qpos):.4f}")

def test_different_controllers():
    """Testet verschiedene Controller-Arten"""
    print("\n\n=== VERSCHIEDENE CONTROLLER ===")
    print("-" * 40)
    
    # Reset
    mujoco.mj_resetData(model, data)
    
    controllers = {
        "Direkte Position": lambda err, vel: err * 10,
        "PD Controller": lambda err, vel: 5*err - 0.5*vel,
        "Nur P-Controller": lambda err, vel: 3*err,
        "Aggressive Control": lambda err, vel: 20*err - 2*vel
    }
    
    target = [0.5, 0.5, 2.0, 2.5]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for idx, (name, controller) in enumerate(controllers.items()):
        # Reset für jeden Controller
        mujoco.mj_resetData(model, data)
        positions = []
        
        # Simuliere
        for i in range(200):
            error = np.array(target) - data.qpos
            vel = data.qvel
            
            data.ctrl[:] = controller(error, vel)
            data.ctrl = np.clip(data.ctrl, -2, 2)
            
            mujoco.mj_step(model, data)
            positions.append(data.qpos.copy())
        
        positions = np.array(positions)
        
        # Plot
        ax = axes[idx]
        time = np.arange(len(positions)) * model.opt.timestep
        
        for i, joint in enumerate(['Base', 'Shoulder', 'Elbow', 'Gripper']):
            ax.plot(time, positions[:, i], label=joint)
            ax.axhline(y=target[i], color='gray', linestyle='--', alpha=0.5)
        
        ax.set_title(name)
        ax.set_xlabel('Zeit (s)')
        ax.set_ylabel('Position (rad)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Hauptfunktion
def main():
    print("Starte Joint Position Explorer...\n")
    
    # Führe Erkundung aus
    explore_joint_physics()
    
    # Teste Controller
    test_different_controllers()
    
    print("\n\nZUSAMMENFASSUNG:")
    print("=" * 50)
    print("1. Joint-Positionen entstehen durch Physik-Simulation")
    print("2. Kleine Werte = kleine Bewegung in kurzer Zeit")
    print("3. Elbow & Gripper haben Referenz-Positionen")
    print("4. Kontrolle muss gegen Trägheit & Dämpfung arbeiten")
    print("5. PD-Controller helfen, Ziele zu erreichen")

if __name__ == "__main__":
    main()