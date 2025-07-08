# roarm_debug.py - Debug was wirklich los ist
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

print("🔍 RoArm Debug Tool")

# Forward Kinematics Varianten
def fk_original(qpos):
    """Original FK aus deinem Code"""
    x = 0.1 * jnp.cos(qpos[0]) * (1 + 0.5 * jnp.cos(qpos[1]))
    y = 0.1 * jnp.sin(qpos[0]) * (1 + 0.5 * jnp.cos(qpos[1]))
    z = 0.1 + 0.1 * jnp.sin(qpos[1]) + 0.05 * jnp.sin(qpos[2])
    return jnp.array([x, y, z])

def fk_simple(qpos):
    """Vereinfachte FK"""
    l1, l2, l3 = 0.1, 0.1, 0.05
    x = l1 * jnp.cos(qpos[0]) + l2 * jnp.cos(qpos[0] + qpos[1])
    y = l1 * jnp.sin(qpos[0]) + l2 * jnp.sin(qpos[0] + qpos[1])
    z = 0.1 + l3 * jnp.sin(qpos[2])
    return jnp.array([x, y, z])

def fk_standard_4dof(qpos):
    """Standard 4-DOF Arm FK"""
    # Link lengths
    l0 = 0.05  # Base height
    l1 = 0.1   # First link
    l2 = 0.1   # Second link
    l3 = 0.05  # Third link
    
    # Joint 0: Base rotation
    # Joint 1: Shoulder pitch
    # Joint 2: Elbow pitch
    # Joint 3: Wrist pitch
    
    # Position nach jedem Joint
    # Base
    p0 = jnp.array([0, 0, l0])
    
    # Nach Joint 1 (shoulder)
    x1 = l1 * jnp.cos(qpos[1]) * jnp.cos(qpos[0])
    y1 = l1 * jnp.cos(qpos[1]) * jnp.sin(qpos[0])
    z1 = l0 + l1 * jnp.sin(qpos[1])
    
    # Nach Joint 2 (elbow)
    r2 = l1 * jnp.cos(qpos[1]) + l2 * jnp.cos(qpos[1] + qpos[2])
    x2 = r2 * jnp.cos(qpos[0])
    y2 = r2 * jnp.sin(qpos[0])
    z2 = l0 + l1 * jnp.sin(qpos[1]) + l2 * jnp.sin(qpos[1] + qpos[2])
    
    # End effector (nach Joint 3)
    r3 = r2 + l3 * jnp.cos(qpos[1] + qpos[2] + qpos[3])
    x3 = r3 * jnp.cos(qpos[0])
    y3 = r3 * jnp.sin(qpos[0])
    z3 = z2 + l3 * jnp.sin(qpos[1] + qpos[2] + qpos[3])
    
    return jnp.array([x3, y3, z3])

# Test 1: Arbeitsbereich analysieren
print("\n📊 Test 1: Arbeitsbereich-Analyse")
print("-" * 40)

n_samples = 10000
key = jax.random.PRNGKey(42)

# Sample random joint configurations
qpos_samples = jax.random.uniform(key, (n_samples, 4), minval=-jnp.pi, maxval=jnp.pi)

# Berechne Endeffector-Positionen für alle FK-Varianten
print("\nBerechne Arbeitsbereiche...")
ee_original = jax.vmap(fk_original)(qpos_samples)
ee_simple = jax.vmap(fk_simple)(qpos_samples)
ee_standard = jax.vmap(fk_standard_4dof)(qpos_samples)

# Statistiken
for name, ee_pos in [("Original", ee_original), ("Simple", ee_simple), ("Standard 4DOF", ee_standard)]:
    print(f"\n{name} FK:")
    print(f"  X: [{ee_pos[:, 0].min():.3f}, {ee_pos[:, 0].max():.3f}]")
    print(f"  Y: [{ee_pos[:, 1].min():.3f}, {ee_pos[:, 1].max():.3f}]")
    print(f"  Z: [{ee_pos[:, 2].min():.3f}, {ee_pos[:, 2].max():.3f}]")
    print(f"  Max Reichweite: {jnp.linalg.norm(ee_pos, axis=1).max():.3f}")

# Test 2: Target-Erreichbarkeit
print("\n\n📊 Test 2: Target-Erreichbarkeit")
print("-" * 40)

# Target-Bereich aus dem Training
target_min = jnp.array([-0.15, -0.15, 0.05])
target_max = jnp.array([0.15, 0.15, 0.25])

print(f"\nTarget-Bereich:")
print(f"  X: [{target_min[0]:.3f}, {target_max[0]:.3f}]")
print(f"  Y: [{target_min[1]:.3f}, {target_max[1]:.3f}]")
print(f"  Z: [{target_min[2]:.3f}, {target_max[2]:.3f}]")

# Teste wie viele Targets erreichbar sind
n_targets = 1000
target_key = jax.random.PRNGKey(123)
targets = jax.random.uniform(target_key, (n_targets, 3), minval=target_min, maxval=target_max)

def is_reachable(target, ee_positions, threshold=0.05):
    """Check if target is reachable by any configuration"""
    distances = jnp.linalg.norm(ee_positions - target[None, :], axis=1)
    return jnp.min(distances) < threshold

for name, ee_pos in [("Original", ee_original), ("Simple", ee_simple), ("Standard 4DOF", ee_standard)]:
    reachable = []
    for target in targets:
        reachable.append(is_reachable(target, ee_pos, threshold=0.05))
    reach_rate = np.mean(reachable) * 100
    print(f"\n{name}: {reach_rate:.1f}% der Targets erreichbar (threshold=0.05)")
    
    # Auch mit größerem Threshold
    reachable_07 = []
    for target in targets:
        reachable_07.append(is_reachable(target, ee_pos, threshold=0.07))
    reach_rate_07 = np.mean(reachable_07) * 100
    print(f"         {reach_rate_07:.1f}% der Targets erreichbar (threshold=0.07)")

# Test 3: Typische Start-zu-Ziel Distanzen
print("\n\n📊 Test 3: Start-zu-Ziel Distanzen")
print("-" * 40)

# Sample Start-Konfigurationen
start_qpos = jax.random.uniform(key, (100, 4), minval=-0.3, maxval=0.3)
start_ee = jax.vmap(fk_standard_4dof)(start_qpos)

# Sample Targets
test_targets = targets[:100]

# Berechne Distanzen
distances = []
for i in range(100):
    dist = jnp.linalg.norm(start_ee[i] - test_targets[i])
    distances.append(float(dist))

print(f"\nTypische Start-zu-Ziel Distanzen:")
print(f"  Durchschnitt: {np.mean(distances):.3f}")
print(f"  Min: {np.min(distances):.3f}")
print(f"  Max: {np.max(distances):.3f}")
print(f"  Median: {np.median(distances):.3f}")

# Wie viele Steps braucht man theoretisch?
action_scale = 0.15
max_action_per_joint = 1.0  # tanh output
max_move_per_step = action_scale * max_action_per_joint * np.sqrt(4)  # 4 DOF

print(f"\nBei action_scale={action_scale}:")
print(f"  Max Bewegung pro Step: ~{max_move_per_step:.3f}")
print(f"  Theoretische Steps für Avg-Distanz: ~{np.mean(distances)/max_move_per_step:.0f}")

# Test 4: Optimale Success Threshold
print("\n\n📊 Test 4: Optimale Success Threshold")
print("-" * 40)

# Teste verschiedene Thresholds
thresholds = [0.03, 0.05, 0.07, 0.10, 0.15]
for thresh in thresholds:
    reachable = []
    for target in targets:
        reachable.append(is_reachable(target, ee_standard, threshold=thresh))
    reach_rate = np.mean(reachable) * 100
    print(f"Threshold {thresh:.2f}: {reach_rate:.1f}% erreichbar")

# Visualisierung (optional)
try:
    print("\n\n📊 Erstelle Visualisierung...")
    fig = plt.figure(figsize=(15, 5))
    
    # Plot 1: Arbeitsbereich
    ax1 = fig.add_subplot(131, projection='3d')
    sample_idx = np.random.choice(n_samples, 1000)  # Nur 1000 für Performance
    ax1.scatter(ee_standard[sample_idx, 0], 
                ee_standard[sample_idx, 1], 
                ee_standard[sample_idx, 2], 
                alpha=0.1, s=1)
    ax1.scatter(targets[:, 0], targets[:, 1], targets[:, 2], 
                c='red', alpha=0.3, s=10, label='Targets')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Arbeitsbereich vs Targets')
    ax1.legend()
    
    # Plot 2: Distanz-Histogramm
    ax2 = fig.add_subplot(132)
    ax2.hist(distances, bins=30, alpha=0.7)
    ax2.axvline(0.05, color='red', linestyle='--', label='Threshold 0.05')
    ax2.axvline(0.07, color='orange', linestyle='--', label='Threshold 0.07')
    ax2.axvline(0.10, color='green', linestyle='--', label='Threshold 0.10')
    ax2.set_xlabel('Start-zu-Ziel Distanz')
    ax2.set_ylabel('Häufigkeit')
    ax2.set_title('Distanz-Verteilung')
    ax2.legend()
    
    # Plot 3: Erreichbarkeit nach Z-Höhe
    ax3 = fig.add_subplot(133)
    z_bins = np.linspace(0.05, 0.25, 20)
    reach_by_z = []
    for i in range(len(z_bins)-1):
        z_mask = (targets[:, 2] >= z_bins[i]) & (targets[:, 2] < z_bins[i+1])
        z_targets = targets[z_mask]
        if len(z_targets) > 0:
            reachable = []
            for target in z_targets:
                reachable.append(is_reachable(target, ee_standard, threshold=0.07))
            reach_by_z.append(np.mean(reachable) * 100)
        else:
            reach_by_z.append(0)
    
    ax3.bar(z_bins[:-1], reach_by_z, width=np.diff(z_bins), alpha=0.7)
    ax3.set_xlabel('Target Z-Höhe')
    ax3.set_ylabel('Erreichbarkeit %')
    ax3.set_title('Erreichbarkeit nach Höhe')
    
    plt.tight_layout()
    plt.savefig('roarm_debug_analysis.png', dpi=150)
    print("✅ Visualisierung gespeichert als 'roarm_debug_analysis.png'")
except Exception as e:
    print(f"⚠️  Visualisierung fehlgeschlagen: {e}")

# EMPFEHLUNGEN
print("\n\n🎯 EMPFEHLUNGEN:")
print("="*50)

if reach_rate < 80:
    print("❌ PROBLEM: Viele Targets sind NICHT ERREICHBAR!")
    print("   → Passe Target-Bereich an den Arbeitsbereich an")
    print("   → Oder nutze andere Forward Kinematics")

if np.mean(distances) > 0.2:
    print("⚠️  Große Start-zu-Ziel Distanzen")
    print("   → Mehr Steps oder größere action_scale nötig")

optimal_threshold = None
for thresh in [0.10, 0.08, 0.07, 0.06, 0.05]:
    reachable = []
    for target in targets:
        reachable.append(is_reachable(target, ee_standard, threshold=thresh))
    if np.mean(reachable) > 0.95:
        optimal_threshold = thresh
        break

if optimal_threshold:
    print(f"✅ Empfohlene Success Threshold: {optimal_threshold}")
else:
    print("❌ Selbst mit großer Threshold sind nicht alle Targets erreichbar!")

print("\n💡 Nächste Schritte:")
print("1. Wähle die passende Forward Kinematics")
print("2. Passe Target-Bereich an")
print("3. Setze Success Threshold auf", optimal_threshold or 0.10)
print("4. Trainiere nochmal mit angepassten Parametern")