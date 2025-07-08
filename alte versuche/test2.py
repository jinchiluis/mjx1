
import jax
import jax.numpy as jnp
import numpy as np
import pickle
import time

# ==========================================
# 1. POLICY FORWARD FUNCTION
# ==========================================
@jax.jit
def forward_policy(params, obs):
    """Policy forward pass mit Normalisierung"""
    # Normalisiere Observations (gleich wie im Training)
    obs_normalized = obs / jnp.array([jnp.pi, jnp.pi, jnp.pi, jnp.pi,  # qpos
                                     10.0, 10.0, 10.0, 10.0,  # qvel
                                     0.5, 0.5, 0.5])  # relative position
    
    h = jax.nn.relu(obs_normalized @ params["w1"] + params["b1"])
    h = jax.nn.relu(h @ params["w2"] + params["b2"])
    return jnp.tanh(h @ params["w3"] + params["b3"])


# ==========================================
# 2. FORWARD KINEMATICS
# ==========================================
def forward_kinematics(qpos):
    """Berechnet End-Effektor Position aus Gelenkwinkeln"""
    x = 0.1 * jnp.cos(qpos[0]) * (1 + 0.5 * jnp.cos(qpos[1]))
    y = 0.1 * jnp.sin(qpos[0]) * (1 + 0.5 * jnp.cos(qpos[1]))
    z = 0.1 + 0.1 * jnp.sin(qpos[1]) + 0.05 * jnp.sin(qpos[2])
    return jnp.array([x, y, z])


# ==========================================
# 3. PARAMETER LOADING AND FILTERING
# ==========================================
def load_and_filter_params(filename):
    """Lädt Parameter und filtert nur JAX-Arrays"""
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    
    print(f"📦 Geladene Daten: {type(data)}")
    
    # Falls es ein Dictionary ist, schaue nach 'params' key
    if isinstance(data, dict):
        print("🔍 Gefundene Keys:")
        for key, value in data.items():
            print(f"   {key}: {type(value)}")
        
        # Extrahiere die Parameter aus dem 'params' key
        if 'params' in data:
            params_dict = data['params']
            print(f"\n✅ Parameter gefunden unter 'params' key")
            print(f"   Parameter keys: {list(params_dict.keys())}")
            
            # Konvertiere alle Parameter zu JAX arrays
            params = {}
            for key, value in params_dict.items():
                if hasattr(value, 'shape') and hasattr(value, 'dtype'):
                    params[key] = jnp.array(value)
                    print(f"   ✅ {key}: {params[key].shape} {params[key].dtype}")
                else:
                    print(f"   ⚠️  Überspringe {key}: {type(value)}")
            
            # Zeige auch config info falls vorhanden
            if 'config' in data:
                config = data['config']
                print(f"\n📋 Config gefunden:")
                for key, value in config.items():
                    if key != 'forward_kinematics':  # Skip function
                        print(f"   {key}: {value}")
                
                # Gib config zurück für weitere Verwendung
                return params, None, config
            
            return params, None
        
        # Falls keine 'params' key, versuche direkte Parameter
        else:
            print("⚠️  Kein 'params' key gefunden, versuche direkte Parameter...")
            params = {}
            for key, value in data.items():
                if hasattr(value, 'shape') and hasattr(value, 'dtype'):
                    params[key] = jnp.array(value)
                    print(f"✅ Verwende {key}: {params[key].shape}")
                else:
                    print(f"⚠️  Überspringe {key}: {type(value)}")
            return params
    
    else:
        print(f"❌ Unerwarteter Datentyp: {type(data)}")
        return None


# ==========================================
# 4. TEST FUNCTION
# ==========================================
def test_policy(params, target, max_steps=200, visualize=True):
    """
    Testet die Policy mit einem gegebenen Ziel
    """
    # Initialer Zustand
    qpos = jnp.array([0.0, 0.0, 0.0, 0.0])  # Alle Gelenke bei 0
    qvel = jnp.array([0.0, 0.0, 0.0, 0.0])  # Keine Geschwindigkeit
    
    # Parameter
    action_scale = 0.1
    success_threshold = 0.05
    
    # Historie für Visualisierung
    history = {
        'qpos': [qpos],
        'ee_pos': [forward_kinematics(qpos)],
        'actions': [],
        'distances': []
    }
    
    print(f"\n🎯 Ziel: {target}")
    print(f"📍 Start End-Effektor: {history['ee_pos'][0]}")
    print(f"📏 Start Distanz: {jnp.linalg.norm(history['ee_pos'][0] - target):.4f}")
    print("\n" + "="*50)
    
    # Simulation Loop
    for step in range(max_steps):
        # Berechne aktuelle End-Effektor Position
        ee_pos = forward_kinematics(qpos)
        
        # Berechne Distanz zum Ziel
        distance = jnp.linalg.norm(ee_pos - target)
        history['distances'].append(float(distance))
        
        # Check ob Ziel erreicht
        if distance < success_threshold:
            print(f"\n✅ Ziel erreicht in {step} Schritten!")
            print(f"📏 Finale Distanz: {distance:.4f}")
            break
        
        # Erstelle Observation (OHNE Geschwindigkeiten!)
        obs = jnp.concatenate([qpos, target - ee_pos])  # 4 + 3 = 7 dimensions
        
        # Hole Action von Policy
        action = forward_policy(params, obs[None, :])[0]
        history['actions'].append(action)
        
        # Update Gelenkpositionen
        new_qpos = qpos + action * action_scale
        new_qpos = jnp.clip(new_qpos, -jnp.pi, jnp.pi)
        
        # Update Geschwindigkeiten (für Anzeige, aber nicht für Policy verwendet)
        new_qvel = (new_qpos - qpos) / 0.01
        
        # Speichere neuen Zustand
        qpos = new_qpos
        qvel = new_qvel  # Nur für Anzeige, Policy nutzt keine Geschwindigkeiten
        
        history['qpos'].append(qpos)
        history['ee_pos'].append(ee_pos)
        
        # Ausgabe alle 20 Schritte
        if step % 20 == 0:
            print(f"Step {step:3d} | Distanz: {distance:.4f} | "
                  f"Gelenke: [{qpos[0]:.2f}, {qpos[1]:.2f}, {qpos[2]:.2f}, {qpos[3]:.2f}] | "
                  f"Action: [{action[0]:.2f}, {action[1]:.2f}, {action[2]:.2f}, {action[3]:.2f}]")
    
    else:
        print(f"\n❌ Ziel nicht erreicht nach {max_steps} Schritten")
        print(f"📏 Finale Distanz: {distance:.4f}")
    
    # Zusammenfassung
    print("\n" + "="*50)
    print("📊 Zusammenfassung:")
    print(f"   Schritte: {len(history['actions'])}")
    print(f"   Min Distanz: {min(history['distances']):.4f}")
    print(f"   Finale Gelenke: {qpos}")
    print(f"   Finale EE Pos: {ee_pos}")
    
    # Visualisierung
    if visualize:
        visualize_trajectory(history, target)
    
    return history


def visualize_trajectory(history, target):
    """Einfache ASCII Visualisierung der Trajektorie"""
    print("\n📈 Distanz über Zeit:")
    
    distances = history['distances']
    max_dist = max(distances)
    
    # Plotte Distanz
    for i in range(0, len(distances), 5):
        dist = distances[i]
        bar_length = int(40 * (1 - dist/max_dist))
        bar = "█" * bar_length + "░" * (40 - bar_length)
        print(f"Step {i:3d}: {bar} {dist:.3f}")


# ==========================================
# 5. MAIN TEST
# ==========================================
def main():
    print("\n🤖 RoArm Policy Test Script")
    print("="*60)
    
    # Lade und filtere Parameter
    result = load_and_filter_params('roarm_working_params.pkl')
    
    if isinstance(result, tuple):
        params, config = result
    else:
        params, config = result, None
    
    if params is None:
        print("❌ Fehler beim Laden der Parameter!")
        return
    
    # Überprüfe ob alle nötigen Parameter vorhanden sind
    required_keys = ["w1", "b1", "w2", "b2", "w3", "b3"]
    missing_keys = [key for key in required_keys if key not in params]
    
    if missing_keys:
        print(f"❌ Fehlende Parameter: {missing_keys}")
        print("🔍 Verfügbare Parameter:")
        for key in params.keys():
            if hasattr(params[key], 'shape'):
                print(f"   {key}: {params[key].shape}")
        
        # Versuche alternative Parameter-Namen zu finden
        print("\n🔧 Versuche alternative Parameter-Namen zu mappen...")
        
        # Erstelle Dummy-Parameter falls nötig
        if len(params) == 0:
            print("⚠️  Keine JAX-Parameter gefunden, verwende Dummy-Parameter für Test")
            key = jax.random.PRNGKey(42)
            params = {
                "w1": jax.random.normal(key, (11, 128)) * 0.1,
                "b1": jnp.zeros(128),
                "w2": jax.random.normal(key, (128, 128)) * 0.1,
                "b2": jnp.zeros(128),
                "w3": jax.random.normal(key, (128, 4)) * 0.01,
                "b3": jnp.zeros(4),
            }
        
        return
    
    print("✅ Alle Parameter erfolgreich geladen!")
    
    # Test verschiedene Ziele
    test_targets = [
        jnp.array([0.05, 0.05, 0.15]),   # Einfaches Ziel (nah)
        jnp.array([0.1, 0.0, 0.2]),       # Mittleres Ziel
        jnp.array([-0.05, 0.08, 0.15]),   # Seitliches Ziel
    ]
    
    for i, target in enumerate(test_targets):
        print(f"\n\n{'='*60}")
        print(f"🧪 TEST {i+1}: Target = {target}")
        print(f"{'='*60}")
        
        history = test_policy(params, target, max_steps=200)
        
        # Warte kurz zwischen Tests
        if i < len(test_targets) - 1:
            print("\n⏳ Nächster Test in 2 Sekunden...")
            time.sleep(2)
    
    print("\n\n✅ Alle Tests abgeschlossen!")


if __name__ == "__main__":
    main()