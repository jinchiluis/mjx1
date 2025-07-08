import mujoco              # native Python-Bindings
import mujoco.mjx as mjx   # JAX-Wrapper
from mujoco import MjModel, MjData

# 1. Modell und Daten anlegen
model    = MjModel.from_xml_path("roarm_m2s.xml")
mjx_model = mjx.put_model(model)
data     = MjData(model)

# 2. Kurze Simulations-Schleife
for step in range(100):
    # z.B. alle Aktuatoren auf 0 setzen
    data.ctrl[:] = 0.1
    # klassischer Schritt (für Visualisierung/Debug)
    mujoco.mj_step(model, data)
    # JAX-Schritt (für vmap/jit/batched Simulationen)
    # data = mjx.step(mjx_model, data, jnp.zeros(model.nu))

    # Gelenkwinkel auslesen
    qpos = data.qpos.copy()
    print(f"Step {step}: qpos = {qpos}")

print("Simulation fertig!")