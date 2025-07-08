# roarm_rl_debug.py - Mit ausführlichen Debug-Ausgaben
import os
import sys
import time
import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple
from jax.tree_util import tree_map, tree_leaves

print("🚀 Starte RoArm RL Training...")
print(f"Python: {sys.version}")
print(f"JAX: {jax.__version__}")

# MuJoCo-Import mit Fehlerbehandlung
try:
    import mujoco
    import mujoco.mjx as mjx
    print(f"✓ MuJoCo geladen: {mujoco.__version__}")
except ImportError as e:
    print(f"❌ MuJoCo Import-Fehler: {e}")
    print("Installiere mit: pip install mujoco")
    sys.exit(1)

# ==========================================
# 1. RL-UMGEBUNG (vereinfacht ohne MuJoCo für Tests)
# ==========================================
class RoArmRLEnvSimple:
    """Vereinfachte RL-Umgebung ohne MuJoCo-Abhängigkeit"""
    DOF = 4
    TARGET_DIM = 3
    
    def __init__(self, task: str = "reach"):
        print(f"📦 Initialisiere Umgebung für Task: {task}")
        self.task = task
        self.action_scale = 0.1
        self.max_steps = 200
        self.obs_dim = 2 * self.DOF + self.TARGET_DIM  # 11
        self.action_dim = self.DOF
        print(f"✓ Umgebung bereit: obs_dim={self.obs_dim}, action_dim={self.action_dim}")

    def reset(self, key):
        qpos_key, qvel_key, target_key = jax.random.split(key, 3)
        qpos = jax.random.uniform(qpos_key, (self.DOF,), minval=-0.5, maxval=0.5)
        qvel = jnp.zeros(self.DOF)
        target = jax.random.uniform(
            target_key,
            (self.TARGET_DIM,),
            minval=jnp.array([-0.1, -0.1, 0.1]),
            maxval=jnp.array([0.1, 0.1, 0.3])
        )
        state = {"qpos": qpos, "qvel": qvel, "ee_pos": self._forward_kinematics(qpos)}
        return state, target, 0

    def _forward_kinematics(self, qpos):
        """Einfache FK für 4-DOF-Arm"""
        # Vereinfachte Berechnung
        x = 0.1 * jnp.cos(qpos[0]) * (1 + 0.5 * jnp.cos(qpos[1]))
        y = 0.1 * jnp.sin(qpos[0]) * (1 + 0.5 * jnp.cos(qpos[1]))
        z = 0.1 + 0.1 * jnp.sin(qpos[1]) + 0.05 * jnp.sin(qpos[2])
        return jnp.array([x, y, z])

    def step(self, state, action, target, step_count):
        # Update joint positions
        new_qpos = state["qpos"] + action * self.action_scale
        new_qpos = jnp.clip(new_qpos, -jnp.pi, jnp.pi)
        
        # Simple velocity update
        new_qvel = (new_qpos - state["qpos"]) / 0.01
        
        # Update end-effector
        new_ee_pos = self._forward_kinematics(new_qpos)
        
        new_state = {
            "qpos": new_qpos,
            "qvel": new_qvel,
            "ee_pos": new_ee_pos
        }
        
        # Check if target reached
        dist = jnp.linalg.norm(new_state["ee_pos"] - target)
        target_reached = dist < 0.02
        
        reward = self._compute_reward(new_state, target, action)
        done = jnp.logical_or(step_count >= self.max_steps, target_reached)
        
        return new_state, reward, done, step_count + 1

    def _compute_reward(self, state, target, action):
        dist = jnp.linalg.norm(state["ee_pos"] - target)
        dist_reward = -dist
        reached_reward = jax.lax.cond(dist < 0.02, lambda: 50.0, lambda: 0.0)  # Reduziert von 10.0
        action_penalty = -0.01 * jnp.sum(action ** 2)
        return dist_reward + reached_reward + action_penalty


# ==========================================
# 2. POLICY-NETZ
# ==========================================
class PolicyNetwork:
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        print(f"🧠 Policy-Netz: {obs_dim} → {hidden_dim} → {hidden_dim} → {action_dim}")

    def init_params(self, key):
        k1, k2, k3 = jax.random.split(key, 3)
        return {
            "w1": jax.random.normal(k1, (self.obs_dim, self.hidden_dim)) * 0.1,
            "b1": jnp.zeros(self.hidden_dim),
            "w2": jax.random.normal(k2, (self.hidden_dim, self.hidden_dim)) * 0.1,
            "b2": jnp.zeros(self.hidden_dim),
            "w3": jax.random.normal(k3, (self.hidden_dim, self.action_dim)) * 0.01,
            "b3": jnp.zeros(self.action_dim),
        }

    def forward(self, params, obs):
        h = jax.nn.relu(obs @ params["w1"] + params["b1"])
        h = jax.nn.relu(h @ params["w2"] + params["b2"])
        return jnp.tanh(h @ params["w3"] + params["b3"])


# ==========================================
# 3. HILFSFUNKTIONEN
# ==========================================
def make_observation(state, target):
    qpos = state["qpos"]
    qvel = state["qvel"]
    rel_pos = target - state["ee_pos"]
    return jnp.concatenate([qpos, qvel, rel_pos])


def rollout_episode(policy_params, policy_net, env, key):
    """Episode-Durchlauf (ohne JIT wegen Objekten)"""
    reset_key, noise_key = jax.random.split(key)
    state, target, step = env.reset(reset_key)
    total_reward = 0.0

    for _ in range(env.max_steps):
        obs = make_observation(state, target)
        action = policy_net.forward(policy_params, obs)
        
        noise_key, step_key = jax.random.split(noise_key)
        action += jax.random.normal(step_key, action.shape) * 0.1
        
        state, reward, done, step = env.step(state, action, target, step)
        total_reward += reward
        
        if done:
            break
    
    return total_reward


# ==========================================
# 4. EVOLUTION STRATEGIES TRAINING
# ==========================================
def evaluate_population(params_pop, policy_net, env, key):
    n_pop = tree_leaves(params_pop)[0].shape[0]
    n_episodes = 5  # <-- Mehr Episoden!
    
    rewards = []
    for i in range(n_pop):
        individual = tree_map(lambda x: x[i], params_pop)
        episode_rewards = []
        
        # Mehrere Episoden pro Individuum
        for ep in range(n_episodes):
            ep_key = jax.random.PRNGKey(key[0] * 1000 + i * 100 + ep)
            reward = rollout_episode(individual, policy_net, env, ep_key)
            episode_rewards.append(reward)
        
        # Durchschnitt über alle Episoden
        rewards.append(jnp.mean(jnp.array(episode_rewards)))
    
    return jnp.array(rewards)


def train_with_evolution(
    env,
    policy_net,
    n_generations: int = 100,
    population_size: int = 50,
    top_k: int = 10,
):
    print(f"\n🏃 Starte Evolution Strategies Training")
    print(f"   Generationen: {n_generations}")
    print(f"   Population: {population_size}")
    print(f"   Elite: {top_k}")
    
    start_time = time.time()
    
    # Initialisiere Population
    print("\n📊 Initialisiere Population...")
    key = jax.random.PRNGKey(42)
    pop_keys = jax.random.split(key, population_size)
    population = jax.vmap(policy_net.init_params)(pop_keys)
    print("✓ Population initialisiert")

    best_reward = -jnp.inf
    best_params = None
    
    # Kompiliere Evaluierungsfunktion
    print("\n🔧 Kompiliere JAX-Funktionen...")
    eval_key = jax.random.PRNGKey(0)
    _ = evaluate_population(population, policy_net, env, eval_key)
    print("✓ Funktionen kompiliert")

    print("\n🎯 Starte Training...\n")
    
    for gen in range(n_generations):
        gen_start = time.time()
        
        # Evaluiere Population
        eval_key = jax.random.PRNGKey(gen)
        rewards = evaluate_population(population, policy_net, env, eval_key)
        
        # Update bester
        best_idx = int(jnp.argmax(rewards))
        if rewards[best_idx] > best_reward:
            best_reward = float(rewards[best_idx])
            best_params = tree_map(lambda x: x[best_idx], population)
        
        # Erstelle neue Generation
        top_idx = jnp.argsort(rewards)[-top_k:]
        new_pop = []
        mut_key = jax.random.PRNGKey(gen + 1234)
        
        for i in range(population_size):
            if i < top_k:
                new_pop.append(tree_map(lambda x: x[top_idx[i]], population))
            else:
                parent_idx = int(top_idx[i % top_k])
                parent = tree_map(lambda x: x[parent_idx], population)
                mut_key, noise_key = jax.random.split(mut_key)
                noise = tree_map(
                    lambda p: jax.random.normal(noise_key, p.shape) * 0.1, parent
                )
                child = tree_map(jnp.add, parent, noise)
                new_pop.append(child)
        
        population = tree_map(lambda *xs: jnp.stack(xs), *new_pop)
        
        gen_time = time.time() - gen_start
        
        # Ausgabe
        if gen % 5 == 0:
            avg_reward = float(jnp.mean(rewards))
            print(f"Gen {gen:3d} | "
                  f"Best: {best_reward:6.2f} | "
                  f"Avg: {avg_reward:6.2f} | "
                  f"Zeit: {gen_time:.2f}s")
    
    total_time = time.time() - start_time
    print(f"\n✓ Training abgeschlossen in {total_time:.1f}s")
    print(f"🏆 Beste Belohnung: {best_reward:.2f}")
    
    return best_params


# ==========================================
# 5. MAIN
# ==========================================
def main():
    print("\n" + "="*50)
    print("🤖 RoArm-M2-S REINFORCEMENT LEARNING 🤖")
    print("="*50)
    
    # Verwende vereinfachte Umgebung
    print("\n📋 Verwende vereinfachte Umgebung (ohne MuJoCo)")
    env = RoArmRLEnvSimple(task="reach")
    
    # Erstelle Policy-Netz
    policy_net = PolicyNetwork(env.obs_dim, env.action_dim)
    
    # Trainiere
    best_params = train_with_evolution(
        env, 
        policy_net, 
        n_generations=30,
        population_size=20,
        top_k=5
    )
    
    print("\n✅ Fertig!")
    
    # Test des trainierten Modells
    print("\n🧪 Teste trainiertes Modell...")
    test_key = jax.random.PRNGKey(999)
    test_reward = rollout_episode(best_params, policy_net, env, test_key)
    print(f"Test-Belohnung: {float(test_reward):.2f}")


if __name__ == "__main__":
    main()