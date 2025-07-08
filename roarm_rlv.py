# roarm_rl_robust.py - Robuste Version mit balancierten Rewards
import os
import sys
import time
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import Tuple, Dict

# WICHTIG: Wähle Platform
# os.environ['JAX_PLATFORM_NAME'] = 'cpu'  # Für CPU
os.environ['JAX_PLATFORM_NAME'] = 'gpu'  # Für GPU

# Fix für Deprecation Warnings
if hasattr(jax, 'tree'):
    tree_map = jax.tree.map
else:
    tree_map = jax.tree_util.tree_map

print("🚀 Starte RoArm RL Training (Robuste Version)...")
print(f"Python: {sys.version}")
print(f"JAX: {jax.__version__}")
print(f"Devices: {jax.devices()}")

# ==========================================
# 1. VEKTORISIERTE RL-UMGEBUNG
# ==========================================
class RoArmRLEnvVectorized:
    """Vollständig vektorisierte Umgebung für JAX"""
    DOF = 4
    TARGET_DIM = 3
    
    def __init__(self, task: str = "reach"):
        self.task = task
        self.action_scale = 0.1
        self.max_steps = 200
        self.obs_dim = 2 * self.DOF + self.TARGET_DIM
        self.action_dim = self.DOF
        self.success_threshold = 0.05
        print(f"✓ Vektorisierte Umgebung: obs_dim={self.obs_dim}, action_dim={self.action_dim}")

    @partial(jax.jit, static_argnums=(0,))
    def reset_single(self, key):
        qpos_key, target_key = jax.random.split(key)
        qpos = jax.random.uniform(qpos_key, (self.DOF,), minval=-0.5, maxval=0.5)
        qvel = jnp.zeros(self.DOF)
        target = jax.random.uniform(
            target_key,
            (self.TARGET_DIM,),
            minval=jnp.array([-0.1, -0.1, 0.1]),
            maxval=jnp.array([0.1, 0.1, 0.3])
        )
        ee_pos = self._forward_kinematics(qpos)
        return qpos, qvel, ee_pos, target

    @partial(jax.jit, static_argnums=(0,))
    def _forward_kinematics(self, qpos):
        x = 0.1 * jnp.cos(qpos[0]) * (1 + 0.5 * jnp.cos(qpos[1]))
        y = 0.1 * jnp.sin(qpos[0]) * (1 + 0.5 * jnp.cos(qpos[1]))
        z = 0.1 + 0.1 * jnp.sin(qpos[1]) + 0.05 * jnp.sin(qpos[2])
        return jnp.array([x, y, z])

    @partial(jax.jit, static_argnums=(0,))
    def step_single(self, qpos, qvel, ee_pos, target, action):
        # Update positions
        new_qpos = qpos + action * self.action_scale
        new_qpos = jnp.clip(new_qpos, -jnp.pi, jnp.pi)
        new_qvel = (new_qpos - qpos) / 0.01
        new_ee_pos = self._forward_kinematics(new_qpos)
        
        # BALANCIERTE REWARD-FUNKTION
        dist = jnp.linalg.norm(new_ee_pos - target)
        prev_dist = jnp.linalg.norm(ee_pos - target)
        
        # Basis-Rewards (moderater)
        progress_reward = (prev_dist - dist) * 20.0  # Reduziert von 50
        success_bonus = jnp.where(dist < self.success_threshold, 50.0, 0.0)  # Reduziert von 100
        distance_penalty = -dist * 1.0  # Noch weniger aggressiv
        action_penalty = -0.05 * jnp.sum(action ** 2)  # Etwas höher für Smoothness
        
        # Zeit-Penalty um schnelle Lösungen zu fördern (aber nicht zu stark)
        time_penalty = -0.1
        
        reward = progress_reward + success_bonus + distance_penalty + action_penalty + time_penalty
        
        done = dist < self.success_threshold
        
        return new_qpos, new_qvel, new_ee_pos, reward, done


# ==========================================
# 2. VEKTORISIERTE POLICY
# ==========================================
class PolicyNetworkVectorized:
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.weight_decay = 0.001  # L2 Regularisierung
        print(f"🧠 Policy-Netz: {obs_dim} → {hidden_dim} → {hidden_dim} → {action_dim}")

    def init_params(self, key):
        k1, k2, k3 = jax.random.split(key, 3)
        # Bessere Initialisierung mit Xavier/He
        scale1 = jnp.sqrt(2.0 / self.obs_dim)
        scale2 = jnp.sqrt(2.0 / self.hidden_dim)
        scale3 = 0.01  # Kleinere finale Layer für sanftere Actions
        
        return {
            "w1": jax.random.normal(k1, (self.obs_dim, self.hidden_dim)) * scale1,
            "b1": jnp.zeros(self.hidden_dim),
            "w2": jax.random.normal(k2, (self.hidden_dim, self.hidden_dim)) * scale2,
            "b2": jnp.zeros(self.hidden_dim),
            "w3": jax.random.normal(k3, (self.hidden_dim, self.action_dim)) * scale3,
            "b3": jnp.zeros(self.action_dim),
        }

    def get_weight_penalty(self, params):
        """L2 weight decay für Regularisierung"""
        penalty = 0.0
        for key in ["w1", "w2", "w3"]:
            penalty += jnp.sum(params[key] ** 2)
        return penalty * self.weight_decay


# ==========================================
# 3. VEKTORISIERTE ROLLOUTS
# ==========================================
@jax.jit
def forward_policy(params, obs):
    """Policy forward pass mit Normalisierung"""
    # Normalisiere Observations
    obs_normalized = obs / jnp.array([jnp.pi, jnp.pi, jnp.pi, jnp.pi,  # qpos
                                     10.0, 10.0, 10.0, 10.0,  # qvel
                                     0.5, 0.5, 0.5])  # relative position
    
    h = jax.nn.relu(obs_normalized @ params["w1"] + params["b1"])
    h = jax.nn.relu(h @ params["w2"] + params["b2"])
    return jnp.tanh(h @ params["w3"] + params["b3"])


@partial(jax.jit, static_argnums=(2, 3))
def rollout_vectorized(params_batch, keys, env, use_noise=False):
    """
    Führt parallele Rollouts für eine ganze Population durch.
    Mehr Episoden für robustere Evaluation!
    """
    n_pop = keys.shape[0]
    n_episodes = 15  # Erhöht von 5 auf 15 für stabilere Evaluation
    
    def rollout_single(params, key):
        """Ein kompletter Rollout für ein Individuum"""
        episode_keys = jax.random.split(key, n_episodes)
        
        def single_episode(ep_key):
            reset_key, noise_key = jax.random.split(ep_key)
            qpos, qvel, ee_pos, target = env.reset_single(reset_key)
            
            def step_fn(carry, _):
                qpos, qvel, ee_pos, total_reward, noise_key, done, steps = carry
                
                # Observation
                obs = jnp.concatenate([qpos, qvel, target - ee_pos])
                
                # Action (OHNE Noise während Evaluation!)
                action = forward_policy(params, obs[None, :])[0]
                
                # Nur Noise hinzufügen wenn explizit gewünscht
                if use_noise:
                    noise_key, act_key = jax.random.split(noise_key)
                    # Adaptiver Noise: weniger wenn näher am Ziel
                    dist_to_target = jnp.linalg.norm(target - ee_pos)
                    noise_scale = 0.05 * jnp.minimum(dist_to_target / 0.1, 1.0)
                    noise = jax.random.normal(act_key, action.shape) * noise_scale
                    action = action + noise
                
                action = jnp.where(done, jnp.zeros_like(action), action)
                
                # Step (nur wenn nicht done)
                new_qpos, new_qvel, new_ee_pos, reward, step_done = env.step_single(
                    qpos, qvel, ee_pos, target, action
                )
                
                # Conditional update basierend auf done
                qpos = jnp.where(done, qpos, new_qpos)
                qvel = jnp.where(done, qvel, new_qvel)
                ee_pos = jnp.where(done[..., None], ee_pos, new_ee_pos)
                
                # WICHTIG: Keine weiteren negativen Rewards nach Erfolg!
                reward = jnp.where(done, 0.0, reward)
                done = jnp.logical_or(done, step_done)
                
                # Proportionaler Early-Termination Bonus (sanfter)
                time_bonus = jnp.where(
                    step_done,
                    (env.max_steps - steps) * 0.1,  # 0.1 pro gesparter Step
                    0.0
                )
                
                # Update carry
                new_carry = (qpos, qvel, ee_pos, 
                           total_reward + reward + time_bonus, 
                           noise_key, done, steps + 1)
                
                return new_carry, None
            
            # Unroll episode
            init_carry = (qpos, qvel, ee_pos, 0.0, noise_key, False, 0)
            final_carry, _ = jax.lax.scan(step_fn, init_carry, None, length=env.max_steps)
            
            total_reward = final_carry[3]
            final_done = final_carry[5]
            
            # Return reward und success flag
            return total_reward, final_done
        
        # Führe alle Episoden aus
        episode_results = jax.vmap(single_episode)(episode_keys)
        episode_rewards = episode_results[0]
        episode_successes = episode_results[1]
        
        # Robuste Metrik: Mittelwert + Erfolgsrate-Bonus
        mean_reward = jnp.mean(episode_rewards)
        success_rate = jnp.mean(episode_successes.astype(jnp.float32))
        robust_fitness = mean_reward + success_rate * 20.0  # Bonus für Konsistenz
        
        return robust_fitness, success_rate
    
    # Vektorisiere über die ganze Population
    results = jax.vmap(rollout_single)(params_batch, keys)
    fitnesses = results[0]
    success_rates = results[1]
    
    return fitnesses, success_rates


# ==========================================
# 4. TOURNAMENT SELECTION
# ==========================================
@jax.jit
def tournament_selection(population, fitnesses, key, tournament_size=7):
    """Tournament Selection für bessere Diversität"""
    pop_size = fitnesses.shape[0]
    
    def select_one(key):
        # Wähle tournament_size zufällige Individuen
        indices = jax.random.choice(key, pop_size, shape=(tournament_size,), replace=False)
        tournament_fitnesses = fitnesses[indices]
        
        # Wähle den Besten aus dem Tournament
        winner_idx = indices[jnp.argmax(tournament_fitnesses)]
        return winner_idx
    
    # Wähle pop_size Gewinner
    selection_keys = jax.random.split(key, pop_size)
    selected_indices = jax.vmap(select_one)(selection_keys)
    
    return selected_indices


@jax.jit
def create_new_generation_robust(population, fitnesses, success_rates, key):
    """Robuste Generationserzeugung mit Tournament Selection"""
    population_size = fitnesses.shape[0]
    
    # Elite: Top 10% werden direkt übernommen
    n_elite = population_size // 10
    elite_idx = jnp.argsort(fitnesses)[-n_elite:]
    
    # Tournament Selection für Rest
    selection_key, crossover_key, mutation_key = jax.random.split(key, 3)
    
    # Wähle Eltern via Tournament
    parent_indices = tournament_selection(population, fitnesses, selection_key)
    
    # Adaptive Mutation basierend auf Erfolgsrate
    avg_success = jnp.mean(success_rates)
    base_mutation = 0.1
    # Mehr Mutation wenn Erfolgsrate niedrig
    adaptive_mutation = base_mutation * (2.0 - avg_success)
    
    def create_individual(i, keys):
        parent_key, mutation_key = keys
        
        # Elite direkt übernehmen
        is_elite = i < n_elite
        
        # Für Crossover: Zwei verschiedene Eltern
        p1_idx = parent_indices[i]
        p2_idx = parent_indices[(i + population_size // 2) % population_size]
        
        # Hole beide Eltern
        parent1 = tree_map(lambda x: x[p1_idx], population)
        parent2 = tree_map(lambda x: x[p2_idx], population)
        
        # Elite bekommt direkt den Elite-Parent
        elite_parent = tree_map(lambda x: x[elite_idx[i % n_elite]], population)
        
        # Uniform Crossover für Nicht-Elite
        mask_key, noise_key = jax.random.split(mutation_key)
        
        def crossover(p1, p2, elite_p, key):
            # Für Elite: nimm elite_p, sonst crossover
            mask = jax.random.uniform(key, p1.shape) > 0.5
            crossover_child = jnp.where(mask, p1, p2)
            return jnp.where(is_elite, elite_p, crossover_child)
        
        # Erstelle Kind (Elite oder Crossover)
        child = tree_map(
            lambda p1, p2, ep: crossover(p1, p2, ep, mask_key),
            parent1, parent2, elite_parent
        )
        
        # Mutation (keine für Elite, adaptiv für Rest)
        noise_scale = jnp.where(is_elite, 0.0, adaptive_mutation)
        
        # Mutation mit Momentum (50% Chance für gerichtete Mutation)
        use_momentum = jax.random.uniform(parent_key) > 0.5
        
        def mutate(param, key):
            noise = jax.random.normal(key, param.shape)
            # Bei Momentum: verstärke Richtung der letzten Änderung
            momentum_noise = jnp.where(use_momentum, 
                                     jnp.sign(param) * jnp.abs(noise),
                                     noise)
            return param + momentum_noise * noise_scale
        
        # Generiere verschiedene Keys für jeden Parameter
        noise_keys = jax.random.split(noise_key, len(child))
        mutated = tree_map(mutate, child, 
                          dict(zip(child.keys(), noise_keys)))
        
        return mutated
    
    # Erstelle neue Generation
    ind_keys = jax.random.split(mutation_key, population_size)
    parent_keys = jax.random.split(crossover_key, population_size)
    keys = jnp.stack([parent_keys, ind_keys], axis=1)
    
    new_population = jax.vmap(create_individual)(jnp.arange(population_size), keys)
    
    return new_population


def train_robust(
    env,
    policy_net,
    n_generations: int = 300,
    population_size: int = 500,
):
    print(f"\n🏃 Starte robustes Training")
    print(f"   Generationen: {n_generations}")
    print(f"   Population: {population_size}")
    print(f"   Episoden pro Eval: 15")
    
    start_time = time.time()
    
    # Initialisiere Population
    key = jax.random.PRNGKey(42)
    pop_keys = jax.random.split(key, population_size)
    population = jax.vmap(policy_net.init_params)(pop_keys)
    
    # Pre-compile
    print("\n🔧 Kompiliere JAX-Funktionen...")
    compile_start = time.time()
    
    dummy_keys = jax.random.split(jax.random.PRNGKey(0), population_size)
    _ = rollout_vectorized(population, dummy_keys, env, use_noise=False)
    
    compile_time = time.time() - compile_start
    print(f"✓ Kompilierung abgeschlossen in {compile_time:.1f}s")
    
    best_fitness = -jnp.inf
    best_params = None
    history = []
    
    print("\n🎯 Starte Training...\n")
    
    for gen in range(n_generations):
        gen_start = time.time()
        
        # Evaluiere Population
        eval_keys = jax.random.split(jax.random.PRNGKey(gen * 1000), population_size)
        fitnesses, success_rates = rollout_vectorized(
            population, eval_keys, env, use_noise=False
        )
        
        # Update bester
        best_idx = int(jnp.argmax(fitnesses))
        if fitnesses[best_idx] > best_fitness:
            best_fitness = float(fitnesses[best_idx])
            best_params = tree_map(lambda x: x[best_idx], population)
        
        # Neue Generation
        gen_key = jax.random.PRNGKey(gen + 5000)
        population = create_new_generation_robust(
            population, fitnesses, success_rates, gen_key
        )
        
        gen_time = time.time() - gen_start
        
        # Statistiken
        avg_fitness = float(jnp.mean(fitnesses))
        std_fitness = float(jnp.std(fitnesses))
        avg_success = float(jnp.mean(success_rates)) * 100
        top10_success = float(jnp.mean(success_rates[jnp.argsort(fitnesses)[-50:]])) * 100
        
        history.append({
            'gen': gen,
            'best': best_fitness,
            'avg': avg_fitness,
            'std': std_fitness,
            'success_rate': avg_success,
            'top10_success': top10_success
        })
        
        # Ausgabe
        if gen % 5 == 0:
            print(f"Gen {gen:3d} | "
                  f"Best: {best_fitness:6.1f} | "
                  f"Avg: {avg_fitness:6.1f} ± {std_fitness:5.1f} | "
                  f"Success: {avg_success:4.1f}% (Top10: {top10_success:4.1f}%) | "
                  f"Zeit: {gen_time:.3f}s")
            
            # Early Stopping nur bei hoher Erfolgsrate
            if gen > 30 and avg_success > 80 and top10_success > 95:
                print("\n🎉 Konvergiert! Hohe Erfolgsrate erreicht.")
                break
    
    total_time = time.time() - start_time
    print(f"\n✓ Training abgeschlossen in {total_time:.1f}s")
    print(f"🏆 Beste Fitness: {best_fitness:.2f}")
    
    return best_params, history


# ==========================================
# 5. ANALYSE UND TEST
# ==========================================
def analyze_population(env, population, key):
    """Analysiere die Verteilung der Population"""
    eval_keys = jax.random.split(key, len(population))
    fitnesses, success_rates = rollout_vectorized(
        population, eval_keys, env, use_noise=False
    )
    
    # Finde "Superhelden"
    sorted_idx = jnp.argsort(fitnesses)
    top_performers = sorted_idx[-10:]
    
    print("\n📊 Populations-Analyse:")
    print(f"   Top 10 Fitness: {fitnesses[top_performers]}")
    print(f"   Top 10 Success: {success_rates[top_performers] * 100}%")
    print(f"   Fitness Quantile: P25={jnp.percentile(fitnesses, 25):.1f}, "
          f"P50={jnp.percentile(fitnesses, 50):.1f}, "
          f"P75={jnp.percentile(fitnesses, 75):.1f}")
    
    return fitnesses, success_rates


# ==========================================
# 6. MAIN
# ==========================================
def main():
    print("\n" + "="*60)
    print("🤖 RoArm-M2-S REINFORCEMENT LEARNING (ROBUST) 🤖")
    print("="*60)
    
    # Umgebung
    env = RoArmRLEnvVectorized(task="reach")
    
    # Policy
    policy_net = PolicyNetworkVectorized(
        env.obs_dim, 
        env.action_dim,
        hidden_dim=128
    )
    
    # Training
    best_params, history = train_robust(
        env, 
        policy_net, 
        n_generations=1200,
        population_size=500
    )
    
    print("\n✅ Training abgeschlossen!")
    
    # Ausführlicher Test
    print("\n🧪 Teste trainiertes Modell (50 Runs)...")
    test_fitnesses = []
    test_successes = []
    
    for i in range(50):
        test_key = jax.random.PRNGKey(10000 + i)
        fitness, success = rollout_vectorized(
            tree_map(lambda x: x[None, ...], best_params),
            jnp.array([test_key]),
            env,
            use_noise=False
        )
        test_fitnesses.append(float(fitness[0]))
        test_successes.append(float(success[0]))
    
    print(f"Test-Fitness: {np.mean(test_fitnesses):.2f} ± {np.std(test_fitnesses):.2f}")
    print(f"Erfolgsrate: {np.mean(test_successes) * 100:.1f}% "
          f"({sum(test_successes)}/{len(test_successes)} erfolgreich)")
    
    # Zeige Trainingsfortschritt
    if len(history) > 10:
        print("\n📈 Trainingsfortschritt:")
        milestones = [0, len(history)//4, len(history)//2, 3*len(history)//4, -1]
        for i in milestones:
            h = history[i]
            print(f"   Gen {h['gen']:3d}: "
                  f"Best={h['best']:6.1f}, "
                  f"Avg={h['avg']:6.1f}, "
                  f"Success={h['success_rate']:4.1f}%")
    
    # Performance-Breakdown
    success_array = np.array(test_successes)
    if np.sum(success_array) > 0:
        print("\n🎯 Performance-Details:")
        print(f"   Konsistente Erfolge (>80% in 10er Gruppen): "
              f"{sum(success_array[i:i+10].mean() > 0.8 for i in range(0, 50, 10))}/5")
        
    # Nach "✅ Training abgeschlossen!"
    import pickle

    # Speichere die besten Parameter
    with open('roarm_trained_params.pkl', 'wb') as f:
        pickle.dump(best_params, f)
    print("💾 Parameter gespeichert in 'roarm_trained_params.pkl'")

    # Optional: Speichere auch die History
    with open('roarm_training_history.pkl', 'wb') as f:
        pickle.dump(history, f)        


if __name__ == "__main__":
    main()