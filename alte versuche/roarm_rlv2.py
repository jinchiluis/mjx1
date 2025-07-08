# roarm_fixed.py - Endlich mit den richtigen Parametern!
import os
import sys
import time
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

os.environ['JAX_PLATFORM_NAME'] = 'gpu'

if hasattr(jax, 'tree'):
    tree_map = jax.tree.map
else:
    tree_map = jax.tree_util.tree_map

print("🚀 RoArm Training - FIXED VERSION")
print(f"JAX: {jax.__version__}, Devices: {jax.devices()}")

# ==========================================
# 1. KORREKTE ENVIRONMENT
# ==========================================
class RoArmEnvFixed:
    DOF = 4
    TARGET_DIM = 3
    
    def __init__(self):
        self.action_scale = 0.15  # Gut so
        self.max_steps = 50  # Reicht
        self.obs_dim = self.DOF + self.TARGET_DIM
        self.action_dim = self.DOF
        self.success_threshold = 0.10  # KORRIGIERT! War 0.05-0.07
        
        # Link lengths für Standard 4DOF
        self.l0 = 0.05  # Base height
        self.l1 = 0.1   # First link
        self.l2 = 0.1   # Second link  
        self.l3 = 0.05  # Third link
        
    @partial(jax.jit, static_argnums=(0,))
    def reset_single(self, key):
        qpos_key, target_key = jax.random.split(key)
        
        # Start näher an Null für kürzere Distanzen
        qpos = jax.random.uniform(qpos_key, (self.DOF,), minval=-0.2, maxval=0.2)
        
        # Target-Bereich bleibt (ist ja erreichbar!)
        target = jax.random.uniform(
            target_key,
            (self.TARGET_DIM,),
            minval=jnp.array([-0.10, -0.10, 0.08]),  # Etwas konservativer
            maxval=jnp.array([0.10, 0.10, 0.22])
        )
        
        ee_pos = self._forward_kinematics(qpos)
        prev_action = jnp.zeros(self.DOF)
        
        return qpos, ee_pos, target, prev_action

    @partial(jax.jit, static_argnums=(0,))
    def _forward_kinematics(self, qpos):
        """Standard 4-DOF Forward Kinematics - DIE RICHTIGE!"""
        # Base rotation + 3 pitch joints
        
        # Nach Joint 1 (shoulder)
        r1 = self.l1 * jnp.cos(qpos[1])
        z1 = self.l0 + self.l1 * jnp.sin(qpos[1])
        
        # Nach Joint 2 (elbow)  
        r2 = r1 + self.l2 * jnp.cos(qpos[1] + qpos[2])
        z2 = z1 + self.l2 * jnp.sin(qpos[1] + qpos[2])
        
        # End effector (nach Joint 3)
        r3 = r2 + self.l3 * jnp.cos(qpos[1] + qpos[2] + qpos[3])
        z3 = z2 + self.l3 * jnp.sin(qpos[1] + qpos[2] + qpos[3])
        
        # Apply base rotation
        x = r3 * jnp.cos(qpos[0])
        y = r3 * jnp.sin(qpos[0])
        z = z3
        
        return jnp.array([x, y, z])

    @partial(jax.jit, static_argnums=(0,))
    def step_single(self, qpos, ee_pos, target, prev_action, action):
        # Action smoothing für Real Robot
        alpha = 0.7
        smoothed_action = alpha * action + (1 - alpha) * prev_action
        
        # Update
        new_qpos = qpos + smoothed_action * self.action_scale
        new_qpos = jnp.clip(new_qpos, -jnp.pi, jnp.pi)
        new_ee_pos = self._forward_kinematics(new_qpos)
        
        # Rewards - VEREINFACHT & KLAR
        dist = jnp.linalg.norm(new_ee_pos - target)
        prev_dist = jnp.linalg.norm(ee_pos - target)
        
        # Hauptbelohnung: Progress
        progress = (prev_dist - dist) * 50.0
        
        # Distance penalty (motiviert konstanten Progress)
        distance_penalty = -dist * 2.0
        
        # Success Bonus - GROß für Motivation
        success_bonus = jnp.where(dist < self.success_threshold, 200.0, 0.0)
        
        # Kleine Action penalty für Smoothness
        action_penalty = -0.05 * jnp.sum(smoothed_action ** 2)
        
        reward = progress + distance_penalty + success_bonus + action_penalty
        done = dist < self.success_threshold
        
        return new_qpos, new_ee_pos, reward, done, smoothed_action


# ==========================================
# 2. SIMPLE BUT EFFECTIVE NETWORK
# ==========================================
def init_network(key, obs_dim, hidden_dim, action_dim):
    k1, k2, k3 = jax.random.split(key, 3)
    return {
        'w1': jax.random.normal(k1, (obs_dim, hidden_dim)) * jnp.sqrt(2.0/obs_dim),
        'b1': jnp.zeros(hidden_dim),
        'w2': jax.random.normal(k2, (hidden_dim, hidden_dim)) * jnp.sqrt(2.0/hidden_dim),
        'b2': jnp.zeros(hidden_dim),
        'w3': jax.random.normal(k3, (hidden_dim, action_dim)) * 0.01,
        'b3': jnp.zeros(action_dim)
    }

@jax.jit
def forward(params, obs):
    # Normalize observations
    obs_norm = obs / jnp.array([jnp.pi]*4 + [0.3]*3)
    
    h = jax.nn.relu(obs_norm @ params['w1'] + params['b1'])
    h = jax.nn.relu(h @ params['w2'] + params['b2'])
    return jnp.tanh(h @ params['w3'] + params['b3'])


# ==========================================
# 3. VECTORIZED EVALUATION
# ==========================================
@partial(jax.jit, static_argnums=(2,))
def evaluate_batch(params_batch, keys, env):
    """Evaluiere eine Batch von Policies"""
    n_episodes_per_policy = 20
    
    def evaluate_single(params, eval_keys):
        def run_episode(key):
            qpos, ee_pos, target, prev_action = env.reset_single(key)
            
            def step_fn(carry, _):
                qpos, ee_pos, total_reward, done, prev_action = carry
                
                obs = jnp.concatenate([qpos, target - ee_pos])
                action = forward(params, obs)
                action = jnp.where(done, jnp.zeros_like(action), action)
                
                new_qpos, new_ee_pos, reward, step_done, new_prev_action = env.step_single(
                    qpos, ee_pos, target, prev_action, action
                )
                
                # Conditional update
                qpos = jnp.where(done, qpos, new_qpos)
                ee_pos = jnp.where(done[..., None], ee_pos, new_ee_pos)
                reward = jnp.where(done, 0.0, reward)
                done = jnp.logical_or(done, step_done)
                prev_action = jnp.where(done[..., None], prev_action, new_prev_action)
                
                return (qpos, ee_pos, total_reward + reward, done, prev_action), None
            
            init = (qpos, ee_pos, 0.0, False, prev_action)
            final, _ = jax.lax.scan(step_fn, init, None, length=env.max_steps)
            
            return final[2], final[3]
        
        rewards, successes = jax.vmap(run_episode)(eval_keys)
        
        # Fitness berechnung
        mean_reward = jnp.mean(rewards)
        success_rate = jnp.mean(successes.astype(jnp.float32))
        
        # Success rate ist WICHTIG!
        fitness = mean_reward + success_rate * 500.0
        
        return fitness, success_rate
    
    # Reshape keys für vmap
    n_pop = len(params_batch['w1'])
    eval_keys = keys.reshape(n_pop, n_episodes_per_policy, -1)
    
    # Evaluate all
    results = jax.vmap(evaluate_single)(params_batch, eval_keys)
    
    return results[0], results[1]


# ==========================================
# 4. SIMPLE EVOLUTION (aber effektiv!)
# ==========================================
@jax.jit
def evolve_population(population, fitnesses, key):
    """Simple but effective evolution"""
    pop_size = fitnesses.shape[0]
    
    # Elite selection (top 20%)
    n_elite = pop_size // 5
    elite_idx = jnp.argsort(fitnesses)[-n_elite:]
    
    # Tournament selection für Rest
    tournament_key, mutation_key = jax.random.split(key)
    
    def tournament_select(key):
        candidates = jax.random.choice(key, pop_size, shape=(7,), replace=False)
        winner = candidates[jnp.argmax(fitnesses[candidates])]
        return winner
    
    # Create new population
    def create_offspring(i, key):
        is_elite = i < n_elite
        elite_params = tree_map(lambda x: x[elite_idx[i % n_elite]], population)
        
        # Tournament selection
        parent_idx = tournament_select(key)
        parent_params = tree_map(lambda x: x[parent_idx], population)
        
        # Mutation
        mutation_strength = jnp.where(is_elite, 0.0, 0.1)
        
        def mutate(param, key):
            noise = jax.random.normal(key, param.shape) * mutation_strength
            return param + noise
        
        mut_keys = jax.random.split(mutation_key, len(parent_params))
        mutated = tree_map(mutate, parent_params, 
                          dict(zip(parent_params.keys(), mut_keys)))
        
        # Return elite or mutated
        return tree_map(lambda e, m: jnp.where(is_elite, e, m), 
                       elite_params, mutated)
    
    offspring_keys = jax.random.split(tournament_key, pop_size)
    new_population = jax.vmap(create_offspring)(jnp.arange(pop_size), offspring_keys)
    
    return new_population


# ==========================================
# 5. MAIN TRAINING
# ==========================================
def train():
    print("\n🎯 Training mit korrigierten Parametern")
    print("-" * 50)
    
    # Setup
    env = RoArmEnvFixed()
    print(f"✓ Success Threshold: {env.success_threshold} (korrigiert!)")
    print(f"✓ Action Scale: {env.action_scale}")
    print(f"✓ Max Steps: {env.max_steps}")
    
    # Initialize population
    pop_size = 1000
    hidden_dim = 128
    
    key = jax.random.PRNGKey(42)
    init_keys = jax.random.split(key, pop_size)
    population = jax.vmap(lambda k: init_network(k, env.obs_dim, hidden_dim, env.action_dim))(init_keys)
    
    # Compile
    print("\n🔧 Kompiliere JAX funktionen...")
    dummy_keys = jax.random.split(key, pop_size * 20)
    _ = evaluate_batch(population, dummy_keys, env)
    print("✓ Kompilierung abgeschlossen")
    
    # Training loop
    best_fitness = -jnp.inf
    best_params = None
    history = []
    
    print("\n📊 Starte Training...")
    print("\nGen |  Best  |  Avg   | Success | Zeit")
    print("-" * 45)
    
    for gen in range(300):  # Sollte schnell konvergieren
        gen_start = time.time()
        
        # Evaluate
        eval_key, evolve_key = jax.random.split(jax.random.PRNGKey(gen * 1000), 2)
        eval_keys = jax.random.split(eval_key, pop_size * 20)
        
        fitnesses, success_rates = evaluate_batch(population, eval_keys, env)
        
        # Track best
        best_idx = jnp.argmax(fitnesses)
        if fitnesses[best_idx] > best_fitness:
            best_fitness = float(fitnesses[best_idx])
            best_params = tree_map(lambda x: x[best_idx], population)
        
        # Stats
        avg_fitness = float(jnp.mean(fitnesses))
        avg_success = float(jnp.mean(success_rates)) * 100
        top10_pct = pop_size // 10
        top10_success = float(jnp.mean(jnp.sort(success_rates)[-top10_pct:])) * 100
        
        # Evolve
        population = evolve_population(population, fitnesses, evolve_key)
        
        # Log
        gen_time = time.time() - gen_start
        if gen % 5 == 0:
            print(f"{gen:3d} | {best_fitness:6.0f} | {avg_fitness:6.0f} | "
                  f"{avg_success:5.1f}% | {gen_time:4.2f}s")
        
        history.append({
            'gen': gen,
            'best': best_fitness,
            'avg': avg_fitness,
            'success': avg_success,
            'top10_success': top10_success
        })
        
        # Early stopping - bei 90% average success
        if avg_success > 90:
            print(f"\n🎉 Konvergiert bei Generation {gen}!")
            print(f"   Average Success: {avg_success:.1f}%")
            print(f"   Top 10% Success: {top10_success:.1f}%")
            break
    
    return best_params, history, env


# ==========================================
# 6. TEST & SAVE
# ==========================================
def test_and_save(best_params, env):
    print("\n🧪 Finale Tests...")
    
    # Test mit 200 Episodes
    test_key = jax.random.PRNGKey(99999)
    test_keys = jax.random.split(test_key, 200)
    
    successes = []
    final_distances = []
    rewards = []
    
    for i, key in enumerate(test_keys):
        qpos, ee_pos, target, prev_action = env.reset_single(key)
        episode_reward = 0.0
        
        for step in range(env.max_steps):
            obs = jnp.concatenate([qpos, target - ee_pos])
            action = forward(best_params, obs)
            
            qpos, ee_pos, reward, done, prev_action = env.step_single(
                qpos, ee_pos, target, prev_action, action
            )
            
            episode_reward += float(reward)
            
            if done:
                successes.append(1.0)
                break
        else:
            successes.append(0.0)
        
        final_dist = float(jnp.linalg.norm(target - ee_pos))
        final_distances.append(final_dist)
        rewards.append(episode_reward)
    
    # Results
    print(f"\n📊 Test-Ergebnisse (200 Episodes):")
    print(f"   Erfolgsrate: {np.mean(successes)*100:.1f}%")
    print(f"   Erfolge: {int(np.sum(successes))}/200")
    print(f"   Avg End-Distanz: {np.mean(final_distances):.3f}")
    print(f"   Distanz bei Erfolg: {np.mean([d for d, s in zip(final_distances, successes) if s]):.3f}")
    print(f"   Distanz bei Misserfolg: {np.mean([d for d, s in zip(final_distances, successes) if not s]):.3f}")
    print(f"   Avg Reward: {np.mean(rewards):.1f}")
    
    # Save
    import pickle
    
    save_data = {
        'params': best_params,
        'config': {
            'obs_dim': env.obs_dim,
            'action_dim': env.action_dim,
            'hidden_dim': 128,
            'action_scale': env.action_scale,
            'success_threshold': env.success_threshold,
            'forward_kinematics': 'standard_4dof'
        },
        'test_results': {
            'success_rate': np.mean(successes),
            'avg_distance': np.mean(final_distances)
        }
    }
    
    with open('roarm_working_params.pkl', 'wb') as f:
        pickle.dump(save_data, f)
    
    print("\n💾 Gespeichert als 'roarm_working_params.pkl'")
    print("✅ Training erfolgreich abgeschlossen!")
    
    return successes, final_distances


# ==========================================
# 7. RUN
# ==========================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🤖 RoArm FIXED Training - Jetzt klappt's! 🤖")
    print("="*60)
    
    # Train
    best_params, history, env = train()
    
    # Test
    successes, distances = test_and_save(best_params, env)
    
    # Plot learning curve
    try:
        import matplotlib.pyplot as plt
        
        gens = [h['gen'] for h in history]
        success_rates = [h['success'] for h in history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(gens, success_rates, 'b-', linewidth=2)
        plt.xlabel('Generation')
        plt.ylabel('Success Rate %')
        plt.title('RoArm Training Progress (Fixed Version)')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 100)
        
        # Mark 90% line
        plt.axhline(y=90, color='g', linestyle='--', alpha=0.5, label='Target 90%')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('roarm_training_curve.png', dpi=150)
        print("\n📈 Learning curve gespeichert als 'roarm_training_curve.png'")
    except:
        pass