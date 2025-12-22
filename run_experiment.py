
import numpy as np
import torch
import matplotlib.pyplot as plt
from tuning_env import TuningEnv
from ppo_agent import PPOAgent
import pandas as pd

def train_tuner():
    env = TuningEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Lower learning rate for stable convergence on params
    agent = PPOAgent(state_dim, action_dim, lr=3e-4, gamma=0.0, eps_clip=0.2, k_epochs=4)
    
    max_episodes = 1000 # Enough for convergence demo
    update_timestep = 2 # Update every 20 episodes (batch size 20)
    
    memory = {'states': [], 'actions': [], 'log_probs': [], 'rewards': [], 'is_terminals': []}
    
    history_rewards = []
    history_params = []
    
    print("Starting tuning...")
    for i in range(max_episodes):
        state, _ = env.reset()
        
        action, log_prob, _ = agent.select_action(state)
        # Add exploration noise if needed, but PPO handles stochastic policy.
        
        _, reward, done, _, info = env.step(action)
        
        # Reward Scaling: Rastrigin can be ~80. Scale to ~8.
        reward_scaled = reward * 0.1 
        
        memory['states'].append(state)
        memory['actions'].append(action) 
        memory['log_probs'].append(log_prob)
        memory['rewards'].append(reward_scaled)
        memory['is_terminals'].append(True) # Always terminal
        
        if (i + 1) % update_timestep == 0:
            agent.update(memory)
            memory = {'states': [], 'actions': [], 'log_probs': [], 'rewards': [], 'is_terminals': []}
        
        history_rewards.append(reward)
        history_params.append(info)
        
        verbose_step_size = max(update_timestep, 10)
        # Print less frequently to avoid spam
        if i % verbose_step_size == 0:
            avg_r = np.mean(history_rewards[-verbose_step_size:])
            avg_c = np.mean([p['mean_cost'] for p in history_params[-verbose_step_size:]])
            print(f"Trial {i} | Reward: {avg_r:.2f} | Cost: {-avg_c:.2f}\tParams: T0={info['init_temp']:.2f}, Alpha={info['cooling_rate']:.3f}, Step={info['step_size']:.2f}, #Steps={info['num_steps']}")

    return agent, history_rewards, history_params

from sa_algorithm import rastrigin_2d, run_sa

def visualize_trajectory_evolution(params_history):
    # Checkpoints: 0%, 25%, 50%, 75%, 100%
    n = len(params_history)
    checkpoints = [0, int(n*0.25), int(n*0.50), int(n*0.75), n-1]
    percentages = [0, 25, 50, 75, 100]
    
    # Create meshgrid for background
    x = np.linspace(-5.12, 5.12, 100)
    y = np.linspace(-5.12, 5.12, 100)
    X, Y = np.meshgrid(x, y)
    Z = rastrigin_2d(X, Y)
    
    bounds = [-5.12, 5.12]
    
    for i, idx in enumerate(checkpoints):
        # Allow for edge case where list is short
        if idx >= n: idx = n - 1
        
        params = params_history[idx]
        
        plt.figure(figsize=(8, 6))
        plt.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.9)
        plt.colorbar(label='Cost')
        
        # Run 10 times to show stochasticity
        for r in range(10):
            # params contains: init_temp, cooling_rate, step_size, num_steps
            _, _, last_trajectory, _ = run_sa(
                params['init_temp'], 
                params['cooling_rate'], 
                params['step_size'], 
                params['num_steps'], 
                bounds, 
                seed=None # Random seed
            )
            traj = np.array(last_trajectory)
            if len(traj) > 0:
                plt.plot(traj[:, 0], traj[:, 1], 'w-', alpha=0.4, linewidth=1)
                plt.plot(traj[-1, 0], traj[-1, 1], 'rx', markersize=6) # End point
        
        plt.title(f"Trajectory Evolution {percentages[i]}% (Ep {idx})\n"
                  f"T0={params['init_temp']:.1f}, α={params['cooling_rate']:.3f}, "
                  f"Step={params['step_size']:.2f}, N={params['num_steps']}")
        plt.xlim(bounds[0], bounds[1])
        plt.ylim(bounds[0], bounds[1])
        plt.savefig(f'trajectory_evolution_{percentages[i]}pct.png')
        plt.close()
        print(f"Saved trajectory_evolution_{percentages[i]}pct.png")

def visualize_2d(env, title="Optimization Trajectory"):
    # Create meshgrid for Rastrigin
    x = np.linspace(-5.12, 5.12, 100)
    y = np.linspace(-5.12, 5.12, 100)
    X, Y = np.meshgrid(x, y)
    Z = rastrigin_2d(X, Y)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(X, Y, Z, levels=50, cmap='viridis')
    plt.colorbar(label='Cost')
    
    # Plot trajectory
    traj = np.array(env.last_trajectory)
    if len(traj) > 0:
        plt.plot(traj[:, 0], traj[:, 1], 'r.-', alpha=0.6, label='Path')
        plt.plot(traj[0, 0], traj[0, 1], 'go', markersize=10, label='Start')
        plt.plot(traj[-1, 0], traj[-1, 1], 'bx', markersize=10, label='End')
    
    plt.title(title)
    plt.legend()
    plt.savefig('trajectory_2d.png')
    print("Saved trajectory_2d.png")

def evaluate_and_plot(agent):
    env = TuningEnv()
    
    # 1. Run with Learned Params
    print("\nEvaluating Learned Params...")
    state, _ = env.reset(seed=42) # Seed only affects random start position logic inside env effectively?
    # Actually run multiple times to verify robustness
    costs = []
    
    # We want to visualize one good run
    best_run_cost = float('inf')
    
    for _ in range(10):
        # Taking Mean action for evaluation? Or sample?
        # Usually mean for eval.
        # But we implemented sample in select_action.
        # Let's trust the policy has converged to low std dev or just sample.
        action, _, _ = agent.select_action(state)
        
        _, reward, _, _, info = env.step(action)
        costs.append(info['final_cost'])
        
        if info['final_cost'] < best_run_cost:
            best_run_cost = info['final_cost']
            # Retain this env state/history for plotting?
            # Env stores last_trajectory.
            # We need to save the trajectory if this is the best run.
            # Hack: We just re-run the viz function after the loop if we want?
            # Or just plot the last one?
            # Let's plot the last one of the loop for simplicity or a specific seed.
    
    print(f"PPO Mean Cost (10 runs): {np.mean(costs):.4f}")
    print(f"Params used (last run): T0={info['init_temp']:.2f}, Alpha={info['cooling_rate']:.3f}, Step={info['step_size']:.2f}")
    
    visualize_2d(env, title=f"PPO Tuned SA (Cost: {costs[-1]:.2f})")
    
    # 2. Random/Bad Params Comparison
    # Bad: Low Temp, Fast Cool (Quench)
    print("\nRunning Bad Params (Quench)...")
    # T0=0.1, Alpha=0.1, Step=0.1 (Small steps, no heat)
    # Action mapping inverse is hard, let's just cheat and override step?
    # Or just manually disable agent and force params in a manual run if we want to reproduce 'bad'.
    # But better to ask Env to run with specific params? 
    # TuningEnv takes actions.
    # Let's make a manual action.
    # Using new param_scaling: NN=-5 gives minimum values for all params
    # init_temp=0.1, cooling_rate=0.5, step_size=0.1, num_steps=20
    manual_action = np.array([-5.0, -5.0, -5.0, 0.0], dtype=np.float32)  # Low temp, slow cool, small step, medium steps
    _, _, _, _, info_bad = env.step(manual_action)
    print(f"Bad Params Cost: {info_bad['final_cost']:.4f}")
    
    return costs, info, info_bad

if __name__ == '__main__':
    agent, rewards, params_history = train_tuner()
    
    # Plot training
    plt.figure()
    plt.plot(rewards)
    plt.title('PPO Tuning Progress (Reward)')
    plt.xlabel('Trial')
    plt.ylabel('Negative Cost')
    plt.savefig('tuning_curve.png')
    
    # Parameter Evolution
    t0s = [p['init_temp'] for p in params_history]
    alphas = [p['cooling_rate'] for p in params_history]
    steps = [p['step_size'] for p in params_history]
    num_steps = [p['num_steps'] for p in params_history]
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 4, 1)
    plt.plot(t0s)
    plt.title('Init Temp')
    plt.subplot(1, 4, 2)
    plt.plot(alphas)
    plt.title('Cooling Rate')
    plt.subplot(1, 4, 3)
    plt.plot(steps)
    plt.title('Step Size')
    plt.subplot(1, 4, 4)
    plt.plot(num_steps)
    plt.title("Num Steps")
    plt.tight_layout()
    plt.savefig('params_evolution.png')
    
    visualize_trajectory_evolution(params_history)
    
    evaluate_and_plot(agent)
