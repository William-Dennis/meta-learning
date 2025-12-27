import numpy as np
import matplotlib.pyplot as plt
from core.tuning_env import TuningEnv
from core.ppo_agent import PPOAgent
from core.sa_algorithms import python_serial


def init_memory():
    """Initialize memory dictionary for PPO."""
    return {
        'states': [], 'actions': [], 'log_probs': [],
        'rewards': [], 'is_terminals': []
    }


def train_step(env, agent, memory):
    """Execute one training step."""
    state, _ = env.reset()
    action, log_prob, _ = agent.select_action(state)
    _, reward, _, _, info = env.step(action)
    
    reward_scaled = reward * 0.1
    memory['states'].append(state)
    memory['actions'].append(action)
    memory['log_probs'].append(log_prob)
    memory['rewards'].append(reward_scaled)
    memory['is_terminals'].append(True)
    
    return reward, info


def print_progress(i, history_rewards, history_params, step):
    """Print training progress."""
    recent = history_rewards[-step:]
    avg_r = np.mean(recent)
    params_recent = history_params[-step:]
    avg_c = np.mean([p['mean_cost'] for p in params_recent])
    print(f"Trial {i} | Reward: {avg_r:.2f} | Cost: {-avg_c:.2f}")


def init_training():
    """Initialize training components."""
    env = TuningEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = PPOAgent(
        state_dim, action_dim, lr=3e-4,
        gamma=0.0, eps_clip=0.2, k_epochs=4
    )
    
    return env, agent


def train_tuner():
    """Train PPO agent to tune SA hyperparameters."""
    env, agent = init_training()
    
    max_episodes = 1000
    update_timestep = 10
    memory = init_memory()
    history_rewards = []
    history_params = []
    
    print("Starting tuning...")
    for i in range(max_episodes):
        reward, info = train_step(env, agent, memory)
        
        if (i + 1) % update_timestep == 0:
            agent.update(memory)
            memory = init_memory()
        
        history_rewards.append(reward)
        history_params.append(info)
        
        if i % update_timestep == 0:
            print_progress(
                i, history_rewards, history_params, update_timestep
            )
    
    return agent, history_rewards, history_params


def plot_training_curve(rewards):
    """Plot PPO training progress."""
    plt.figure()
    plt.plot(rewards)
    plt.title('PPO Tuning Progress')
    plt.xlabel('Trial')
    plt.ylabel('Reward')
    plt.savefig('outputs/tuning_curve.png')
    plt.close()
    print("Saved outputs/tuning_curve.png")


def plot_param_evolution(params_history):
    """Plot evolution of SA parameters during training."""
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
    plt.title('Num Steps')
    plt.tight_layout()
    plt.savefig('outputs/params_evolution.png')
    plt.close()
    print("Saved outputs/params_evolution.png")


def create_meshgrid():
    """Create meshgrid for Rastrigin visualization."""
    x = np.linspace(-5.12, 5.12, 100)
    y = np.linspace(-5.12, 5.12, 100)
    X, Y = np.meshgrid(x, y)
    Z = python_serial.rastrigin_2d(X, Y)
    return X, Y, Z


def plot_trajectory(params, idx, pct, bounds):
    """Plot SA trajectory for given parameters."""
    X, Y, Z = create_meshgrid()
    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.9)
    plt.colorbar(label='Cost')
    
    for r in range(10):
        _, _, traj, _ = python_serial.run_sa(
            params['init_temp'], params['cooling_rate'],
            params['step_size'], params['num_steps'],
            bounds, seed=None
        )
        traj = np.array(traj)
        if len(traj) > 0:
            plt.plot(traj[:, 0], traj[:, 1], 'w-', alpha=0.4, linewidth=1)
            plt.plot(traj[-1, 0], traj[-1, 1], 'rx', markersize=6)
    
    title = f"Trajectory {pct}% (Ep {idx})\n"
    title += f"T0={params['init_temp']:.1f}, α={params['cooling_rate']:.3f}, "
    title += f"Step={params['step_size']:.2f}, N={params['num_steps']}"
    plt.title(title)
    plt.xlim(bounds[0], bounds[1])
    plt.ylim(bounds[0], bounds[1])
    filename = f'outputs/trajectory_evolution_{pct}pct.png'
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")


def visualize_trajectory_evolution(params_history):
    """Visualize trajectory at different training stages."""
    n = len(params_history)
    checkpoints = [0, int(n*0.25), int(n*0.50), int(n*0.75), n-1]
    percentages = [0, 25, 50, 75, 100]
    bounds = [-5.12, 5.12]
    
    for i, idx in enumerate(checkpoints):
        if idx >= n:
            idx = n - 1
        params = params_history[idx]
        plot_trajectory(params, idx, percentages[i], bounds)


def visualize_2d(env, title="Optimization Trajectory"):
    """Visualize 2D trajectory of SA optimization."""
    X, Y, Z = create_meshgrid()
    plt.figure(figsize=(10, 8))
    plt.contourf(X, Y, Z, levels=50, cmap='viridis')
    plt.colorbar(label='Cost')
    
    traj = np.array(env.last_trajectory)
    if len(traj) > 0:
        plt.plot(traj[:, 0], traj[:, 1], 'r.-', alpha=0.6, label='Path')
        plt.plot(traj[0, 0], traj[0, 1], 'go', markersize=10, label='Start')
        plt.plot(traj[-1, 0], traj[-1, 1], 'bx', markersize=10, label='End')
    
    plt.title(title)
    plt.legend()
    plt.savefig('outputs/trajectory_2d.png')
    plt.close()
    print("Saved outputs/trajectory_2d.png")


def evaluate_and_plot(agent):
    """Evaluate trained agent and create visualizations."""
    env = TuningEnv()
    print("\nEvaluating learned params...")
    state, _ = env.reset(seed=42)
    
    costs = []
    for _ in range(10):
        action, _, _ = agent.select_action(state)
        _, reward, _, _, info = env.step(action)
        costs.append(info['final_cost'])
    
    print(f"Mean Cost (10 runs): {np.mean(costs):.4f}")
    print(f"Params: T0={info['init_temp']:.2f}, "
          f"Alpha={info['cooling_rate']:.3f}, "
          f"Step={info['step_size']:.2f}")
    
    title = f"PPO Tuned SA (Cost: {costs[-1]:.2f})"
    visualize_2d(env, title)
    
    return costs, info


if __name__ == '__main__':
    agent, rewards, params_history = train_tuner()
    plot_training_curve(rewards)
    plot_param_evolution(params_history)
    visualize_trajectory_evolution(params_history)
    evaluate_and_plot(agent)
