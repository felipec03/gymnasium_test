"""
Simple Q-Learning example with the GridWorld environment
This demonstrates how to train a simple RL agent on your custom environment
"""
import gymnasium as gym
import gymnasium_env  # Import to register the environment
import numpy as np
import matplotlib.pyplot as plt

class SimpleQLearningAgent:
    """A simple Q-Learning agent for the GridWorld environment"""
    
    def __init__(self, env, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        
        # Initialize Q-table
        # State space: agent_x, agent_y, target_x, target_y
        grid_size = env.unwrapped.size
        self.q_table = np.zeros((grid_size, grid_size, grid_size, grid_size, env.action_space.n))
        
    def get_state_index(self, observation):
        """Convert observation to state index for Q-table"""
        agent_pos = observation['agent']
        target_pos = observation['target']
        return tuple(agent_pos) + tuple(target_pos)
    
    def choose_action(self, observation):
        """Choose action using epsilon-greedy policy"""
        state = self.get_state_index(observation)
        
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()  # Explore
        else:
            return np.argmax(self.q_table[state])  # Exploit
    
    def update_q_table(self, state, action, reward, next_state, done):
        """Update Q-table using Q-learning update rule"""
        current_q = self.q_table[state + (action,)]
        
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])
        
        self.q_table[state + (action,)] += self.lr * (target - current_q)

def train_agent(episodes=1000):
    """Train the Q-Learning agent"""
    print("🤖 Training Q-Learning Agent on GridWorld")
    print("=" * 45)
    
    # Create environment
    env = gym.make("gymnasium_env/GridWorld-v0", render_mode=None, size=5)
    agent = SimpleQLearningAgent(env)
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    success_rate = []
    
    for episode in range(episodes):
        observation, info = env.reset()
        state = agent.get_state_index(observation)
        
        total_reward = 0
        steps = 0
        max_steps = 100
        
        while steps < max_steps:
            action = agent.choose_action(observation)
            next_observation, reward, terminated, truncated, info = env.step(action)
            next_state = agent.get_state_index(next_observation)
            
            # Update Q-table
            agent.update_q_table(state, action, reward, next_state, terminated)
            
            total_reward += reward
            steps += 1
            
            if terminated:
                break
            
            state = next_state
            observation = next_observation
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        # Calculate success rate over last 100 episodes
        if episode >= 99:
            recent_successes = sum(1 for r in episode_rewards[-100:] if r > 0)
            success_rate.append(recent_successes / 100.0)
        
        # Print progress
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_length = np.mean(episode_lengths[-100:])
            current_success = success_rate[-1] if success_rate else 0
            print(f"Episode {episode + 1:4d} | Avg Reward: {avg_reward:.3f} | Avg Length: {avg_length:.1f} | Success Rate: {current_success:.2f}")
    
    env.close()
    return agent, episode_rewards, episode_lengths, success_rate

def test_trained_agent(agent, num_episodes=10):
    """Test the trained agent"""
    print(f"\n🧪 Testing Trained Agent ({num_episodes} episodes)")
    print("=" * 35)
    
    env = gym.make("gymnasium_env/GridWorld-v0", render_mode=None, size=5)
    agent.epsilon = 0  # No exploration during testing
    
    test_rewards = []
    test_lengths = []
    
    for episode in range(num_episodes):
        observation, info = env.reset()
        total_reward = 0
        steps = 0
        max_steps = 50
        
        while steps < max_steps:
            action = agent.choose_action(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            steps += 1
            
            if terminated:
                break
        
        test_rewards.append(total_reward)
        test_lengths.append(steps)
        
        success = "✅" if total_reward > 0 else "❌"
        print(f"Test {episode + 1:2d}: {success} Reward: {total_reward} | Steps: {steps}")
    
    success_rate = sum(1 for r in test_rewards if r > 0) / num_episodes
    avg_length = np.mean([l for r, l in zip(test_rewards, test_lengths) if r > 0])
    
    print(f"\n📊 Test Results:")
    print(f"Success Rate: {success_rate:.1%}")
    print(f"Avg Steps (successful episodes): {avg_length:.1f}")
    
    env.close()
    return test_rewards, test_lengths

def plot_training_progress(episode_rewards, episode_lengths, success_rate):
    """Plot training progress"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('GridWorld Q-Learning Training Progress', fontsize=16)
    
    # Plot 1: Episode rewards (moving average)
    window = 100
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        axes[0, 0].plot(range(window-1, len(episode_rewards)), moving_avg)
    axes[0, 0].set_title('Episode Rewards (100-episode moving average)')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Average Reward')
    axes[0, 0].grid(True)
    
    # Plot 2: Episode lengths (moving average)
    if len(episode_lengths) >= window:
        moving_avg_length = np.convolve(episode_lengths, np.ones(window)/window, mode='valid')
        axes[0, 1].plot(range(window-1, len(episode_lengths)), moving_avg_length)
    axes[0, 1].set_title('Episode Length (100-episode moving average)')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Average Steps')
    axes[0, 1].grid(True)
    
    # Plot 3: Success rate
    if success_rate:
        axes[1, 0].plot(range(99, len(episode_rewards)), success_rate)
    axes[1, 0].set_title('Success Rate (over last 100 episodes)')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Success Rate')
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].grid(True)
    
    # Plot 4: Reward distribution
    axes[1, 1].hist(episode_rewards, bins=30, alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Distribution of Episode Rewards')
    axes[1, 1].set_xlabel('Reward')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_progress.png', dpi=300, bbox_inches='tight')
    print("📈 Training plots saved as 'training_progress.png'")
    plt.show()

if __name__ == "__main__":
    print("🎯 GridWorld Q-Learning Training Example")
    print("=" * 50)
    
    # Train the agent
    trained_agent, rewards, lengths, success_rates = train_agent(episodes=1000)
    
    # Test the trained agent
    test_rewards, test_lengths = test_trained_agent(trained_agent)
    
    # Plot results
    try:
        plot_training_progress(rewards, lengths, success_rates)
    except Exception as e:
        print(f"Note: Could not generate plots: {e}")
    
    print("\n✅ Training and testing completed!")
    print("\nYou can now:")
    print("1. Run 'python visual_demo.py' to see the environment visually")
    print("2. Modify the training parameters to experiment with different settings")
    print("3. Try integrating with more advanced RL libraries like Stable-Baselines3")
