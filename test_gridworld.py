"""
Test script for the custom GridWorld environment
This script demonstrates how to use your custom Gymnasium environment
"""
import gymnasium as gym
import gymnasium_env  # Import to register the environment
import numpy as np
from gymnasium_env.wrappers import ClipReward

def test_basic_environment():
    """Test the basic GridWorld environment functionality"""
    print("=== Testing Basic GridWorld Environment ===")
    
    # Create the environment
    env = gym.make("gymnasium_env/GridWorld-v0", render_mode="human", size=5)
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Action space size: {env.action_space.n}")
    
    # Reset the environment
    observation, info = env.reset(seed=42)
    print(f"Initial observation: {observation}")
    print(f"Initial info: {info}")
    
    # Run a few random steps
    total_reward = 0
    steps = 0
    max_steps = 50
    
    while steps < max_steps:
        # Take a random action
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        steps += 1
        
        print(f"Step {steps}: Action={action}, Reward={reward}, Distance={info['distance']:.2f}")
        print(f"  Agent: {observation['agent']}, Target: {observation['target']}")
        
        if terminated:
            print(f"✅ Episode completed! Agent reached the target in {steps} steps!")
            break
        
        if truncated:
            print("Episode truncated (max steps reached)")
            break
    
    print(f"Total reward: {total_reward}")
    print(f"Final distance to target: {info['distance']:.2f}")
    
    env.close()
    return total_reward, steps

def test_with_wrapper():
    """Test the environment with a reward wrapper"""
    print("\n=== Testing GridWorld with ClipReward Wrapper ===")
    
    # Create environment with wrapper
    base_env = gym.make("gymnasium_env/GridWorld-v0", render_mode=None, size=4)
    env = ClipReward(base_env, min_reward=-1, max_reward=1)
    
    print(f"Wrapped reward range: {env.reward_range}")
    
    observation, info = env.reset(seed=123)
    
    for step in range(10):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        
        print(f"Step {step+1}: Reward={reward} (clipped)")
        
        if terminated:
            print("Episode completed with wrapper!")
            break
    
    env.close()

def test_manual_control():
    """Demonstrate manual control of the agent"""
    print("\n=== Manual Control Demo ===")
    
    env = gym.make("gymnasium_env/GridWorld-v0", render_mode=None, size=3)
    observation, info = env.reset(seed=456)
    
    print(f"Starting position - Agent: {observation['agent']}, Target: {observation['target']}")
    
    # Define a simple strategy: move towards target
    actions_taken = []
    
    for step in range(10):
        agent_pos = observation['agent']
        target_pos = observation['target']
        
        # Simple heuristic: move towards target
        diff = target_pos - agent_pos
        
        if diff[0] > 0:  # Target is to the right
            action = 0  # Move right
        elif diff[0] < 0:  # Target is to the left
            action = 2  # Move left
        elif diff[1] > 0:  # Target is above
            action = 1  # Move up
        elif diff[1] < 0:  # Target is below
            action = 3  # Move down
        else:
            action = env.action_space.sample()  # Random if already at target
        
        actions_taken.append(action)
        observation, reward, terminated, truncated, info = env.step(action)
        
        action_names = ['right', 'up', 'left', 'down']
        print(f"Step {step+1}: Moved {action_names[action]} -> Agent: {observation['agent']}, Distance: {info['distance']:.2f}")
        
        if terminated:
            print(f"🎯 Target reached in {step+1} steps using heuristic strategy!")
            break
    
    print(f"Actions taken: {[['right', 'up', 'left', 'down'][a] for a in actions_taken]}")
    env.close()

if __name__ == "__main__":
    print("🎮 Testing Custom GridWorld Environment")
    print("=" * 50)
    
    try:
        # Test basic functionality
        reward, steps = test_basic_environment()
        
        # Test with wrapper
        test_with_wrapper()
        
        # Test manual control
        test_manual_control()
        
        print("\n✅ All tests completed successfully!")
        print("\nTo run with visual rendering, the environment will open a PyGame window.")
        print("You can also create your own training loops or integrate with RL libraries like Stable-Baselines3.")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
