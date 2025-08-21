"""
Visual demonstration of the GridWorld environment
This script shows the environment with visual rendering using PyGame
"""
import gymnasium as gym
import gymnasium_env  # Import to register the environment
import time

def visual_demo():
    """Demonstrate the environment with visual rendering"""
    print("🎮 Visual GridWorld Demo")
    print("=" * 30)
    print("A PyGame window will open showing:")
    print("- Blue circle: Agent")
    print("- Red rectangle: Target")
    print("- Grid lines for reference")
    print("\nThe agent will move randomly until it reaches the target.")
    print("Close the window to end the demo.")
    
    # Create environment with human rendering
    env = gym.make("gymnasium_env/GridWorld-v0", render_mode="human", size=8)
    
    observation, info = env.reset(seed=42)
    print(f"\nStarting position:")
    print(f"Agent: {observation['agent']}")
    print(f"Target: {observation['target']}")
    print(f"Initial distance: {info['distance']:.2f}")
    
    step_count = 0
    total_reward = 0
    
    try:
        while True:
            # Take a random action
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            
            step_count += 1
            total_reward += reward
            
            # Action names for display
            action_names = ['→ Right', '↑ Up', '← Left', '↓ Down']
            print(f"Step {step_count}: {action_names[action]} | Distance: {info['distance']:.1f} | Reward: {reward}")
            
            # Add a small delay to make it easier to follow
            time.sleep(0.3)
            
            if terminated:
                print(f"\n🎯 SUCCESS! Target reached in {step_count} steps!")
                print(f"Total reward: {total_reward}")
                time.sleep(2)  # Keep window open for 2 seconds to see the final state
                break
                
            if step_count > 400:  # Safety limit
                print("\nReached step limit. Ending demo.")
                break
                
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    except Exception as e:
        print(f"Error during demo: {e}")
    finally:
        env.close()
        print("Window closed. Demo ended.")

if __name__ == "__main__":
    visual_demo()
