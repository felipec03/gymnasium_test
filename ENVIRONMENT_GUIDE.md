# GridWorld Environment - Complete Guide

## 📋 Overview

Your custom Gymnasium RL environment implements a **GridWorld** - a classic reinforcement learning environment where an agent must navigate a 2D grid to reach a target location.

## 🏗️ Implementation Details

### Core Environment (`GridWorldEnv`)

**Location**: `gymnasium_env/envs/grid_world.py`

**Key Features**:
- **Grid Size**: Configurable (default 5x5)
- **Agent**: Blue circle that moves in 4 directions
- **Target**: Red rectangle placed randomly
- **Observation Space**: Dictionary with agent and target positions as 2D coordinates
- **Action Space**: Discrete(4) - [right=0, up=1, left=2, down=3]
- **Reward System**: Binary sparse rewards (1 when target reached, 0 otherwise)
- **Rendering**: PyGame visualization + RGB array support
- **Episode Termination**: When agent reaches target

### Available Wrappers

1. **ClipReward**: Clips rewards to specified min/max range
2. **DiscreteActions**: Restricts action space to a subset  
3. **RelativePosition**: Computes relative position between agent and target
4. **ReacherRewardWrapper**: Weights reward terms for reacher environments

## 🚀 How to Run

### 1. Environment Setup

Your environment is already installed! The installation was completed with:
```bash
pip install -e .
```

### 2. Basic Usage

```python
import gymnasium as gym
import gymnasium_env  # Required to register the environment

# Create environment
env = gym.make("gymnasium_env/GridWorld-v0", render_mode="human", size=5)

# Reset and run
observation, info = env.reset()
action = env.action_space.sample()
observation, reward, terminated, truncated, info = env.step(action)
```

### 3. Available Scripts

**Test the environment**:
```bash
python test_gridworld.py
```

**Visual demonstration**:
```bash
python visual_demo.py
```

**RL Training example**:
```bash
python rl_training_example.py
```

## 🧪 Test Results

### Basic Functionality ✅
- Environment registration: Working
- Observation/action spaces: Correctly defined
- Reset/step functions: Functioning properly
- Reward system: Binary rewards (0 until target reached, then 1)

### Q-Learning Training Results ✅
- **Training Episodes**: 1000
- **Final Success Rate**: 57%
- **Average Steps**: 46 (successful episodes)
- **Learning Progress**: Agent learned to navigate more efficiently over time

### Visual Rendering ✅
- PyGame window displays correctly
- Agent (blue circle) and target (red rectangle) visible
- Grid lines provide clear spatial reference
- Real-time updates during gameplay

## 🎯 Environment Characteristics

### Strengths
- **Simple but Complete**: Perfect for learning RL concepts
- **Configurable**: Grid size can be adjusted
- **Well-structured**: Follows Gymnasium standards
- **Visual**: Clear PyGame rendering
- **Extensible**: Multiple wrapper examples included

### Use Cases
- **RL Education**: Ideal for learning basic RL algorithms
- **Algorithm Testing**: Simple environment for testing new methods
- **Research**: Baseline environment for navigation tasks
- **Prototyping**: Quick testing of RL ideas

## 🛠️ Customization Options

### Environment Parameters
```python
env = gym.make("gymnasium_env/GridWorld-v0", 
               render_mode="human",  # "human", "rgb_array", or None
               size=8)               # Grid size (NxN)
```

### Reward Modification
You can modify the reward structure in `grid_world.py`:
```python
# Current: Binary rewards
reward = 1 if terminated else 0

# Alternative: Distance-based rewards
reward = -np.linalg.norm(self._agent_location - self._target_location)
```

### Adding Obstacles
The environment can be extended to include obstacles by modifying the `step()` method to check for collision with obstacle positions.

## 📈 Integration with RL Libraries

### Stable-Baselines3 Example
```python
import gymnasium as gym
import gymnasium_env
from stable_baselines3 import PPO

env = gym.make("gymnasium_env/GridWorld-v0")
model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
```

### Custom Training Loop
See `rl_training_example.py` for a complete Q-Learning implementation that achieved 57% success rate.

## 🔧 Next Steps

1. **Experiment with Parameters**: Try different grid sizes and reward structures
2. **Add Complexity**: Include obstacles, multiple targets, or time limits  
3. **Advanced RL**: Test with modern algorithms like PPO, DQN, or A3C
4. **Multi-agent**: Extend to multiple agents sharing the same grid
5. **Curriculum Learning**: Start with smaller grids and gradually increase difficulty

## 📁 File Structure
```
gymnasium_test/
├── gymnasium_env/
│   ├── __init__.py                    # Environment registration
│   ├── envs/
│   │   ├── __init__.py
│   │   └── grid_world.py              # Main environment implementation
│   └── wrappers/
│       ├── __init__.py
│       ├── clip_reward.py             # Reward clipping wrapper
│       ├── discrete_actions.py        # Action space restriction
│       ├── relative_position.py       # Observation transformation
│       └── reacher_weighted_reward.py # Reward weighting
├── test_gridworld.py                  # Basic functionality tests
├── visual_demo.py                     # Visual demonstration
├── rl_training_example.py             # Q-Learning training example
├── training_progress.png              # Generated training plots
└── pyproject.toml                     # Package configuration
```

Your environment is working perfectly and ready for RL experimentation! 🎉
