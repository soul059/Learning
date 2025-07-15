# 06. Reinforcement Learning

## 🎯 Learning Objectives
- Understand reinforcement learning fundamentals
- Master key RL algorithms and techniques
- Learn about policy and value-based methods
- Apply RL to real-world problems

---

## 1. Introduction to Reinforcement Learning

**Reinforcement Learning (RL)** is a type of machine learning where an agent learns to make decisions by interacting with an environment to maximize cumulative reward.

### 1.1 Key Concepts 🟢

#### Core Components:
- **Agent**: The learner/decision maker
- **Environment**: The world the agent interacts with
- **State (s)**: Current situation of the agent
- **Action (a)**: What the agent can do
- **Reward (r)**: Feedback signal from environment
- **Policy (π)**: Agent's strategy for choosing actions

#### RL vs Other ML Paradigms:
```
Supervised Learning: Learn from labeled examples
Unsupervised Learning: Find patterns in data
Reinforcement Learning: Learn through trial and error
```

### 1.2 The RL Framework 🟢

#### Agent-Environment Interaction:
```
At time t:
1. Agent observes state St
2. Agent selects action At based on policy π
3. Environment returns reward Rt+1 and new state St+1
4. Process repeats
```

#### Mathematical Formulation:
```
State space: S
Action space: A
Reward function: R(s,a) → ℝ
Transition function: P(s'|s,a) → [0,1]
Policy: π(a|s) → [0,1]
```

#### Episode vs Continuing Tasks:
- **Episodic**: Tasks with clear end (games, navigation)
- **Continuing**: Tasks that go on indefinitely (stock trading, robot control)

### 1.3 Markov Decision Process (MDP) 🟢

**Definition**: Mathematical framework for modeling decision-making in RL.

#### Markov Property:
```
P(St+1|St, At, St-1, At-1, ..., S0, A0) = P(St+1|St, At)
```
*Future depends only on current state, not history*

#### MDP Components:
- **S**: Set of states
- **A**: Set of actions  
- **P**: Transition probabilities
- **R**: Reward function
- **γ**: Discount factor (0 ≤ γ ≤ 1)

#### Return and Value Functions:
```
Return: Gt = Rt+1 + γRt+2 + γ²Rt+3 + ... = Σk=0∞ γᵏRt+k+1

State Value: V^π(s) = E[Gt|St = s]
Action Value: Q^π(s,a) = E[Gt|St = s, At = a]
```

---

## 2. Dynamic Programming

### 2.1 Policy Evaluation 🟢

**Goal**: Compute value function V^π for a given policy π.

#### Bellman Equation for V^π:
```
V^π(s) = Σa π(a|s) Σs' P(s'|s,a)[R(s,a,s') + γV^π(s')]
```

#### Iterative Policy Evaluation:
```python
import numpy as np

def policy_evaluation(policy, env, gamma=0.9, theta=1e-8):
    """
    Evaluate a policy in an environment
    
    Args:
        policy: Policy to evaluate π(a|s)
        env: Environment with states, actions, transitions
        gamma: Discount factor
        theta: Convergence threshold
    """
    V = np.zeros(env.num_states)
    
    while True:
        delta = 0
        for s in range(env.num_states):
            v = V[s]
            V[s] = 0
            
            for a in range(env.num_actions):
                for s_next, prob, reward in env.transitions[s][a]:
                    V[s] += policy[s][a] * prob * (reward + gamma * V[s_next])
            
            delta = max(delta, abs(v - V[s]))
        
        if delta < theta:
            break
    
    return V
```

### 2.2 Policy Improvement 🟢

**Goal**: Improve policy based on current value function.

#### Policy Improvement Theorem:
```
If Q^π(s, π'(s)) ≥ V^π(s) for all s, then π' ≥ π
```

#### Greedy Policy Improvement:
```python
def policy_improvement(V, env, gamma=0.9):
    """
    Improve policy based on value function
    """
    policy = np.zeros([env.num_states, env.num_actions])
    
    for s in range(env.num_states):
        action_values = np.zeros(env.num_actions)
        
        for a in range(env.num_actions):
            for s_next, prob, reward in env.transitions[s][a]:
                action_values[a] += prob * (reward + gamma * V[s_next])
        
        best_action = np.argmax(action_values)
        policy[s][best_action] = 1.0
    
    return policy
```

### 2.3 Policy Iteration 🟢

**Algorithm**: Alternate between policy evaluation and improvement.

```python
def policy_iteration(env, gamma=0.9, theta=1e-8):
    """
    Policy Iteration algorithm
    """
    # Initialize random policy
    policy = np.ones([env.num_states, env.num_actions]) / env.num_actions
    
    while True:
        # Policy Evaluation
        V = policy_evaluation(policy, env, gamma, theta)
        
        # Policy Improvement
        new_policy = policy_improvement(V, env, gamma)
        
        # Check for convergence
        if np.array_equal(policy, new_policy):
            break
        
        policy = new_policy
    
    return policy, V
```

### 2.4 Value Iteration 🟢

**Algorithm**: Directly iterate on Bellman optimality equation.

#### Bellman Optimality Equation:
```
V*(s) = max_a Σs' P(s'|s,a)[R(s,a,s') + γV*(s')]
```

```python
def value_iteration(env, gamma=0.9, theta=1e-8):
    """
    Value Iteration algorithm
    """
    V = np.zeros(env.num_states)
    
    while True:
        delta = 0
        for s in range(env.num_states):
            v = V[s]
            action_values = np.zeros(env.num_actions)
            
            for a in range(env.num_actions):
                for s_next, prob, reward in env.transitions[s][a]:
                    action_values[a] += prob * (reward + gamma * V[s_next])
            
            V[s] = np.max(action_values)
            delta = max(delta, abs(v - V[s]))
        
        if delta < theta:
            break
    
    # Extract optimal policy
    policy = policy_improvement(V, env, gamma)
    
    return policy, V
```

---

## 3. Monte Carlo Methods

### 3.1 Monte Carlo Prediction 🟡

**Concept**: Learn value functions from complete episodes without model knowledge.

#### First-Visit MC:
```python
def first_visit_mc_prediction(policy, env, num_episodes=1000, gamma=0.9):
    """
    First-visit Monte Carlo prediction
    """
    V = np.zeros(env.num_states)
    returns = {s: [] for s in range(env.num_states)}
    
    for episode in range(num_episodes):
        # Generate episode following policy
        states, actions, rewards = generate_episode(env, policy)
        
        # Calculate returns
        G = 0
        visited_states = set()
        
        # Work backwards through episode
        for t in reversed(range(len(states))):
            G = gamma * G + rewards[t]
            
            if states[t] not in visited_states:
                visited_states.add(states[t])
                returns[states[t]].append(G)
                V[states[t]] = np.mean(returns[states[t]])
    
    return V

def generate_episode(env, policy):
    """Generate single episode following policy"""
    states, actions, rewards = [], [], []
    state = env.reset()
    
    while not env.done:
        action = np.random.choice(env.num_actions, p=policy[state])
        next_state, reward, done = env.step(action)
        
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        
        state = next_state
    
    return states, actions, rewards
```

#### Every-Visit MC:
```python
def every_visit_mc_prediction(policy, env, num_episodes=1000, gamma=0.9):
    """
    Every-visit Monte Carlo prediction
    """
    V = np.zeros(env.num_states)
    returns = {s: [] for s in range(env.num_states)}
    
    for episode in range(num_episodes):
        states, actions, rewards = generate_episode(env, policy)
        
        G = 0
        # Work backwards through episode
        for t in reversed(range(len(states))):
            G = gamma * G + rewards[t]
            returns[states[t]].append(G)
            V[states[t]] = np.mean(returns[states[t]])
    
    return V
```

### 3.2 Monte Carlo Control 🟡

#### Exploring Starts:
```python
def mc_control_exploring_starts(env, num_episodes=1000, gamma=0.9):
    """
    Monte Carlo Control with Exploring Starts
    """
    Q = np.zeros([env.num_states, env.num_actions])
    returns = {(s, a): [] for s in range(env.num_states) 
                           for a in range(env.num_actions)}
    
    # Initialize policy (greedy w.r.t. Q)
    policy = np.zeros([env.num_states, env.num_actions])
    for s in range(env.num_states):
        policy[s][np.argmax(Q[s])] = 1.0
    
    for episode in range(num_episodes):
        # Generate episode with exploring start
        states, actions, rewards = generate_episode_exploring_starts(env, policy)
        
        G = 0
        visited_pairs = set()
        
        for t in reversed(range(len(states))):
            G = gamma * G + rewards[t]
            state_action = (states[t], actions[t])
            
            if state_action not in visited_pairs:
                visited_pairs.add(state_action)
                returns[state_action].append(G)
                Q[states[t]][actions[t]] = np.mean(returns[state_action])
                
                # Update policy (greedy)
                policy[states[t]] = 0
                policy[states[t]][np.argmax(Q[states[t]])] = 1.0
    
    return Q, policy
```

#### ε-Greedy Policy:
```python
def epsilon_greedy_policy(Q, state, epsilon=0.1):
    """
    ε-greedy action selection
    """
    if np.random.random() < epsilon:
        return np.random.choice(len(Q[state]))  # Random action
    else:
        return np.argmax(Q[state])  # Greedy action

def mc_control_epsilon_greedy(env, num_episodes=1000, epsilon=0.1, gamma=0.9):
    """
    Monte Carlo Control with ε-greedy policy
    """
    Q = np.zeros([env.num_states, env.num_actions])
    returns = {(s, a): [] for s in range(env.num_states) 
                           for a in range(env.num_actions)}
    
    for episode in range(num_episodes):
        # Generate episode
        states, actions, rewards = [], [], []
        state = env.reset()
        
        while not env.done:
            action = epsilon_greedy_policy(Q, state, epsilon)
            next_state, reward, done = env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
            state = next_state
        
        # Update Q-values
        G = 0
        visited_pairs = set()
        
        for t in reversed(range(len(states))):
            G = gamma * G + rewards[t]
            state_action = (states[t], actions[t])
            
            if state_action not in visited_pairs:
                visited_pairs.add(state_action)
                returns[state_action].append(G)
                Q[states[t]][actions[t]] = np.mean(returns[state_action])
    
    return Q
```

---

## 4. Temporal Difference Learning

### 4.1 TD(0) Prediction 🟡

**Concept**: Learn from individual steps, not complete episodes.

#### TD Update Rule:
```
V(St) ← V(St) + α[Rt+1 + γV(St+1) - V(St)]
```

```python
def td_prediction(policy, env, num_episodes=1000, alpha=0.1, gamma=0.9):
    """
    TD(0) prediction
    """
    V = np.zeros(env.num_states)
    
    for episode in range(num_episodes):
        state = env.reset()
        
        while not env.done:
            action = np.random.choice(env.num_actions, p=policy[state])
            next_state, reward, done = env.step(action)
            
            # TD update
            if done:
                target = reward
            else:
                target = reward + gamma * V[next_state]
            
            V[state] = V[state] + alpha * (target - V[state])
            state = next_state
    
    return V
```

#### TD vs MC Comparison:
- **MC**: Unbiased but high variance, requires complete episodes
- **TD**: Biased but low variance, learns from incomplete episodes

### 4.2 SARSA (On-Policy TD Control) 🟡

**Algorithm**: State-Action-Reward-State-Action

#### SARSA Update:
```
Q(St, At) ← Q(St, At) + α[Rt+1 + γQ(St+1, At+1) - Q(St, At)]
```

```python
def sarsa(env, num_episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1):
    """
    SARSA algorithm
    """
    Q = np.zeros([env.num_states, env.num_actions])
    
    for episode in range(num_episodes):
        state = env.reset()
        action = epsilon_greedy_policy(Q, state, epsilon)
        
        while not env.done:
            next_state, reward, done = env.step(action)
            
            if done:
                target = reward
            else:
                next_action = epsilon_greedy_policy(Q, next_state, epsilon)
                target = reward + gamma * Q[next_state][next_action]
            
            # SARSA update
            Q[state][action] = Q[state][action] + alpha * (target - Q[state][action])
            
            state = next_state
            if not done:
                action = next_action
    
    return Q
```

### 4.3 Q-Learning (Off-Policy TD Control) 🟡

**Key Idea**: Learn optimal policy while following different policy.

#### Q-Learning Update:
```
Q(St, At) ← Q(St, At) + α[Rt+1 + γ max_a Q(St+1, a) - Q(St, At)]
```

```python
def q_learning(env, num_episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1):
    """
    Q-Learning algorithm
    """
    Q = np.zeros([env.num_states, env.num_actions])
    
    for episode in range(num_episodes):
        state = env.reset()
        
        while not env.done:
            action = epsilon_greedy_policy(Q, state, epsilon)
            next_state, reward, done = env.step(action)
            
            if done:
                target = reward
            else:
                target = reward + gamma * np.max(Q[next_state])
            
            # Q-Learning update
            Q[state][action] = Q[state][action] + alpha * (target - Q[state][action])
            
            state = next_state
    
    return Q
```

#### SARSA vs Q-Learning:
- **SARSA**: On-policy, learns value of policy being followed
- **Q-Learning**: Off-policy, learns optimal policy regardless of behavior

### 4.4 Expected SARSA 🟡

**Improvement**: Use expected value instead of sampled next action.

```python
def expected_sarsa(env, num_episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1):
    """
    Expected SARSA algorithm
    """
    Q = np.zeros([env.num_states, env.num_actions])
    
    for episode in range(num_episodes):
        state = env.reset()
        
        while not env.done:
            action = epsilon_greedy_policy(Q, state, epsilon)
            next_state, reward, done = env.step(action)
            
            if done:
                target = reward
            else:
                # Expected value under ε-greedy policy
                expected_q = 0
                best_action = np.argmax(Q[next_state])
                
                for a in range(env.num_actions):
                    if a == best_action:
                        prob = 1 - epsilon + epsilon / env.num_actions
                    else:
                        prob = epsilon / env.num_actions
                    expected_q += prob * Q[next_state][a]
                
                target = reward + gamma * expected_q
            
            Q[state][action] = Q[state][action] + alpha * (target - Q[state][action])
            state = next_state
    
    return Q
```

---

## 5. Function Approximation

### 5.1 Linear Function Approximation 🔴

**Problem**: State/action spaces too large for tabular methods.

#### Value Function Approximation:
```
V̂(s, w) ≈ V^π(s)
Q̂(s, a, w) ≈ Q^π(s, a)
```

#### Linear Approximation:
```
V̂(s, w) = w^T φ(s)
Q̂(s, a, w) = w^T φ(s, a)
```

#### Gradient Descent Update:
```
w ← w + α[target - V̂(s, w)]∇_w V̂(s, w)
```

```python
def linear_td_prediction(env, features, num_episodes=1000, alpha=0.01, gamma=0.9):
    """
    TD prediction with linear function approximation
    """
    w = np.zeros(features.shape[1])  # Weight vector
    
    for episode in range(num_episodes):
        state = env.reset()
        
        while not env.done:
            action = random_policy(env)  # Random policy for simplicity
            next_state, reward, done = env.step(action)
            
            # Feature vectors
            phi_s = features[state]
            
            if done:
                target = reward
            else:
                phi_s_next = features[next_state]
                target = reward + gamma * np.dot(w, phi_s_next)
            
            # Gradient descent update
            prediction = np.dot(w, phi_s)
            w = w + alpha * (target - prediction) * phi_s
            
            state = next_state
    
    return w
```

### 5.2 Deep Q-Networks (DQN) 🔴

**Concept**: Use neural networks to approximate Q-function.

#### DQN Improvements:
1. **Experience Replay**: Store and sample experiences
2. **Target Network**: Separate network for target values
3. **Double DQN**: Reduce overestimation bias

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size, lr=0.001, gamma=0.99, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Neural networks
        self.q_network = DQN(state_size, action_size)
        self.target_network = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Experience replay
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        
        # Update target network
        self.update_target_network()
    
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Choose action using ε-greedy policy"""
        if np.random.random() <= self.epsilon:
            return random.choice(range(self.action_size))
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())
    
    def replay(self):
        """Train the model on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_dqn(env, agent, num_episodes=1000, target_update_freq=100):
    """Train DQN agent"""
    scores = []
    
    for episode in range(num_episodes):
        state = env.reset()
        score = 0
        
        while True:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            
            state = next_state
            score += reward
            
            if done:
                break
            
            agent.replay()
        
        scores.append(score)
        
        # Update target network periodically
        if episode % target_update_freq == 0:
            agent.update_target_network()
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Average Score: {np.mean(scores[-100:])}")
    
    return scores
```

---

## 6. Policy Gradient Methods

### 6.1 REINFORCE 🔴

**Concept**: Directly optimize policy using gradient ascent.

#### Policy Gradient Theorem:
```
∇J(θ) = E[∇ log π(a|s, θ) Q^π(s, a)]
```

#### REINFORCE Algorithm:
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=1)

class REINFORCEAgent:
    def __init__(self, state_size, action_size, lr=0.01, gamma=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        
        self.policy_network = PolicyNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
        
        self.log_probs = []
        self.rewards = []
    
    def act(self, state):
        """Select action based on policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.policy_network(state_tensor)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        
        self.log_probs.append(action_dist.log_prob(action))
        return action.item()
    
    def update(self):
        """Update policy using REINFORCE"""
        discounted_rewards = []
        G = 0
        
        # Calculate discounted returns
        for reward in reversed(self.rewards):
            G = reward + self.gamma * G
            discounted_rewards.insert(0, G)
        
        # Normalize rewards
        discounted_rewards = torch.FloatTensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / \
                           (discounted_rewards.std() + 1e-8)
        
        # Calculate policy loss
        policy_loss = []
        for log_prob, reward in zip(self.log_probs, discounted_rewards):
            policy_loss.append(-log_prob * reward)
        
        policy_loss = torch.cat(policy_loss).sum()
        
        # Update policy
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
        # Clear memory
        self.log_probs = []
        self.rewards = []

def train_reinforce(env, agent, num_episodes=1000):
    """Train REINFORCE agent"""
    scores = []
    
    for episode in range(num_episodes):
        state = env.reset()
        score = 0
        
        while True:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.rewards.append(reward)
            
            state = next_state
            score += reward
            
            if done:
                break
        
        agent.update()
        scores.append(score)
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Average Score: {np.mean(scores[-100:])}")
    
    return scores
```

### 6.2 Actor-Critic Methods 🔴

**Concept**: Combine value function (critic) with policy (actor).

#### Actor-Critic Architecture:
- **Actor**: Policy network π(a|s, θ)
- **Critic**: Value network V(s, w)

```python
class ActorCriticNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(ActorCriticNetwork, self).__init__()
        
        # Shared layers
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        # Actor head
        self.actor_head = nn.Linear(hidden_size, action_size)
        
        # Critic head
        self.critic_head = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        action_probs = F.softmax(self.actor_head(x), dim=1)
        state_value = self.critic_head(x)
        
        return action_probs, state_value

class ActorCriticAgent:
    def __init__(self, state_size, action_size, lr=0.01, gamma=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        
        self.network = ActorCriticNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
    def act(self, state):
        """Select action and get value estimate"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs, state_value = self.network(state_tensor)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        
        return action.item(), action_dist.log_prob(action), state_value
    
    def update(self, log_prob, value, reward, next_value, done):
        """Update actor and critic"""
        if done:
            target = reward
        else:
            target = reward + self.gamma * next_value
        
        advantage = target - value
        
        actor_loss = -log_prob * advantage.detach()
        critic_loss = F.mse_loss(value, target.detach())
        
        total_loss = actor_loss + critic_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()
```

---

## 7. Advanced Topics

### 7.1 Multi-Agent Reinforcement Learning 🔴

**Challenges**:
- Non-stationary environment
- Partial observability
- Coordination vs competition

#### Independent Q-Learning:
```python
class MultiAgentEnvironment:
    def __init__(self, num_agents, state_size, action_size):
        self.num_agents = num_agents
        self.agents = [DQNAgent(state_size, action_size) for _ in range(num_agents)]
    
    def step(self, actions):
        # Execute joint action
        next_states, rewards, done = self.env.step(actions)
        return next_states, rewards, done
    
    def train_episode(self):
        states = self.env.reset()
        
        while not self.env.done:
            actions = []
            for i, agent in enumerate(self.agents):
                action = agent.act(states[i])
                actions.append(action)
            
            next_states, rewards, done = self.step(actions)
            
            # Update each agent independently
            for i, agent in enumerate(self.agents):
                agent.remember(states[i], actions[i], rewards[i], next_states[i], done)
                agent.replay()
            
            states = next_states
```

### 7.2 Hierarchical Reinforcement Learning 🔴

**Concept**: Decompose complex tasks into hierarchy of subtasks.

#### Options Framework:
- **Option**: (π, β, I) where:
  - π: Policy
  - β: Termination condition
  - I: Initiation set

```python
class Option:
    def __init__(self, policy, termination_fn, initiation_set):
        self.policy = policy
        self.termination_fn = termination_fn
        self.initiation_set = initiation_set
    
    def can_initiate(self, state):
        return state in self.initiation_set
    
    def should_terminate(self, state):
        return self.termination_fn(state)
    
    def get_action(self, state):
        return self.policy(state)
```

### 7.3 Inverse Reinforcement Learning 🔴

**Goal**: Learn reward function from expert demonstrations.

#### Maximum Entropy IRL:
```python
def maximum_entropy_irl(demonstrations, feature_fn, learning_rate=0.01, 
                       num_iterations=100):
    """
    Maximum Entropy Inverse Reinforcement Learning
    """
    # Initialize reward weights
    theta = np.random.normal(0, 1, feature_fn.num_features)
    
    for iteration in range(num_iterations):
        # Compute policy for current reward
        rewards = np.dot(feature_fn.features, theta)
        policy = compute_policy(rewards)
        
        # Compute feature expectations
        mu_expert = compute_feature_expectations(demonstrations, feature_fn)
        mu_policy = compute_feature_expectations_policy(policy, feature_fn)
        
        # Gradient step
        gradient = mu_expert - mu_policy
        theta += learning_rate * gradient
    
    return theta
```

---

## 8. Practical Applications

### 8.1 Game Playing 🟡

#### OpenAI Gym Integration:
```python
import gym

def train_cartpole():
    env = gym.make('CartPole-v1')
    agent = DQNAgent(state_size=4, action_size=2)
    
    scores = []
    for episode in range(1000):
        state = env.reset()
        score = 0
        
        while True:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            
            state = next_state
            score += reward
            
            if done:
                break
            
            agent.replay()
        
        scores.append(score)
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Average Score: {np.mean(scores[-100:])}")
    
    return agent, scores
```

### 8.2 Robotics 🔴

#### Continuous Control:
```python
class DDPGAgent:
    """Deep Deterministic Policy Gradient for continuous control"""
    
    def __init__(self, state_size, action_size, action_high, action_low):
        self.state_size = state_size
        self.action_size = action_size
        self.action_high = action_high
        self.action_low = action_low
        
        # Actor and Critic networks
        self.actor = ActorNetwork(state_size, action_size)
        self.critic = CriticNetwork(state_size, action_size)
        self.target_actor = ActorNetwork(state_size, action_size)
        self.target_critic = CriticNetwork(state_size, action_size)
        
        # Noise for exploration
        self.noise = OUNoise(action_size)
        
    def act(self, state, add_noise=True):
        action = self.actor(state)
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, self.action_low, self.action_high)
```

### 8.3 Finance and Trading 🔴

#### Trading Environment:
```python
class TradingEnvironment:
    def __init__(self, data, initial_balance=10000):
        self.data = data
        self.initial_balance = initial_balance
        self.reset()
    
    def reset(self):
        self.balance = self.initial_balance
        self.shares = 0
        self.current_step = 0
        return self.get_state()
    
    def get_state(self):
        # Return price features, technical indicators, portfolio state
        return np.array([
            self.data[self.current_step],  # Current price
            self.balance,
            self.shares,
            self.balance + self.shares * self.data[self.current_step]  # Portfolio value
        ])
    
    def step(self, action):
        # action: 0=hold, 1=buy, 2=sell
        current_price = self.data[self.current_step]
        
        if action == 1 and self.balance >= current_price:  # Buy
            shares_to_buy = self.balance // current_price
            self.shares += shares_to_buy
            self.balance -= shares_to_buy * current_price
            
        elif action == 2 and self.shares > 0:  # Sell
            self.balance += self.shares * current_price
            self.shares = 0
        
        self.current_step += 1
        
        # Calculate reward (change in portfolio value)
        new_portfolio_value = self.balance + self.shares * self.data[self.current_step]
        reward = new_portfolio_value - self.initial_balance
        
        done = self.current_step >= len(self.data) - 1
        
        return self.get_state(), reward, done, {}
```

---

## 🎯 Key Takeaways

### Algorithm Selection Guide:

#### For Discrete Actions:
- **Simple environments**: Q-Learning, SARSA
- **Large state spaces**: DQN, Double DQN
- **Need interpretability**: Policy Iteration
- **Continuous learning**: TD methods

#### For Continuous Actions:
- **Continuous control**: DDPG, TD3, SAC
- **High-dimensional**: Policy gradient methods
- **Sample efficiency important**: Model-based methods

#### For Multi-Agent:
- **Cooperative**: Centralized training, decentralized execution
- **Competitive**: Self-play, population-based training
- **Mixed**: Multi-agent actor-critic

### Best Practices:
1. **Start simple**: Tabular methods for small problems
2. **Exploration vs Exploitation**: Balance carefully
3. **Hyperparameter tuning**: Critical for performance
4. **Environment design**: Reward engineering is crucial
5. **Evaluation**: Use multiple random seeds, statistical testing

### Common Challenges:
- **Sample efficiency**: RL often needs many samples
- **Stability**: Function approximation can be unstable
- **Exploration**: Balancing exploration and exploitation
- **Reward design**: Sparse or misleading rewards
- **Generalization**: Overfitting to specific environments

---

## 📚 Next Steps

Continue your ML journey with:
- **[Deep Learning](07_Deep_Learning.md)** - Neural networks and deep architectures
- **[Natural Language Processing](08_Natural_Language_Processing.md)** - Text and language understanding

---

## 🛠️ Practical Exercises

### Exercise 1: Grid World Navigation
Implement and compare Q-Learning vs SARSA:
1. Create simple grid world environment
2. Implement both algorithms
3. Compare convergence and final policies
4. Analyze exploration strategies

### Exercise 2: Deep Q-Network
Build DQN for Atari game:
1. Implement experience replay
2. Add target network
3. Compare with Double DQN
4. Analyze training stability

### Exercise 3: Policy Gradient
Implement REINFORCE for continuous control:
1. Design continuous action space
2. Implement baseline for variance reduction
3. Compare with actor-critic
4. Analyze sample efficiency

---

*Next: [Deep Learning →](07_Deep_Learning.md)*
