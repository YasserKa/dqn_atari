import random

import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)

    def push(self, obs, action, next_obs, reward):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = (obs, action, next_obs, reward)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Samples batch_size transitions from the replay memory and returns a tuple
            (obs, action, next_obs, reward)
        """
        sample = random.sample(self.memory, batch_size)
        return tuple(zip(*sample))


class DQN(nn.Module):
    def __init__(self, env_config):
        super(DQN, self).__init__()

        # Save hyperparameters needed in the DQN class.
        self.batch_size = env_config["batch_size"]
        self.gamma = env_config["gamma"]
        self.eps_start = env_config["eps_start"]
        self.eps_end = env_config["eps_end"]
        self.anneal_length = env_config["anneal_length"]
        self.n_actions = env_config["n_actions"]

        self.fc1 = nn.Linear(4, 256)
        self.fc2 = nn.Linear(256, self.n_actions)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        self.replay_memory_start_size = env_config['memory_size']
        self.max_frames = 1

        # Slopes and intercepts for exploration decrease
        self.slope = -(self.eps_start - self.eps_end)/self.anneal_length
        self.intercept = self.eps_start - self.slope
        self.slope_2 = -(self.eps_end)/(self.max_frames -
                                        self.anneal_length
                                        )
        self.intercept_2 = self.slope_2*self.max_frames

    def forward(self, x):
        """Runs the forward pass of the NN depending on architecture."""
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def act(self, observation, step_number, exploit=False):
        """Selects an action with an epsilon-greedy exploration strategy."""
        # Implement action selection using the Deep Q-network. This function
        # takes an observation tensor and should return a tensor of actions. For
        # example, if the state dimension is 4 and the batch size is 32, the
        # input would be a [32, 4] tensor and the output a [32, 1] tensor.
        if exploit:
            eps = 0
        elif step_number < self.replay_memory_start_size:
            eps = self.eps_start
        elif step_number >= self.replay_memory_start_size and step_number < self.replay_memory_start_size + self.anneal_length:
            eps = self.slope*step_number + self.intercept
        elif step_number >= self.replay_memory_start_size + self.anneal_length:
            eps = self.slope_2*step_number + self.intercept_2

        if random.random() < eps:
            return torch.tensor([[random.randrange(self.n_actions)]], device=device, dtype=torch.long)
        else:
            return self(observation).max(1)[1].view(1, 1)


def optimize(dqn, target_dqn, memory, optimizer):
    """This function samples a batch from the replay buffer and optimizes the Q-network."""
    # If we don't have enough transitions stored yet, we don't train.
    if len(memory) < dqn.batch_size:
        return

    # Sample a batch from the replay memory and concatenate so that there are
    # four tensors in total: observations, actions, next observations and
    # rewards. Remember to move them to GPU if it is available, e.g., by using
    # Tensor.to(device). Note that special care is needed for terminal
    # transitions!
    batch = memory.sample(dqn.batch_size)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch[2])), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch[2] if s is not None])

    state_batch = torch.cat(batch[0])
    action_batch = torch.cat(batch[1])
    reward_batch = torch.cat(batch[3])

    # Compute the current estimates of the Q-values for each state-action pair
    # (s,a). Here, torch.gather() is useful for selecting the Q-values
    # corresponding to the chosen actions.
    q_values = dqn(state_batch).gather(1, action_batch)

    # Compute the Q-value targets. Only do this for non-terminal transitions!
    next_state_values = torch.zeros(dqn.batch_size, device=device)
    next_state_values[non_final_mask] = target_dqn(
        non_final_next_states).max(1)[0].squeeze().detach()
    # Compute the expected Q values
    q_value_targets = (next_state_values * dqn.gamma) + reward_batch

    # Compute loss.
    loss = F.mse_loss(q_values.squeeze(), q_value_targets)

    # Perform gradient descent.
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()

    return loss.item()
