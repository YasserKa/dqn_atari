import argparse

# train
# python train.py --env CartPole-v0
# evaluation
# python evaluate.py --path models/CartPole-v0_best.pt --env CartPole-v0 --n_eval_episodes 5

import gym
import torch
import torch.nn as nn

import config
from utils import preprocess
from evaluate import evaluate_policy
from dqn import DQN, ReplayMemory, optimize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--env', choices=['CartPole-v0'])
parser.add_argument('--evaluate_freq', type=int, default=25,
                    help='How often to run evaluation.', nargs='?')
parser.add_argument('--evaluation_episodes', type=int, default=5,
                    help='Number of evaluation episodes.', nargs='?')

# Hyperparameter configurations for different environments. See config.py.
ENV_CONFIGS = {
    'CartPole-v0': config.CartPole
}

if __name__ == '__main__':
    args = parser.parse_args()

    # Initialize environment and config.
    env = gym.make(args.env)
    env_config = ENV_CONFIGS[args.env]

    # Initialize deep Q-networks.
    dqn = DQN(env_config=env_config).to(device)
    # Create and initialize target Q-network.
    target_dqn = DQN(env_config=env_config).to(device)
    target_dqn.load_state_dict(dqn.state_dict())
    target_dqn.eval()

    # Create replay memory.
    memory = ReplayMemory(env_config['memory_size'])

    # Initialize optimizer used for training the DQN. We use Adam rather than RMSProp.
    optimizer = torch.optim.Adam(dqn.parameters(), lr=env_config['lr'])

    # Keep track of best evaluation mean return achieved so far.
    best_mean_return = -float("Inf")

    # Used for eps annealing
    step_number = 0

    # Used for the plot
    all_means = []

    for episode in range(env_config['n_episodes']):
        done = False

        obs = preprocess(env.reset(), env=args.env).unsqueeze(0)

        while not done:
            # Get action from DQN.
            action = dqn.act(obs, step_number)

            # Act in the true environment.
            next_obs, reward, done, info = env.step(action.item())

            reward = torch.tensor([reward], device=device)
            next_obs = torch.tensor(
                next_obs, device=device, dtype=torch.float)
            step_number += 1

            # Preprocess incoming observation.
            if not done:
                next_obs = preprocess(next_obs, env=args.env).unsqueeze(0)
            else:
                next_obs = None

            # Add the transition to the replay memory
            memory.push(obs, action, next_obs, reward)

            obs = next_obs

            # Run DQN.optimize() every env_config["train_frequency"] steps.
            if step_number % env_config['train_frequency'] == 0:
                optimize(dqn, target_dqn, memory, optimizer)

            # Update the target network every env_config["target_update_frequency"] steps.
            if step_number % env_config['target_update_frequency'] == 0:
                target_dqn.load_state_dict(dqn.state_dict())

        # Evaluate the current agent.
        if episode % args.evaluate_freq == 0:
            mean_return = evaluate_policy(
                dqn, env, env_config, args, n_episodes=args.evaluation_episodes)

            all_means.append(mean_return)

            print(
                f'Episode {episode}/{env_config["n_episodes"]}: {mean_return}')

            # Save current agent if it has the best performance so far.
            if mean_return >= best_mean_return:
                best_mean_return = mean_return

                print('Best performance so far! Saving model.')
                torch.save(dqn, f'models/{args.env}_best.pt')

    print(all_means)

    # Close environment after training is completed.
    env.close()
