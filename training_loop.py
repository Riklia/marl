import numpy as np
import pickle
from agent_architecture import PPOAgent, RandomAgent
from env_wrapper import BoardsWrapper


def train_agents(env: BoardsWrapper, sender_agent: PPOAgent | RandomAgent, receiver_agent: PPOAgent | RandomAgent, n_episodes: int):
    if (env.max_moves % 2) != 0:
        raise ValueError("The environment should be set to an even number of moves.")
    
    performances_dist = []

    # Additional stats - there might be bugs for frozen and RandomAgent, refactor needed
    # Training stats
    final_rewards_dist = [] # final_performance * self.end_multiplier
    receiver_instant_rewards_dist = []
    sender_instant_rewards_dist = []
    episode_lengths_dist = []
    receiver_entropy_dist = []
    receiver_actor_loss_dist = []
    receiver_critic_loss_dist = []
    receiver_total_loss_dist = []
    sender_entropy_dist = []
    sender_actor_loss_dist = []
    sender_critic_loss_dist = []
    sender_total_loss_dist = []

    # Agents actions stats
    useless_actions_sender_dist = []
    useless_actions_receiver_dist = []
    empty_actions_sender_dist = []
    empty_actions_receiver_dist = []

    # Communication and game stats
    # distance_delta_dist = list() # guess - landmark greedy distance, expensive
    # clues_changes_dist = list() # in current situation there is only 1 move per turn, so it wouldnt give us much


    for episode in range(n_episodes):
        env.reset()
        done = False

        # Accumulators for stats
        final_reward = 0.0
        final_performance = 0.0
        useless_action_sender = 0.0
        useless_action_receiver = 0.0
        empty_action_sender = 0.0
        empty_action_receiver = 0.0
        sender_instant = 0.0
        receiver_instant = 0.0
        episode_length = 0.0

        while not done:
            sender_state = env.sender_observe()
            sender_action, sender_action_probs, sender_value = sender_agent.choose_action(sender_state)
            sender_reward, _ = env.sender_act(sender_action)

            sender_instant += sender_reward
            useless_action_sender += env.get_useless_action_val()
            if (sender_action == 0):
                empty_action_sender += 1
            
            receiver_state = env.receiver_observe()
            receiver_action, receiver_action_probs, receiver_value = receiver_agent.choose_action(receiver_state)
            receiver_reward, done = env.receiver_act(receiver_action)

            receiver_instant += receiver_reward
            useless_action_receiver += env.get_useless_action_val()
            if (receiver_action == 0):
                empty_action_receiver += 1
            
            if done:
                final_reward = env.get_final_reward()
                final_performance = env.get_final_performance()
                episode_length = env.num_moves
                if isinstance(sender_agent, PPOAgent):
                    sender_agent.remember(sender_state, sender_action, sender_action_probs, sender_value, final_reward, True)
                if isinstance(receiver_agent, PPOAgent):
                    receiver_agent.remember(receiver_state, receiver_action, receiver_action_probs, receiver_value, final_reward, True)
            else:
                if isinstance(sender_agent, PPOAgent):
                    sender_agent.remember(sender_state, sender_action, sender_action_probs, sender_value, sender_reward, False)
                if isinstance(receiver_agent, PPOAgent):
                    receiver_agent.remember(receiver_state, receiver_action, receiver_action_probs, receiver_value, receiver_reward, False)
        
        performances_dist.append(final_performance)
        final_rewards_dist.append(final_reward)
        receiver_instant_rewards_dist.append(receiver_instant)
        sender_instant_rewards_dist.append(sender_instant)
        episode_lengths_dist.append(episode_length)
        useless_actions_sender_dist.append(useless_action_sender)
        useless_actions_receiver_dist.append(useless_action_receiver)
        empty_actions_sender_dist.append(empty_action_sender)
        empty_actions_receiver_dist.append(empty_action_receiver)

        if isinstance(sender_agent, PPOAgent):
            entropy_dist, actor_loss_dist, critic_loss_dist, total_loss_dist = sender_agent.learn()
            sender_entropy_dist.append(entropy_dist)
            sender_actor_loss_dist.append(actor_loss_dist)
            sender_critic_loss_dist.append(critic_loss_dist)
            sender_total_loss_dist.append(total_loss_dist)
        if isinstance(receiver_agent, PPOAgent):
            entropy_dist, actor_loss_dist, critic_loss_dist, total_loss_dist = receiver_agent.learn()
            receiver_entropy_dist.append(entropy_dist)
            receiver_actor_loss_dist.append(actor_loss_dist)
            receiver_critic_loss_dist.append(critic_loss_dist)
            receiver_total_loss_dist.append(total_loss_dist)
        if episode % max(n_episodes // 20, 1) == 0:
            print(f"Episode {episode}, Performance: {final_performance:.4f}")
    
    stats = {'performances_dist': performances_dist,
             'final_rewards_dist': final_rewards_dist,
             'receiver_instant_rewards_dist': receiver_instant_rewards_dist,
             'sender_instant_rewards_dist': sender_instant_rewards_dist,
             'episode_lengths_dist': episode_lengths_dist,
             'useless_actions_sender_dist': useless_actions_sender_dist,
             'useless_actions_receiver_dist': useless_actions_receiver_dist,
             'empty_actions_sender_dist': empty_actions_sender_dist,
             'empty_actions_receiver_dist': empty_actions_receiver_dist,
             'receiver_entropy_dist': receiver_entropy_dist,
             'receiver_actor_loss_dist': receiver_actor_loss_dist,
             'receiver_critic_loss_dist': receiver_critic_loss_dist,
             'receiver_total_loss_dist': receiver_total_loss_dist,
             'sender_entropy_dist': sender_entropy_dist,
             'sender_actor_loss_dist': sender_actor_loss_dist,
             'sender_critic_loss_dist': sender_critic_loss_dist,
             'sender_total_loss_dist': sender_total_loss_dist}
    return stats

def save_stats(stats: dict(), file_path: str):
    with open(file_path, 'wb') as file:
        pickle.dump(stats, file)

def load_stats(file_path: str):
    with open(file_path, 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data