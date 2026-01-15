from agent_architecture import PPOAgent, RandomAgent
from env_wrapper import BoardsWrapper

def train_agents(env: BoardsWrapper, sender_agent: PPOAgent | RandomAgent, receiver_agent: PPOAgent | RandomAgent, n_episodes: int):
    if (env.max_moves % 2) != 0:
        raise ValueError("The environment should be set to an even number of moves.")
    
    performances = list()

    for episode in range(n_episodes):
        env.reset()
        done = False
        final_reward = 0.0
        final_performance = 0.0
        while not done:
            sender_state = env.sender_observe()
            sender_action, sender_action_probs, sender_value = sender_agent.choose_action(sender_state)
            sender_reward, _ = env.sender_act(sender_action)
            
            receiver_state = env.receiver_observe()
            receiver_action, receiver_action_probs, receiver_value = receiver_agent.choose_action(receiver_state)
            receiver_reward, done = env.receiver_act(receiver_action)
            
            if done:
                final_reward = env.get_final_reward()
                final_performance = env.get_final_performance()
                if isinstance(sender_agent, PPOAgent):
                    sender_agent.remember(sender_state, sender_action, sender_action_probs, sender_value, final_reward, True)
                if isinstance(receiver_agent, PPOAgent):
                    receiver_agent.remember(receiver_state, receiver_action, receiver_action_probs, receiver_value, final_reward, True)
            else:
                if isinstance(sender_agent, PPOAgent):
                    sender_agent.remember(sender_state, sender_action, sender_action_probs, sender_value, sender_reward, False)
                if isinstance(receiver_agent, PPOAgent):
                    receiver_agent.remember(receiver_state, receiver_action, receiver_action_probs, receiver_value, receiver_reward, False)
        
        performances.append(final_performance)
        
        if isinstance(sender_agent, PPOAgent):
            sender_agent.learn()
        if isinstance(receiver_agent, PPOAgent):
            receiver_agent.learn()
        
        if episode % max(n_episodes // 20, 1) == 0:
            print(f"Episode {episode}, Performance: {final_performance:.4f}")
    
    return performances
