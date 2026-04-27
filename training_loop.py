import numpy as np
import pickle
from agent_architecture import PPOAgent, RandomAgent
from custom_types import Observation
from env_wrapper import BoardsWrapper


def train_agents(env: BoardsWrapper, sender_agent: PPOAgent | RandomAgent, receiver_agent: PPOAgent | RandomAgent, n_episodes: int, learn_interval: int = 32):
    if (env.max_moves % 2) != 0:
        raise ValueError("The environment should be set to an even number of moves.")
    
    performances_dist = []
    clue_alignment_dist = []  # optimal_distance(clues, landmarks) at episode end: 0 means sender communicated perfectly

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
        receiver_acted_this_episode = False

        while not done:
            sender_state = env.sender_observe()
            sender_action, sender_action_probs, sender_value = sender_agent.choose_action(sender_state)
            sender_reward, done = env.sender_act(sender_action)

            sender_instant += sender_reward
            useless_action_sender += env.get_useless_action_val()
            if sender_action == 0:
                empty_action_sender += 1

            if done:
                final_reward = env.get_final_reward()
                final_performance = env.get_final_performance()
                episode_length = env.num_moves
                if isinstance(sender_agent, PPOAgent):
                    sender_agent.remember(sender_state, sender_action, sender_action_probs, sender_value, final_reward,
                                          True)
                # Episode ended on sender's half-turn: retroactively mark the receiver's
                # last stored experience as terminal so GAE doesn't bootstrap across
                # episode boundaries.
                if isinstance(receiver_agent, PPOAgent) and receiver_acted_this_episode:
                    receiver_agent.memory.rewards[-1] = final_reward
                    receiver_agent.memory.dones[-1] = True
                break
            
            receiver_state = env.receiver_observe()
            receiver_action, receiver_action_probs, receiver_value = receiver_agent.choose_action(receiver_state)
            receiver_reward, done = env.receiver_act(receiver_action)
            receiver_acted_this_episode = True

            receiver_instant += receiver_reward
            useless_action_receiver += env.get_useless_action_val()
            if receiver_action == 0:
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
        clue_alignment_dist.append(env.get_clue_landmark_distance())
        final_rewards_dist.append(final_reward)
        receiver_instant_rewards_dist.append(receiver_instant)
        sender_instant_rewards_dist.append(sender_instant)
        episode_lengths_dist.append(episode_length)
        useless_actions_sender_dist.append(useless_action_sender)
        useless_actions_receiver_dist.append(useless_action_receiver)
        empty_actions_sender_dist.append(empty_action_sender)
        empty_actions_receiver_dist.append(empty_action_receiver)

        should_learn = (episode + 1) % learn_interval == 0 or episode == n_episodes - 1
        if should_learn:
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
             'clue_alignment_dist': clue_alignment_dist,
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

def train_agents_vec(
    envs: list[BoardsWrapper],
    sender_agent: PPOAgent | RandomAgent,
    receiver_agent: PPOAgent | RandomAgent,
    n_episodes: int,
    learn_interval: int = 32,
) -> dict:
    """Vectorized training: N envs run in lockstep, inference is batched."""
    if not envs:
        raise ValueError("envs must be non-empty.")
    if (envs[0].max_moves % 2) != 0:
        raise ValueError("max_moves must be even.")

    N = len(envs)
    use_ppo_sender = isinstance(sender_agent, PPOAgent)
    use_ppo_receiver = isinstance(receiver_agent, PPOAgent)

    performances_dist, clue_alignment_dist, final_rewards_dist = [], [], []
    receiver_instant_rewards_dist, sender_instant_rewards_dist, episode_lengths_dist = [], [], []
    useless_actions_sender_dist, useless_actions_receiver_dist = [], []
    empty_actions_sender_dist, empty_actions_receiver_dist = [], []
    receiver_entropy_dist, receiver_actor_loss_dist, receiver_critic_loss_dist, receiver_total_loss_dist = [], [], [], []
    sender_entropy_dist, sender_actor_loss_dist, sender_critic_loss_dist, sender_total_loss_dist = [], [], [], []

    completed = 0
    while completed < n_episodes:
        batch_n = min(N, n_episodes - completed)
        batch_envs = envs[:batch_n]
        for env in batch_envs:
            env.reset()

        done_flags = [False] * batch_n
        receiver_acted = [False] * batch_n
        sender_instant = [0.0] * batch_n
        receiver_instant = [0.0] * batch_n
        useless_s = [0.0] * batch_n
        useless_r = [0.0] * batch_n
        empty_s = [0.0] * batch_n
        empty_r = [0.0] * batch_n
        # Saved sender state for deferred storage (stored after receiver step)
        saved_sender: list[tuple | None] = [None] * batch_n

        while not all(done_flags):
            # ── SENDER STEP ──────────────────────────────────────────────
            active = [i for i in range(batch_n) if not done_flags[i]]
            sender_raw = [batch_envs[i].sender_observe_raw() for i in active]

            if use_ppo_sender:
                batch_obs = BoardsWrapper.batch_observe_to_tensor(sender_raw, batch_envs[0].device)
                s_actions, s_logps, s_vals = sender_agent.batch_choose_action(batch_obs)
                # Slice individual observations for per-env storage (cheap view, no copy)
                sender_obs_list = [
                    Observation(
                        batch_obs.current_board[j : j + 1],
                        batch_obs.previous_boards[j : j + 1],
                        batch_obs.progress[j : j + 1],
                    )
                    for j in range(len(active))
                ]
            else:
                s_actions, s_logps, s_vals = [], [], []
                sender_obs_list = []
                for raw in sender_raw:
                    obs = batch_envs[0]._to_tensor(*raw)
                    a, lp, v = sender_agent.choose_action(obs)
                    s_actions.append(a); s_logps.append(lp); s_vals.append(v)
                    sender_obs_list.append(obs)

            for j, i in enumerate(active):
                a, lp, v = s_actions[j], s_logps[j], s_vals[j]
                reward, done = batch_envs[i].sender_act(a)
                sender_instant[i] += reward
                useless_s[i] += batch_envs[i].get_useless_action_val()
                if a == 0:
                    empty_s[i] += 1
                saved_sender[i] = (sender_obs_list[j], a, lp, v, reward)

                if done:
                    final_r = batch_envs[i].get_final_reward()
                    if use_ppo_sender:
                        sender_agent.remember(sender_obs_list[j], a, lp, v, final_r, True)
                    if use_ppo_receiver and receiver_acted[i]:
                        receiver_agent.memory.rewards[-1] = final_r
                        receiver_agent.memory.dones[-1] = True
                    performances_dist.append(batch_envs[i].get_final_performance())
                    clue_alignment_dist.append(batch_envs[i].get_clue_landmark_distance())
                    final_rewards_dist.append(final_r)
                    sender_instant_rewards_dist.append(sender_instant[i])
                    receiver_instant_rewards_dist.append(receiver_instant[i])
                    episode_lengths_dist.append(batch_envs[i].num_moves)
                    useless_actions_sender_dist.append(useless_s[i])
                    useless_actions_receiver_dist.append(useless_r[i])
                    empty_actions_sender_dist.append(empty_s[i])
                    empty_actions_receiver_dist.append(empty_r[i])
                    done_flags[i] = True

            # ── RECEIVER STEP ─────────────────────────────────────────────
            active = [i for i in range(batch_n) if not done_flags[i]]
            if not active:
                break
            receiver_raw = [batch_envs[i].receiver_observe_raw() for i in active]

            if use_ppo_receiver:
                batch_obs = BoardsWrapper.batch_observe_to_tensor(receiver_raw, batch_envs[0].device)
                r_actions, r_logps, r_vals = receiver_agent.batch_choose_action(batch_obs)
                receiver_obs_list = [
                    Observation(
                        batch_obs.current_board[j : j + 1],
                        batch_obs.previous_boards[j : j + 1],
                        batch_obs.progress[j : j + 1],
                    )
                    for j in range(len(active))
                ]
            else:
                r_actions, r_logps, r_vals = [], [], []
                receiver_obs_list = []
                for raw in receiver_raw:
                    obs = batch_envs[0]._to_tensor(*raw)
                    a, lp, v = receiver_agent.choose_action(obs)
                    r_actions.append(a); r_logps.append(lp); r_vals.append(v)
                    receiver_obs_list.append(obs)

            for j, i in enumerate(active):
                ra, rlp, rv = r_actions[j], r_logps[j], r_vals[j]
                reward, done = batch_envs[i].receiver_act(ra)
                receiver_acted[i] = True
                receiver_instant[i] += reward
                useless_r[i] += batch_envs[i].get_useless_action_val()
                if ra == 0:
                    empty_r[i] += 1

                s_obs, sa, slp, sv, s_reward = saved_sender[i]  # type: ignore[misc]

                if done:
                    final_r = batch_envs[i].get_final_reward()
                    if use_ppo_sender:
                        sender_agent.remember(s_obs, sa, slp, sv, final_r, True)
                    if use_ppo_receiver:
                        receiver_agent.remember(receiver_obs_list[j], ra, rlp, rv, final_r, True)
                    performances_dist.append(batch_envs[i].get_final_performance())
                    clue_alignment_dist.append(batch_envs[i].get_clue_landmark_distance())
                    final_rewards_dist.append(final_r)
                    sender_instant_rewards_dist.append(sender_instant[i])
                    receiver_instant_rewards_dist.append(receiver_instant[i])
                    episode_lengths_dist.append(batch_envs[i].num_moves)
                    useless_actions_sender_dist.append(useless_s[i])
                    useless_actions_receiver_dist.append(useless_r[i])
                    empty_actions_sender_dist.append(empty_s[i])
                    empty_actions_receiver_dist.append(empty_r[i])
                    done_flags[i] = True
                else:
                    if use_ppo_sender:
                        sender_agent.remember(s_obs, sa, slp, sv, s_reward, False)
                    if use_ppo_receiver:
                        receiver_agent.remember(receiver_obs_list[j], ra, rlp, rv, reward, False)

        completed += batch_n

        should_learn = (completed % learn_interval == 0 or completed >= n_episodes) and completed > 0
        if should_learn:
            if isinstance(sender_agent, PPOAgent):
                e, al, cl, tl = sender_agent.learn()
                sender_entropy_dist.append(e); sender_actor_loss_dist.append(al)
                sender_critic_loss_dist.append(cl); sender_total_loss_dist.append(tl)
            if isinstance(receiver_agent, PPOAgent):
                e, al, cl, tl = receiver_agent.learn()
                receiver_entropy_dist.append(e); receiver_actor_loss_dist.append(al)
                receiver_critic_loss_dist.append(cl); receiver_total_loss_dist.append(tl)

        if completed % max(n_episodes // 20, N) < N:
            last_perf = performances_dist[-1] if performances_dist else float("nan")
            print(f"Episode {completed}, Performance: {last_perf:.4f}")

    return {
        "performances_dist": performances_dist,
        "clue_alignment_dist": clue_alignment_dist,
        "final_rewards_dist": final_rewards_dist,
        "receiver_instant_rewards_dist": receiver_instant_rewards_dist,
        "sender_instant_rewards_dist": sender_instant_rewards_dist,
        "episode_lengths_dist": episode_lengths_dist,
        "useless_actions_sender_dist": useless_actions_sender_dist,
        "useless_actions_receiver_dist": useless_actions_receiver_dist,
        "empty_actions_sender_dist": empty_actions_sender_dist,
        "empty_actions_receiver_dist": empty_actions_receiver_dist,
        "receiver_entropy_dist": receiver_entropy_dist,
        "receiver_actor_loss_dist": receiver_actor_loss_dist,
        "receiver_critic_loss_dist": receiver_critic_loss_dist,
        "receiver_total_loss_dist": receiver_total_loss_dist,
        "sender_entropy_dist": sender_entropy_dist,
        "sender_actor_loss_dist": sender_actor_loss_dist,
        "sender_critic_loss_dist": sender_critic_loss_dist,
        "sender_total_loss_dist": sender_total_loss_dist,
    }


def save_stats(stats, file_path: str):
    with open(file_path, 'wb') as file:
        pickle.dump(stats, file)

def load_stats(file_path: str):
    with open(file_path, 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data