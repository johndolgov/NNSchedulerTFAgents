import tensorflow as tf
import numpy as np

from tf_agents.agents.ppo import ppo_agent, ppo_policy
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_network, value_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from ep_utils.setups import wf_setup
from draw_figures import write_schedule
from copy import deepcopy

from env.context import Context
from argparse import ArgumentParser
from ep_utils.setups import parameter_setup, DEFAULT_CONFIG

tf.compat.v1.enable_v2_behavior()

parser = ArgumentParser()


parser.add_argument('--state-size', type=int, default=20)
parser.add_argument('--agent-tasks', type=int, default=5)

parser.add_argument('--actor-type', type=str, default='fc')
parser.add_argument('--first-layer', type=int, default=1024)
parser.add_argument('--second-layer', type=int, default=512)
parser.add_argument('--third-layer', type=int, default=256)
parser.add_argument('--seq-size', type=int, default=5)
parser.add_argument('--load', type=bool, default=False)
parser.add_argument('--load-path', type=str, default=None)

parser.add_argument('--n_nodes', type=int, default=4)
parser.add_argument('--nodes', type=np.ndarray, default=None)
parser.add_argument('--host', type=str, default='localhost')
parser.add_argument('--port', type=int, default=9900)
parser.add_argument('--model-name', type=str, default='')

parser.add_argument('--task-par', type=int, default=None)
parser.add_argument('--task-par-min', type=int, default=None)
parser.add_argument('--batch-size', type=int, default=None)

parser.add_argument('--wfs-name', type=str, default=None)
parser.add_argument('--is-test', type=bool, default=False)
parser.add_argument('--num-episodes', type=int, default=10000)
parser.add_argument('--logger', type=bool, default=True)
parser.add_argument('--run-name', type=str, default='NoName')
parser.add_argument('--save', type=bool, default=False)
parser.add_argument('--plot-csvs', type=bool, default=False)
parser.add_argument('--result-folder', type=str, default='')


def test_schedule(tf_env, policy):
    time_step = tf_env.reset()
    while not time_step.is_last():
        mask = tf_env.envs[0].get_mask()
        action_step = policy.distribution(time_step)
        action = tf.convert_to_tensor(
            np.argmax(np.asarray(mask) * tf.nn.softmax(action_step.info['dist_params']['logits'], axis=-1)),
            dtype=tf.int32)
        time_step = tf_env.step(action)
        if not time_step.is_last():
            final_env = deepcopy(tf_env.envs[0])

    write_schedule('Test', 0, final_env)


def compute_avg_return(tf_env, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):

        time_step = tf_env.reset()
        episode_return = 0.0

        while not time_step.is_last():
            mask = tf_env.envs[0].get_mask()
            action_step = policy.distribution(time_step)
            action = tf.convert_to_tensor(
                np.argmax(np.asarray(mask) * tf.nn.softmax(action_step.info['dist_params']['logits'], axis=-1)),
                dtype=tf.int32)
            time_step = tf_env.step(action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


def collect_episode(tf_env, policy, num_episodes, replay_buffer):
    episode_counter = 0
    tf_env.reset()
    real_env.reset()

    while episode_counter < num_episodes:
        time_step = tf_env.current_time_step()

        mask = tf_env.envs[0].get_mask()
        action_step = policy.distribution(time_step)
        action_step_act = policy.action(time_step)

        action = tf.convert_to_tensor(np.argmax(np.asarray(mask) * tf.nn.softmax(action_step.info['dist_params']['logits'], axis=-1)),
                                      dtype=tf.int32)
        next_time_step = tf_env.step(action)

        traj = trajectory.from_transition(time_step, action_step_act, next_time_step)
        # Add trajectory to the replay buffer
        replay_buffer.add_batch(traj)

        if traj.is_boundary():
            episode_counter += 1


if __name__ == '__main__':
    args = parser.parse_args()

    config = parameter_setup(args, DEFAULT_CONFIG)
    test_wfs, test_times, test_scores, test_size = wf_setup(config['wfs_name'])
    ttree, tdata, trun_times = test_wfs[0]

    real_env = Context(config['agent_task'], config['nodes'], trun_times, ttree, tdata)
    environment = tf_py_environment.TFPyEnvironment(real_env)

    eval_real_env = Context(config['agent_task'], config['nodes'], trun_times, ttree, tdata)
    eval_environment = tf_py_environment.TFPyEnvironment(eval_real_env)

    actor_net = actor_distribution_network.ActorDistributionNetwork(
        environment.observation_spec(),
        environment.action_spec(),
        fc_layer_params=(200, 100))

    value_net = value_network.ValueNetwork(
        environment.observation_spec(),
        fc_layer_params=(200, 100))

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)
    train_step_counter = tf.compat.v2.Variable(0)

    tf_agent = ppo_agent.PPOAgent(
        time_step_spec=environment.time_step_spec(),
        action_spec=environment.action_spec(),
        actor_net=actor_net,
        value_net=value_net,
        optimizer=optimizer,
        train_step_counter=train_step_counter,
        discount_factor=0.995,
        gradient_clipping=0.5,
        entropy_regularization=1e-2,
        importance_ratio_clipping=0.2,
        use_gae=True,
        use_td_lambda_return=True)

    tf_agent.initialize()

    eval_policy = tf_agent.policy
    collect_policy = tf_agent.collect_policy

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=tf_agent.collect_data_spec,
        batch_size=1,
        max_length=20000)

    tf_agent.train = common.function(tf_agent.train)

    # Reset the train step
    tf_agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    # avg_return = compute_avg_return(environment, tf_agent.policy, 100)
    returns = []

    for _ in range(args.num_episodes):

        # Collect a few episodes using collect_policy and save to the replay buffer.
        collect_episode(environment, tf_agent.collect_policy, True, replay_buffer)

        # Use data from the buffer and update the agent's network.
        experience = replay_buffer.gather_all()
        train_loss = tf_agent.train(experience)
        replay_buffer.clear()

        step = tf_agent.train_step_counter.numpy()

        if step % 10 == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss.loss))

        if step % 10 == 0:
            avg_return = compute_avg_return(eval_environment, collect_policy, 100)
            print('step = {0}: Average Return = {1}'.format(step, avg_return))
            test_schedule(eval_environment, collect_policy)
            returns.append(avg_return)
