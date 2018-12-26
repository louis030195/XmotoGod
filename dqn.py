#!/usr/bin/python
# -*- coding: latin-1 -*-

from __future__ import division, print_function, unicode_literals

# Handle arguments (before slow imports so --help can be fast)
import argparse
parser = argparse.ArgumentParser(
    description="Train a DQN net fro Xmoto")

# Important hparams
#parser.add_argument("-g", "--game", type=str, default="Pong")
parser.add_argument("-n", "--number-steps", type=int, default=1000000, help="total number of training steps")
parser.add_argument("-e", "--explore-steps", type=int, default=100, help="total number of explorartion steps")
parser.add_argument("-c", "--copy-steps", type=int, default=4096, help="number of training steps between copies of online DQN to target DQN")
parser.add_argument("-l", "--learn-freq", type=int, default=4, help="number of game steps between each training step")

# Irrelevant hparams
parser.add_argument("-nl", "--nextlevel", type=int, default=200, help="number of steps to go next level")
parser.add_argument("-p", "--pretrain", action="store_true", default=False, help="pretrain with a action file")
parser.add_argument("-s", "--save-steps", type=int, default=100, help="number of training steps between saving checkpoints")
parser.add_argument("-t", "--test", action="store_true", default=False, help="test (no learning and minimal epsilon)")
parser.add_argument("-v", "--verbosity", action="count", default=1, help="increase output verbosity")
parser.add_argument("-j", "--jobid", default="models", help="SLURM job ID")

args = parser.parse_args()

from collections import deque
import gym
import gym_xmoto
import numpy as np
import os
import tensorflow as tf
import sys
import time

env = gym.make("Xmoto-v0")
env.render()
start_time = time.time()
x = 1 # displays the frame rate every 1 second
counter = 0

def q_network(net, name, reuse=False):
    with tf.variable_scope(name, reuse=reuse) as scope:
        initializer = tf.contrib.layers.variance_scaling_initializer()
        for n_maps, kernel_size, strides, padding, activation in zip(
                [32, 64, 64], [(8,8), (4,4), (3,3)], [4, 2, 1],
                ["SAME"] * 3 , [tf.nn.relu] * 3):
            net = tf.layers.conv2d(net, filters=n_maps, kernel_size=kernel_size, strides=strides,
                padding=padding, activation=activation, kernel_initializer=initializer)
        net = tf.layers.dense(tf.contrib.layers.flatten(net), 256, activation=tf.nn.relu, kernel_initializer=initializer)
        net = tf.layers.dense(net, env.action_space.n, kernel_initializer=initializer)
        #print(tf.summary.tensor_summary("net", net))
    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)
    return net, trainable_vars

# Now for the training operations
learning_rate = 1e-3
#learning_rate = 1 if args.pretrain else 1e-2
training_start = 10  # start training after 100 game steps
discount_rate = 0.99
batch_size = 16

with tf.variable_scope("train"):
    X_state = tf.placeholder(tf.float32,
    shape=[None, 200, 150, 4])
    X_next_state = tf.placeholder(tf.float32,
    shape=[None, 200, 150, 4])
    X_action = tf.placeholder(tf.int32, shape=[None])
    X_done = tf.placeholder(tf.float32, shape=[None])
    X_rewards = tf.placeholder(tf.float32, shape=[None])
    online_q_values, online_vars = q_network(X_state, name="q_networks/online")
    target_q_values, target_vars = q_network(X_next_state, name="q_networks/online", reuse=True)
    max_target_q_values = tf.reduce_max(target_q_values, axis=1)
    target = X_rewards + (1. - X_done) * discount_rate * max_target_q_values
    q_value = tf.reduce_sum(online_q_values * tf.one_hot(X_action, env.action_space.n), axis=1)
    error = tf.abs(q_value - tf.stop_gradient(target))
    clipped_error = tf.clip_by_value(error, 0.0, 1.0)
    linear_error = 2 * (error - clipped_error)
    loss = tf.reduce_mean(tf.square(clipped_error) + linear_error)

    global_step = tf.Variable(0, trainable=False, name='global_step')
    optimizer = tf.train.AdamOptimizer(learning_rate)
    training_op = optimizer.minimize(loss, global_step=global_step)

# We need an operation to copy the online DQN to the target DQN
copy_ops = [target_var.assign(online_var)
            for target_var, online_var in zip(target_vars, online_vars)]
copy_online_to_target = tf.group(*copy_ops)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Let's implement a simple replay memory
replay_memory = deque([], maxlen=10000)

def sample_memories(batch_size):
    indices = np.random.permutation(len(replay_memory))[:batch_size]
    cols = [[], [], [], [], []] # state, action, reward, next_state, continue
    for idx in indices:
        memory = replay_memory[idx]
        for col, value in zip(cols, memory):
            col.append(value)
    cols = [np.array(col) for col in cols]
    return cols

# And on to the epsilon-greedy policy with decaying epsilon
eps_min = 0.10
eps_max = 1.0 if not args.test else eps_min

def epsilon_greedy(q_values, step):
    epsilon = max(eps_min, eps_max - (eps_max-eps_min) * step / args.explore_steps)
    if np.random.rand() < 0.5:
        return 0
    elif np.random.rand() < epsilon:
        return np.random.randint(env.action_space.n) # random action
    else:
        return np.argmax(q_values) # optimal action

done = True # env needs to be reset

# We will keep track of the max Q-Value over time and compute the mean per game
loss_val = np.infty
game_length = 0
total_max_q = 0
mean_max_q = 0.0
returnn = 0.0
returns = []
steps = []

total_actions = [0] * 6 # To log number of times actions have been taken

if args.pretrain:
    file = open("pretrain.txt","r")
    pretrain_actions = file.read()

path = os.path.join(args.jobid, "model")
with tf.Session() as sess:
    if os.path.isfile(path + ".index"):
        saver.restore(sess, path)
    else:
        init.run()
        copy_online_to_target.run()
    for step in range(args.number_steps):
        training_iter = global_step.eval()
        if done: # game over, start again
            if args.verbosity > 0:
                print("Step {}/{} ({:.1f})% Training iters {}   "
                      "Loss {:5f}    Mean Max-Q {:5f}   Return: {:5f}".format(
                step, args.number_steps, step * 100 / args.number_steps,
                training_iter, loss_val, mean_max_q, returnn))
                sys.stdout.flush()
            state = env.reset()

        # Online DQN evaluates what to do
        q_values = online_q_values.eval(feed_dict={X_state: [state]})
        action = epsilon_greedy(q_values, step)
        print(action)
        # Online DQN plays
        if args.pretrain and step * 2 < len(pretrain_actions):
            print(pretrain_actions[step * 2].split(',')[0])
            action = env.ACTION.index(pretrain_actions[step * 2].split(',')[0]) # * 2 because the keylogger
                                                     # Does \n between every key

        print(action)
        next_state, reward, done, info = env.step(action)
        returnn += reward

        #print("Time : ", start_time - time.time())
        start_time = time.time()
        """
        counter+=1
        if (time.time() - start_time) > x :
            print("FPS: ", counter / (time.time() - start_time))
            counter = 0
            start_time = time.time()
        """


        total_actions[action]+=1
        #print("\nw  a  s  d \" \" ent")
        #print("Total actions [%s]" % ", ".join(map(str, total_actions)))
        #print("Rewarded " + str(reward))
        #print("Total " + str(returnn) + "\n")

        # Let's memorize what happened
        replay_memory.append((state, action, reward, next_state, done))
        state = next_state

        if args.test:
            continue

        # Compute statistics for tracking progress (not shown in the book)
        total_max_q += q_values.max()
        game_length += 1
        if done:
            steps.append(step)
            returns.append(returnn)
            returnn = 0.
            mean_max_q = total_max_q / game_length
            total_max_q = 0.0
            game_length = 0

        if step < training_start or step % args.learn_freq != 0:
            continue # only train after warmup period and at regular intervals

        # Sample memories and train the online DQN
        X_state_val, X_action_val, X_rewards_val, X_next_state_val, X_done_val = sample_memories(batch_size)

        _, loss_val = sess.run([training_op, loss],
        {X_state: X_state_val,
        X_action: X_action_val,
        X_rewards: X_rewards_val,
        X_done: X_done_val,
        X_next_state: X_next_state_val})

        # Regularly copy the online DQN to the target DQN
        if step % args.copy_steps == 0:
            copy_online_to_target.run()

        # And save regularly
        if step % args.save_steps == 0:
            saver.save(sess, path)
            np.save(os.path.join(args.jobid, "{}.npy".format(args.jobid)), np.array((steps, returns)))

        # Next level every x steps
        if step % args.nextlevel == 0:
            env.next_level()
