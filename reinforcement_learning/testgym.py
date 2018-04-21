import gym
import tensorflow as tf
import numpy as np

num_inputs = 4
num_hidden = 4
num_outputs = 1
learning_rate = 0.01

initializer = tf.contrib.layers.variance_scaling_initializer()

X = tf.placeholder(tf.float32, shape=[None, num_inputs])

hidden_layer = tf.layers.dense(
    X, num_hidden, activation=tf.nn.elu, kernel_initializer=initializer
)

logits = tf.layers.dense(hidden_layer, num_outputs)
outputs = tf.nn.sigmoid(logits)

probabilities = tf.concat(axis=1, values=[outputs, 1 - outputs])
action = tf.multinomial(probabilities, num_samples=1)

y = 1.0 - tf.to_float(action)

cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
    labels=y, logits=logits)
optimizer = tf.train.AdamOptimizer(learning_rate)

gradients_and_variables = optimizer.compute_gradients(cross_entropy)

gradients = []
gradient_placeholders = []
grads_and_vars_feed = []

for gradient, variable in gradients_and_variables:
    gradients.append(gradient)
    gradient_placeholder = tf.placeholder(
        tf.float32, shape=gradient.get_shape()
    )
    gradient_placeholders.append(gradient_placeholder)
    grads_and_vars_feed.append((gradient_placeholder, variable))

training_op = optimizer.apply_gradients(grads_and_vars_feed)

init = tf.global_variables_initializer()


def helper_discount_rewards(rewards, discount_rate):
    '''
    Takes in rewards and applies discount rate
    '''
    discounted_rewards = np.zeros(len(rewards))
    cumulative_rewards = 0
    for step in reversed(range(len(rewards))):
        cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate
        discounted_rewards[step] = cumulative_rewards
    return discounted_rewards


def discount_and_normalize_rewards(all_rewards, discount_rate):
    '''
    Takes in all rewards, applies helper_discount function and then normalizes
    using mean and std.
    '''
    all_discounted_rewards = []
    for rewards in all_rewards:
        all_discounted_rewards.append(helper_discount_rewards(
            rewards,
            discount_rate)
        )

    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean) /
            reward_std for discounted_rewards in all_discounted_rewards]


saver = tf.train.Saver()

env = gym.make("CartPole-v0")
max_game_steps = 1000
num_game_rounds = 10
num_iterations = 650
discount_rate = 0.9

with tf.Session() as sess:
    sess.run(init)

    for iteration in range(num_iterations):
        print("Currently on Iteration: {} \n".format(iteration))

        all_rewards = []
        all_gradients = []

        # Play n amount of game rounds
        for game in range(num_game_rounds):

            current_rewards = []
            current_gradients = []

            observations = env.reset()

            # Only allow n amount of steps in game
            for step in range(max_game_steps):

                # Get Actions and Gradients
                action_val, gradients_val = sess.run([action, gradients], feed_dict={
                                                     X: observations.reshape(1, num_inputs)})

                # Perform Action
                observations, reward, done, info = env.step(action_val[0][0])

                # Get Current Rewards and Gradients
                current_rewards.append(reward)
                current_gradients.append(gradients_val)

                if done:
                    # Game Ended
                    break

            # Append to list of all rewards
            all_rewards.append(current_rewards)
            all_gradients.append(current_gradients)

        all_rewards = discount_and_normalize_rewards(all_rewards, discount_rate)
        feed_dict = {}

        for var_index, gradient_placeholder in enumerate(gradient_placeholders):
            mean_gradients = np.mean([reward * all_gradients[game_index][step][var_index]
                                      for game_index, rewards in enumerate(all_rewards)
                                      for step, reward in enumerate(rewards)], axis=0)
            feed_dict[gradient_placeholder] = mean_gradients

        sess.run(training_op, feed_dict=feed_dict)

    print('SAVING GRAPH AND SESSION')
    #meta_graph_def = tf.train.export_meta_graph(filename='/my-650-step-model.meta')
    saver.save(sess, '/my-650-step-model')
