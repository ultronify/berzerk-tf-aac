import gym
import numpy as np
import tensorflow as tf
from data_cleaning import sanitize_state
from tqdm import tqdm
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Flatten


class ActorCriticModel(Model):
    def __init__(self, state_shape, action_space_size):
        super(ActorCriticModel, self).__init__()
        self.state_shape = state_shape
        self.action_space_state = action_space_size
        # Image feature extraction layers
        self.conv_input = Conv2D(16, (8, 8), strides=(4, 4), activation='relu', kernel_initializer='he_uniform', input_shape=self.state_shape, trainable=True)
        self.conv_0 = Conv2D(32, (4, 4), strides=(2, 2), activation='relu', kernel_initializer='he_uniform', trainable=True)
        # self.conv_input = tf.keras.applications.MobileNetV2(input_shape=self.state_shape, alpha=1.0, include_top=False, weights='imagenet')
        # self.conv_input.trainable = False
        self.bn_0 = BatchNormalization()
        self.flatten_layer = Flatten()
        # Actor layers
        self.actor_dense_0 = Dense(64, kernel_initializer='he_uniform', activation='relu')
        self.actor_output = Dense(
            self.action_space_state, activation='softmax')
        # Critic layers
        self.critic_dense_0 = Dense(64, kernel_initializer='he_uniform', activation='relu')
        self.critic_output = Dense(1, activation='linear')

    def call(self, inputs, **kwargs):
        # Feature extraction
        x = self.conv_input(inputs)
        x = self.conv_0(x)
        x = self.bn_0(x)
        x = self.flatten_layer(x)
        # Actor network
        p = self.actor_dense_0(x)
        p = self.actor_output(p)
        # Critic network
        v = self.critic_dense_0(x)
        v = self.critic_output(v)
        return p, v


def sample_action(action_space_size, probs, use_max=False):
    if use_max:
        return np.argmax(probs)
    else:
        return np.random.choice(action_space_size, p=probs/probs.sum())


def compute_discounted_rewards(rewards, gamma):
    discounted_reward = 0
    discounted_rewards = []
    for reward in rewards[::-1]:
        discounted_reward = gamma * discounted_reward + reward
        discounted_rewards.append([discounted_reward])
    return discounted_rewards[::-1]


def eval(model, env, max_eps, action_space_size, render):
    total_reward = 0.0
    print('Start running eval')
    for eps in range(max_eps):
        done = False
        state = env.reset()
        # print('Start running eval on episode {0}/{1}'.format(eps, max_eps))
        while not done:
            if render:
                env.render()
            sanitized_state = sanitize_state(np.array(state))
            action_dist, _ = model(tf.convert_to_tensor([sanitized_state], dtype=tf.float32))
            action = sample_action(
                action_space_size, action_dist.numpy()[0], use_max=True)
            state, reward, done, _ = env.step(action)
            total_reward += reward
        # print('Finished running eval on episode {0}/{1}'.format(eps, max_eps))
    avg_reward = total_reward / max_eps
    return avg_reward


def train(max_eps=1000, gamma=0.99, render=False, initial_max_trail_steps=400):
    env = gym.make('Berzerk-v0')
    eval_env = gym.make('Berzerk-v0')
    if render:
        env.render()
        eval_env.render()
    state_shape = sanitize_state(np.array(env.reset())).shape
    action_space_size = env.action_space.n
    print('Construct model with action space size {0} and state shape {1}'.format(
        action_space_size, state_shape))
    model = ActorCriticModel(state_shape, action_space_size)
    optimizer = tf.optimizers.Adam(learning_rate=1e-3)
    max_trail_steps = initial_max_trail_steps
    for eps in range(max_eps):
        done = False
        state = env.reset()
        actions, rewards, states = [], [], []
        # while not done and trail_step_cnt < max_trail_steps:
        while not done:
            if render:
                env.render()
            sanitized_state = sanitize_state(np.array(state))
            action_dist, _ = model(tf.convert_to_tensor(
                [sanitized_state], dtype=tf.float32))
            action = sample_action(action_space_size, action_dist.numpy()[0])
            next_state, reward, done, _ = env.step(action)
            actions.append(action)
            states.append(sanitized_state)
            rewards.append(reward)
            state = next_state
            # print('Sampling step and got reward {0}'.format(reward))
        max_trail_steps += 10
        # Calculate loss and gradients
        # print('Finished sampling trajectory with size {0}'.format(len(states)))
        discounted_rewards = compute_discounted_rewards(rewards, gamma)
        action_onehot_data = tf.one_hot(actions, action_space_size)
        """
        print('states shape: {0}, actions shape: {1}, q vals shape: {2}'.format(
            tf.shape(tf.convert_to_tensor(states, dtype=tf.float32)),
            tf.shape(action_onehot_data),
            tf.shape(tf.convert_to_tensor(discounted_rewards, dtype=tf.float32)),
        ))
        """
        dataset = tf.data.Dataset.from_tensor_slices((
            tf.convert_to_tensor(states, dtype=tf.float32),
            action_onehot_data,
            tf.convert_to_tensor(discounted_rewards, dtype=tf.float32)
        ))
        # print('dataset: ', dataset)
        # print('dataset size: ', len(dataset))
        for batch in tqdm(dataset.batch(32)):
            # print('batch size: ', len(batch))
            with tf.GradientTape() as tape:
                # print('states shape: {0}, actions shape: {1}, q vals shape: {2}'.format(tf.shape(batch[0]), tf.shape(batch[1]), tf.shape(batch[2])))
                probs_raw, vals = model(batch[0])
                # print('probs shape: {0}, vals shape: {1}'.format(tf.shape(probs_raw), tf.shape(vals)))
                probs = tf.clip_by_value(probs_raw, 1e-10, 1-1e-10)
                log_probs = tf.math.log(probs)
                q_vals = batch[2]
                advantage = q_vals - vals
                action_onehot = batch[1]
                val_loss = advantage ** 2
                policy_loss = -(log_probs * action_onehot) * advantage
                entropy_loss = -tf.reduce_sum(probs * log_probs)
                loss = 0.5 * tf.reduce_mean(val_loss) + tf.reduce_mean(policy_loss) + 0.01 * entropy_loss
            # print('Finished calculating loss')
            grads = tape.gradient(loss, model.trainable_weights)
            # print('Finished calculating gradients')
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            # print('Finished applying gradients')
        if eps % 10 == 0:
            eval_score = eval(model, eval_env, 1, action_space_size, render)
            print('Finished training {0}/{1} with score {2}'.format(eps, max_eps, eval_score))
    env.close()
