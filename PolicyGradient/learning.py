import gym
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
import numpy as np
from datetime import datetime
from matplotlib import pyplot as PLT
import time
import csv
import os

# Hyper parameters
L1 = 20
L2 = 10
L3 = 50
L4 = 4
LEARNING_RATE = 0.001
DISCOUNT_RATE = 0.99
REWARD_RATE = 1.5
PLAYING_BATCH = 3
FILTER_SIZE_1 = (3,3)
FILTER_SIZE_2 = (5,5)
POOLING_SIZE = (2,2)

WIDTH=210
HEIGHT=160
FRAME_SIZE = 210*160*1
INPUT_SHAPE = (210, 160, 2)
INPUT_DIM = 2*FRAME_SIZE

# Skips game play and generates random values
TESTING = False

model = Sequential()
model.add(Conv2D(L1, FILTER_SIZE_1, activation='relu', input_shape = INPUT_SHAPE, kernel_initializer='normal'))
model.add(MaxPooling2D(pool_size=POOLING_SIZE))
model.add(Conv2D(L2, FILTER_SIZE_2, activation='relu', kernel_initializer='normal'))
model.add(MaxPooling2D(pool_size=POOLING_SIZE))
# model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(L3, activation = 'relu', kernel_initializer='normal'))
model.add(Dense(L4, activation ='softmax', kernel_initializer='normal'))
adam = Adam(lr=LEARNING_RATE)
model.compile(loss='mean_squared_error', optimizer=adam)

env = gym.make('SpaceInvaders-v0')
highest_score = 0

def convert_prediction_to_action(prediction, game_type_action):
    # Training
    if(TESTING == False):
        index = np.random.choice(4, p=prediction)
    else:
        index = np.argmax(prediction[0])
    # Testing
    # index = np.argmax(prediction[0])

    # DO NOTHING
    if (index == 0):
        if(game_type_action):
            return 0
        else:
            return [1,0,0,0]
    # FIRE
    elif (index == 1):
        if(game_type_action):
            return 1
        else:
            return [0,1,0,0]
    # RIGHT
    elif (index == 2):
        if(game_type_action):
            return 3
        else:
            return [0,0,1,0]
    # LEFT
    elif (index == 3):
        if(game_type_action):
            return 4
        else:
            return [0,0,0,1]
    return 0

def play_game():
    env.reset()
    
    score=0
    states = []
    actions = []
    rewards = []
    predictions = []
    done=False
    action=0
    frames = 0
    
    frame = np.zeros((WIDTH,HEIGHT))
    previous_frame = np.zeros((WIDTH,HEIGHT))
    forPrediction=np.zeros((1,WIDTH,HEIGHT,2))
    state=np.zeros((WIDTH,HEIGHT,2))
    while not done:
        frames += 1
        env.render()
        observation, reward, done, _ = env.step(action)
        frame = observation[:,:,0]
        frame = np.where(frame > 0, 1.0,0)
        difference = frame-previous_frame
        state[:,:,0]=frame
        state[:,:,1]=difference
        states.append(np.copy(state))
        
        forPrediction[0]=state
        prediction = model.predict(forPrediction).flatten()
        predictions.append(prediction.astype(np.float).ravel())
        action = convert_prediction_to_action(prediction, True)
        score+=reward
        rewards.append(reward)
        actions.append(np.array(convert_prediction_to_action(prediction, False)))

        previous_frame = np.copy(frame)
    return states, actions, rewards, predictions, score, frames

def fill_values():
    frames = np.random.randint(1500)
    print(frames)
    states = []
    actions = []
    rewards = []
    predictions = []
    score = 0

    reward = 0
    for _ in range(frames):
        observation_dim = list(INPUT_SHAPE)
        observation_dim.insert(0,1)
        observation_dim = tuple(observation_dim)    
        state = np.random.rand(observation_dim[0], observation_dim[1], observation_dim[2], observation_dim[3])
        states.append(state)
        # prediction = model.predict(state).flatten()
        prediction = np.random.rand(4)
        predictions.append(prediction.astype(np.float).ravel())

        actions.append(np.array(convert_prediction_to_action(prediction, False)))

        if(np.random.randint(100, size=1) == 1):
            reward = 5
        rewards.append(reward)
        score += reward
        reward = 0

    return states, actions, rewards, predictions, score, frames

def compute_advantages(scores):
    scores -= np.mean(scores)
    if (np.std(scores) != 0):
        scores /= np.std(scores)
    return scores

def compute_rewards(scores, rewards, frames):
    print(scores)
    all_discounted_rewards = []
    for i in range(PLAYING_BATCH):
        discounted_rewards = np.zeros((int(frames[i]),1))
        for j in reversed(range(int(frames[i]))):
            discounted_rewards[j] = scores[i]
            scores[i] = scores[i] * DISCOUNT_RATE
            if(rewards[i][j] > 1):
                scores[i] = scores[i] * REWARD_RATE
        # discounted_rewards = np.fliplr([discounted_rewards])[0]
        # if (np.std(discounted_rewards) != 0):
        #     discounted_rewards /= np.std(discounted_rewards)
        all_discounted_rewards.append(discounted_rewards)
    stacked_rewards = np.vstack(all_discounted_rewards)
    if (np.std(stacked_rewards) != 0):
        stacked_rewards = stacked_rewards / np.std(stacked_rewards - np.mean(stacked_rewards))
    return stacked_rewards

def train_model(states, actions, advantages, predictions):
    stacked_predictions = np.vstack(predictions)
    stacked_actions = np.vstack(actions)
    
    print("Predictions: ", stacked_predictions, ", Actions: ", stacked_actions)
    gradients = stacked_actions - stacked_predictions
    gradients *= advantages

    training_data = np.vstack(states)
    target_data = stacked_predictions + LEARNING_RATE * gradients
    
    model.train_on_batch(training_data, target_data)

while True:
    states = []
    actions = []
    rewards = []
    predictions = []
    scores = np.zeros(PLAYING_BATCH)
    frames = np.zeros(PLAYING_BATCH)
    
    for i in range(PLAYING_BATCH):
        if(TESTING == False):
            state, action, reward, prediction, score, frame = play_game()
        else:
            state, action, reward, prediction, score, frame = fill_values()
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        predictions.append(prediction)
        scores[i] = score
        frames[i] = frame
    computed_scores = compute_advantages(scores)
    advantages = compute_rewards(computed_scores, rewards, frames)

    train_model(states, actions, advantages, predictions)
    print(np.argmax(scores))
