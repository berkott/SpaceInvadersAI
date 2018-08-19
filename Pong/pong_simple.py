import gym
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam, RMSprop
import numpy as np
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as PLT
import time
import csv
import h5py
import math

# Hyper parameters
L1 = 200
L2 = 100

LEARNING_RATE = 0.001
DISCOUNT_RATE = 0.99
PLAYING_BATCH = 5
DIFF_IMG_FRAMES_GAP = 7
EPOCHS_PER_EPISODE = 3

WIDTH=80
HEIGHT=80
INPUT_DIM=WIDTH*HEIGHT

# Skips game play and generates random values
TESTING = False

def createModel():
    model = Sequential()
    model.add(Dense(L1, activation = 'relu', input_dim = INPUT_DIM))
    model.add(Dense(L2, activation = 'relu'))
    model.add(Dense(1, activation ='sigmoid'))
    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss='mean_squared_error', optimizer=adam)
    return model

def write_csv(index, data):
    slack_logs[index] = data

    # For slack_logs:
    # [0] Scores
    # [1] All Time HS
    # [2] Start Time
    # [3] Games Played

    with open("logs.csv", "w", newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerows(slack_logs)

def write_scores(data):
    df = pd.DataFrame(data)
    df.to_csv('scores.csv')

def convert_prediction_to_action(prediction):
    print(prediction.shape)
    randomNumber = np.random.random()
    action = 2
    if(randomNumber > prediction[0,0]):
        action = 3 
    print("Prediction ",prediction,prediction[0,0],randomNumber,action)
    return action

def visualize(frame, difference_image):
    PLT.imshow(frame)
    PLT.show()
    PLT.imshow(difference_image)
    PLT.show()

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
    previous_frame = np.zeros((WIDTH, HEIGHT))
    forPrediction=np.zeros((1,WIDTH*HEIGHT))
    state=np.zeros((WIDTH*HEIGHT,1))
    while not done:
        frames += 1
        env.render()
        observation, reward, done, _ = env.step(action)
        frame = observation
        frame = frame[35:195]
        frame = frame[::2, ::2, 0]
        # frame = np.where(frame > 0, 1.0,0)

        difference = frame-previous_frame
        state = difference.flatten()
        
        # if(frames > 20):
        #     visualize(frame, difference)
        
        forPrediction[0]=state
        prediction = model.predict(forPrediction)
        action = convert_prediction_to_action(prediction)

        predictions.append(prediction)
        score+=reward
        rewards.append(reward)
        states.append(np.copy(state))
        actions.append(action-1)
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

def compute_rewards(scores, rewards, frames, computed_frames):
    all_discounted_rewards = []

    for i in range(PLAYING_BATCH):
        discounted_rewards = np.zeros((int(frames[i]),1))
        discounted_rewards.fill(scores[i] + computed_frames[i])
        for j in reversed(range(int(frames[i]))):
            discounted_rewards[j] = scores[i]
            scores[i] = scores[i] * DISCOUNT_RATE
        all_discounted_rewards.append(discounted_rewards)
    
    stacked_rewards = np.vstack(all_discounted_rewards)
    return stacked_rewards

def train_model(states, actions, advantages, predictions):
    stacked_predictions = np.vstack(predictions)
    stacked_actions = np.vstack(actions)
    
    gradients = stacked_actions - stacked_predictions
    gradients *= advantages

    training_data = np.vstack(states)
    target_data = stacked_predictions + LEARNING_RATE * gradients
    
    for _ in range(EPOCHS_PER_EPISODE):
        model.train_on_batch(training_data, target_data)

def save_model():
    model.save_weights('policy_gradients_weights.h5')

def load_model():
    model.load_weights('policy_gradients_weights.h5')

def main():
    write_csv(2, time.time())
    games_played = 0
    all_time_high_score = 0
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
            games_played += 1
            write_csv(3, games_played)
        print("Scores: ", scores)
        write_csv(0, np.max(scores))

        if(np.max(scores) > all_time_high_score):
            all_time_high_score = np.max(scores)
            write_csv(1, all_time_high_score)

        write_scores(scores)

        computed_frames = np.copy(frames)
        computed_scores = compute_advantages(scores)
        advantages = compute_rewards(computed_scores, rewards, frames, computed_frames)

        train_model(states, actions, advantages, predictions)
        
        save_model()

# 
# START the computations
# 
env = gym.make('Pong-v0')
slack_logs = np.zeros((4,1))
model = createModel()

try:
    # load_model()
    main()
except:
    main()
