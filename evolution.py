import gym
import keras as k
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
import numpy as np
from datetime import datetime
from matplotlib import pyplot as PLT
import time

FRAME_SIZE=210*160*1
INPUT_DIM=2*FRAME_SIZE
POPULATION_SIZE = 100
L1=200
L2=4
ELITE_SET_SIZE = 5
MUTATION_RATE = 0.05

env = gym.make('SpaceInvaders-v0')
keepTraining = True

def visualize(featureVector):
    regularImage = featureVector[0,:FRAME_SIZE].reshape((210,160))
    differenceImage = featureVector[0,FRAME_SIZE:].reshape((210,160))
    PLT.imshow(regularImage)
    PLT.show()
    PLT.imshow(differenceImage)
    PLT.show()

def calculatePolicySize():
    return INPUT_DIM*L1+L1+L1*L2+L2

# This function is called each time a new memeber of the population is created
def initPopulation():
    population = np.random.rand(POPULATION_SIZE, calculatePolicySize())
    population = population*2-1
    return population

def convert_prediction_to_action(prediction):
    index = np.argmax(prediction[0])
    # NOOP
    if (index == 0):
        return 0
    # FIRE
    elif (index == 1):
        return 1
    # RIGHT
    elif (index == 2):
        return 3
    # LEFT
    elif (index == 3):
        return 4
    return 0

def playGame(model):
    score=0
    done=False
    action=0
    frame = np.zeros((1,FRAME_SIZE))
    previous_frame = np.zeros((1,FRAME_SIZE))
    final_obervation=np.zeros((1,INPUT_DIM))
    env.reset()
    framenumber=0
    while not done:
        framenumber+=1
        env.render()
        observation, reward, done, info = env.step(action)
        frame = np.reshape(observation[:,:,0],(1,FRAME_SIZE))
        frame = np.where(frame > 0, 1.0,0)
        difference = frame-previous_frame
        final_obervation[0,:FRAME_SIZE]=frame
        final_obervation[0,FRAME_SIZE:]=difference
        prediction = model.predict(final_obervation)
        action = convert_prediction_to_action(prediction)
        score+=reward
        previous_frame = np.copy(frame)

    # print("Score:",score)
    return score

# This is where the weights are put into the neural net to see how well it goes
def evaluate(dnnmodel, population):
    scores=np.zeros(POPULATION_SIZE)
    for i in range(POPULATION_SIZE):
        nnFormatPolicyVector = applyPolicyVectorToNN(population[i])
        dnnmodel.set_weights(nnFormatPolicyVector)
        scores[i] = playGame(dnnmodel)
    return scores


# Constructs the model that is to be used
def buildModel():
    model = Sequential()
    layer1=Dense(L1, activation = 'relu', input_dim = INPUT_DIM, kernel_initializer='uniform')
    model.add(layer1)
    layer2=Dense(L2, activation ='softmax', kernel_initializer='uniform')
    model.add(layer2)
    adam = Adam(lr=0.01)
    model.compile(loss='mean_squared_error', optimizer=adam)
    return model

def applyPolicyVectorToNN(policyVector):
    sec1 = (policyVector[:INPUT_DIM*L1]).reshape(INPUT_DIM,L1)
    sec2 = policyVector[INPUT_DIM*L1:INPUT_DIM*L1+L1]
    sec3 = policyVector[INPUT_DIM*L1+L1:INPUT_DIM*L1+L1+L1*L2].reshape(L1,L2)
    sec4 = policyVector[INPUT_DIM*L1+L1+L1*L2:]

    nnFormat = []
    nnFormat.append(sec1)
    nnFormat.append(sec2)
    nnFormat.append(sec3)
    nnFormat.append(sec4)
    return nnFormat

# This is where the members of the population are ranked
def selection(scores, population):
    eliteSet = []
    scoresTemp=np.copy(scores)
    for _ in range(ELITE_SET_SIZE):
        index = np.argmax(scoresTemp)
        scoresTemp[index] = 0
        eliteSet.append(population[index])
    return eliteSet

def cross(policy1, policy2):
    newPolicy = policy1.copy()
    for i in range(calculatePolicySize()):
        rand = np.random.uniform()
        if rand > 0.5:
            newPolicy[i] = policy2[i]
    return newPolicy

# This is where crossover occurs based on the selection process
def crossover(scores, population):
    crossoverSet = []
    selectionProbability = np.array(scores)/np.sum(scores)
    for i in range(POPULATION_SIZE - ELITE_SET_SIZE):
        randomIndex = np.random.choice(range(POPULATION_SIZE), p=selectionProbability)
        policy1 = population[randomIndex]
        randomIndex = np.random.choice(range(POPULATION_SIZE), p=selectionProbability)
        policy2 = population[randomIndex]
        newPolicy = cross(policy1, policy2)
        crossoverSet.append(newPolicy)
    return crossoverSet

# Lastly, the mutation is a point mutation that sometimes occurs
def mutation(crossoverPopulation):
    for i in range(POPULATION_SIZE - ELITE_SET_SIZE):
        for j in range(calculatePolicySize()):
            rand = np.random.uniform()
            if(rand < MUTATION_RATE):
                crossoverPopulation[i][j] = np.random.random_sample() * 2 - 1
    return crossoverPopulation

def generateNewGeneration(scores, population):
    elitePopulation = selection(scores, population)
    crossoverPopulation = crossover(scores, population)
    mutationPopulation = mutation(crossoverPopulation)
    mutationPopulation.extend(elitePopulation)
    return mutationPopulation

def saveHighestScorePolicy(population, generation, scores):
    if(generation%5==0):
        index = np.argmax(scores)
        filename='generation'+str(generation)+'HS'+str(scores[index])+'.npy'
        np.save(filename,population[index])
        print("Saved generation to file "+filename)

def loadPolicy(filename, population, index):
    policy=np.load(filename)
    print("Loaded\n",policy)
    population[index]=policy

def measureTime():
    global lasttime
    currentTime=time.time()
    diff=currentTime-lasttime
    lasttime=currentTime
    return diff

env.reset()
population = initPopulation()
# loadPolicy('generation0.npy',population,0)
dnnmodel = buildModel()
generation = 0
lasttime=time.time()
while (keepTraining):
    scores = evaluate(dnnmodel, population)
    print(measureTime()," sec Generation: ", generation, " Highest Score: ", np.max(scores), ' Games Played: ', generation*POPULATION_SIZE)
    saveHighestScorePolicy(population,generation, scores)
    population = generateNewGeneration(scores, population)
    print(measureTime()," sec New generation created.")
    generation+=1
