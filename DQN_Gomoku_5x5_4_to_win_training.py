'''
Python 4x4 OX Game - Mark Kang Sep.2019
'''
import random
import numpy as np
import copy
import keras
import sys
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras import optimizers

#from google.colab import drive
#drive.mount('/content/drive')

EPISODES = 5000000

EMPTY_SPACE = 0
WHITE_PLAYER = -1
BLACK_PLAYER = 1


# define width and length of MYMAP, it will be used in this way:
# MYMAP[Y_OF_MYMAP][X_OF_MYMAP]
Y_OF_MYMAP = 5
X_OF_MYMAP = 5
# How many continues chess is WIN?
WIN_CONDITION = 4

MYMAP = np.zeros((X_OF_MYMAP,Y_OF_MYMAP), dtype=int)

Running_MAP = 0 # runtime map, it will initialized as MYMAP.

# Define total actions
gTOTAL_ACTIONs = Y_OF_MYMAP * X_OF_MYMAP

# NN parameters
gGAMMA = 0.75
gLEARNNING_RATE = 0.00001

# Other parameters
WINNER_REWARD = 1
LOSER_REWARD = -1
DRAW_REWARD = 0

#############################

def env_render(pMYMAP):
    for y in range(0,Y_OF_MYMAP,1):
        for x in range(0,X_OF_MYMAP,1):
            if pMYMAP[0][x+y*Y_OF_MYMAP] == WHITE_PLAYER:
                print("O",end="")
            elif pMYMAP[0][x+y*Y_OF_MYMAP]  == BLACK_PLAYER:
                print("X",end="")
            elif pMYMAP[0][x+y*Y_OF_MYMAP]  == EMPTY_SPACE:
                print("_",end="")
            else:
                print("PANIC, unknow element of MYMAP")
                exit()
        print("") # chnage to new line

def check_state_has_winner(nn_state, who_is_playing):
    
    # Check every X line
    for y in range(0, gSTATE_SIZE, Y_OF_MYMAP):
        for x in range(0, X_OF_MYMAP-WIN_CONDITION+1,1):
            chess_count = 0
            if y+x+WIN_CONDITION-1 < y+X_OF_MYMAP:
                for j in range(0, WIN_CONDITION, 1):
                    if nn_state[0][y+x+j] == who_is_playing:                
                        chess_count=chess_count+1
                        if chess_count == WIN_CONDITION:
                            return True
                    else:
                        break
                    
    # Check every Y line
    for x in range(0, X_OF_MYMAP, 1):
        for y in range(0, Y_OF_MYMAP-WIN_CONDITION+1,1):
            chess_count = 0
            if x+X_OF_MYMAP*y+X_OF_MYMAP*(WIN_CONDITION-1) < gSTATE_SIZE:
                for j in range(0, WIN_CONDITION, 1):
                    if nn_state[0][x+X_OF_MYMAP*y+X_OF_MYMAP*j] == who_is_playing:
                        chess_count=chess_count+1
                        if chess_count == WIN_CONDITION:
                            return True                
                    else:
                        break

    # Check \
    for y in range(0, Y_OF_MYMAP, 1):
        for x in range(0, X_OF_MYMAP, 1):
            chess_count = 0
            for j in range(0,WIN_CONDITION,1):
                if (y+WIN_CONDITION-1)<Y_OF_MYMAP and (x+WIN_CONDITION-1)<X_OF_MYMAP:
                    if nn_state[0][x+X_OF_MYMAP*y+X_OF_MYMAP*j+j] == who_is_playing:
                        chess_count=chess_count+1
                        if chess_count == WIN_CONDITION:
                            return True
                    else:
                        break
                        
    # check / 
    for y in range(0, Y_OF_MYMAP, 1):
        for x in range(X_OF_MYMAP - 1, -1, -1):
            chess_count = 0
            for j in range(0,WIN_CONDITION,1):
                if (y+WIN_CONDITION-1)<Y_OF_MYMAP and (x-(WIN_CONDITION-1)) >= 0:
                    if nn_state[0][x+X_OF_MYMAP*y+X_OF_MYMAP*j-j] == who_is_playing:
                        chess_count=chess_count+1
                        if chess_count == WIN_CONDITION:
                            return True
                    else:
                        break
                        
    # No winner
    return False

def env_step(nn_state,internal_action,who_is_playing):
    done = False
    reward = 0
    private_next_state = copy.deepcopy(nn_state)

    if private_next_state[0][internal_action] != EMPTY_SPACE:
        print ("PANIC, env_step(),  private_next_state[internal_action] != EMPTY_SPACE")
    else:
        private_next_state[0][internal_action] = who_is_playing

    done = check_state_has_winner(private_next_state,who_is_playing)
    if done == True:
        reward = 100
        return private_next_state, reward, done, 0

    for index in range (0, Y_OF_MYMAP * X_OF_MYMAP ,1):
        if private_next_state[0][index] != EMPTY_SPACE:
            continue
        else:
            return private_next_state, reward, done, 0

    done = True # Full of map, but no winner
    return private_next_state, reward, done, 0

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size =  state_size
        self.action_size = action_size
        self.memory = deque(maxlen=6000)
        self.gamma = gGAMMA    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.3
        self.epsilon_decay = 0.999995
        self.learning_rate = gLEARNNING_RATE
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(512, input_dim=self.state_size, activation='linear'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(512, activation='relu'))        
        model.add(Dense(self.action_size, activation='linear'))
        #model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta())
        #model.compile(loss='mse',optimizer=Adam(lr=self.learning_rate))
        #model.compile(loss='mse',optimizer=optimizers.SGD(lr=0.0001, clipnorm=1.))
        #model.compile(loss='mse',optimizer=keras.optimizers.Adagrad(lr=gLEARNNING_RATE, epsilon=None, decay=0.0))
        model.compile(loss='mse',optimizer=keras.optimizers.RMSprop(lr=gLEARNNING_RATE,rho=0.9, epsilon=None, decay=0.0))
        #model.compile(loss='mse', optimizer=keras.optimizers.Adadelta())
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def predict(self, nn_state):
        return self.model.predict(nn_state)

    def act(self, nn_state, who_is_playing, _time):
        internal_action = -1
        available_location = copy.deepcopy(MYMAP)
        available_location = np.reshape(available_location, [1, gSTATE_SIZE])
        available_location_count = 0

        for x in range (0, Y_OF_MYMAP * X_OF_MYMAP ,1):
            if (nn_state[0][x] == EMPTY_SPACE ):
                available_location[0][available_location_count] = x
                available_location_count=available_location_count+1

        while (True):
            #if np.random.rand() <= self.epsilon or who_is_playing == WHITE_PLAYER:
            #if np.random.rand() <= self.epsilon or who_is_playing == BLACK_PLAYER:
            if np.random.rand() <= self.epsilon:
                index = random.randrange(0, available_location_count,1)
                internal_action = available_location[0][index]
                if nn_state[0][internal_action] != EMPTY_SPACE:
                    internal_action = -1
                    print("random choosing new action")

            else:
                act_values = self.model.predict(nn_state)
                internal_action = np.argmax(act_values[0])
                #print("AI")
                if nn_state[0][internal_action] != EMPTY_SPACE:
                    #print(" AI missed")
                    max_q = -99999
                    for p in range (0,available_location_count,1 ):
                        if max_q < act_values[0][available_location[0][p]]:
                            max_q = act_values[0][available_location[0][p]]
                            internal_action = available_location[0][p]
                        if nn_state[0][internal_action] != EMPTY_SPACE:
                            internal_action = -1

            if internal_action > -1:
                break

        return internal_action, self.model.predict(nn_state)

    def replay(self, batch_size):
        ######### Orignal fetch 
        #minibatch = random.sample(self.memory, batch_size)
        #########
        minibatch = []
        _memory_length = len(self.memory)
        _state_all=np.zeros(( _memory_length,gSTATE_SIZE), dtype=float)
        _target_f_all=np.zeros(( _memory_length,gSTATE_SIZE), dtype=float)
        _index = 0

        for _ in range(_memory_length):            
            #minibatch.append(self.memory.popleft()) 
            minibatch.append(self.memory.pop()) 

        for state, action, replay_reward, next_state, done in minibatch:
            ###############
            internal_action = -1
            unavailable_location = copy.deepcopy(MYMAP)
            unavailable_location = np.reshape(unavailable_location, [1, gSTATE_SIZE])
            unavailable_location_count = 0           
            for x in range (0, Y_OF_MYMAP * X_OF_MYMAP ,1):
                if (state[0][x] != EMPTY_SPACE ):
                    unavailable_location[0][unavailable_location_count] = x
                    unavailable_location_count=unavailable_location_count+1
            #################
            target = replay_reward
            if not done:
                target = replay_reward + self.gamma * np.amax(self.model.predict(next_state)[0])                         
            target_f = self.model.predict(state)
            target_f[0][action] = target
            ####
            for x in range (0, unavailable_location_count, 1):
                target_f[0][unavailable_location[0][x]] = LOSER_REWARD * 2
            ####
            _state_all[_index]=copy.deepcopy(state)
            _target_f_all[_index]=copy.deepcopy(target_f)
            _index = _index + 1            
            #if done == True:
            #    self.model.fit(state, target_f, shuffle=False, epochs=5, verbose=0)            
            #else:            
            #    self.model.fit(state, target_f, shuffle=False, epochs=5, verbose=0)            
        self.model.fit(_state_all, _target_f_all, batch_size=128, shuffle=False, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

if __name__ == "__main__":
    gSTATE_SIZE = X_OF_MYMAP * Y_OF_MYMAP
    Agent_Black = DQNAgent(gSTATE_SIZE, gTOTAL_ACTIONs)
    Agent_White = DQNAgent(gSTATE_SIZE, gTOTAL_ACTIONs)
    Win_Black_Count = 0
    Win_White_Count = 0
    Win_Draw_Count = 0

    done = False
    BLACK_batch_size = 96
    WHITE_batch_size = 96

    Agent_Black.load("5x5_get_4_14layer_BLACK.h5")
    Agent_White.load("5x5_get_4_14layer_WHITE.h5")

    for e in range(EPISODES+1):
        # Enviroment Reset
        Running_MAP = copy.deepcopy(MYMAP)
        Who_is_playing  = BLACK_PLAYER
        BLACK_Last_Action = -1
        WHITE_Last_Action = -1
        
        NN_state = np.reshape(Running_MAP, [1, gSTATE_SIZE]) # reshape Running_MAP to NN input form
        BLACK_Last_State = copy.deepcopy(NN_state)
        
        #env_render(NN_state)

        for time in range(1, Y_OF_MYMAP * X_OF_MYMAP+1, 1):
            if (Who_is_playing == WHITE_PLAYER):
                Action,Q_Values = Agent_White.act(NN_state, Who_is_playing, time)
                ## AI V.S Human
                #if (Agent_Black.epsilon<0.05):
                #if e > 1000000:
                #    env_render(NN_state)
                #    Action = int(input("Enter a location(0~8): "))

            elif (Who_is_playing == BLACK_PLAYER):
                Action,Q_Values = Agent_Black.act(NN_state, Who_is_playing, time)

            ####### Hard coding actions for debug START ######### 
            # 3x3
            #if time == 1 and Who_is_playing == BLACK_PLAYER:               
            #    Action = 4
            #if time == 2 and Who_is_playing == WHITE_PLAYER:               
            #    Action = 6
            #if time == 3 and Who_is_playing == BLACK_PLAYER:               
            #    Action = 1    
            #if time == 4 and Who_is_playing == WHITE_PLAYER:               
            #    Action = 4
            #if time == 5 and Who_is_playing == BLACK_PLAYER:               
            #    Action = 2
            #if time == 6 and Who_is_playing == WHITE_PLAYER:               
            #    if NN_state[0][7] ==  EMPTY_SPACE:
            #        Action = 7 
            #if time == 7 and Who_is_playing == BLACK_PLAYER:               
            #    Action = 6
            ####### Hard coding actions for debug END #########
            #VERIFY_ROUND = 11
            
            #if time == VERIFY_ROUND:
            #   env_render(NN_state)
            #   for i in range (0, X_OF_MYMAP*Y_OF_MYMAP,X_OF_MYMAP):
            #       #print ("{}\t{}\t{}\t{}\t{}".format(Q_Values[0][i],Q_Values[0][i+1],Q_Values[0][i+2],Q_Values[0][i+3],Q_Values[0][i+4] ))
            #       print ("{}\t{}\t{}\t{}".format(Q_Values[0][i],Q_Values[0][i+1],Q_Values[0][i+2],Q_Values[0][i+3]))
            #       #print ("{}\t{}\t{}".format(Q_Values[0][i],Q_Values[0][i+1],Q_Values[0][i+2] ))
            #   print("time={}---Who_is_playing={}----Qmax={} \t Qmin={}----Action={}--"
            #         .format(time,Who_is_playing,np.argmax(Q_Values[0]),np.argmin(Q_Values[0]),Action))

            ###################
            next_state, reward, done, _ = env_step(NN_state, Action, Who_is_playing) #env.step(action)
            ###################

            if reward == 0:      # DRAW
                BLACK_reward = 0 # TODO may not hard code here.
                WHITE_reward = 0 # TODO may not hard code here.

            if done and reward == 100: # Someone WIN
                if Who_is_playing == BLACK_PLAYER:      # BLACK_PLAYER WIN
                    BLACK_reward = WINNER_REWARD
                    WHITE_reward = LOSER_REWARD

                    #Agent_Black.remember(BLACK_Last_State, BLACK_Last_Action, WINNER_REWARD/2, NN_state, False)
                    Agent_Black.remember(BLACK_Last_State, BLACK_Last_Action, WINNER_REWARD * gGAMMA, NN_state, True)
                    Agent_Black.remember(NN_state, Action, WINNER_REWARD, next_state, done) # Black WIN                    
                    Agent_White.remember(WHITE_Last_State, WHITE_Last_Action, LOSER_REWARD, next_state, True)

                elif Who_is_playing == WHITE_PLAYER:    # WHITE_PLAYER WIN
                    BLACK_reward = LOSER_REWARD
                    WHITE_reward = WINNER_REWARD
                    
                    #Agent_White.remember(WHITE_Last_State, WHITE_Last_Action, WINNER_REWARD/2, NN_state, False)
                    Agent_White.remember(WHITE_Last_State, WHITE_Last_Action, WINNER_REWARD * gGAMMA, NN_state, True)
                    Agent_White.remember(NN_state, Action, WINNER_REWARD, next_state, done) # WHITE WIN
                    Agent_Black.remember(BLACK_Last_State, BLACK_Last_Action, LOSER_REWARD, next_state, True)           

                else:
                    print ("PANIC - unknown who is Winner")
                    sys.exit()
                    
            elif done and reward ==0: ## DRAW
                if Who_is_playing == BLACK_PLAYER:
                    #Agent_Black.remember(BLACK_Last_State, BLACK_Last_Action, DRAW_REWARD, NN_state, False)
                    Agent_Black.remember(NN_state, Action, DRAW_REWARD, next_state, True)
                    Agent_White.remember(WHITE_Last_State, WHITE_Last_Action, DRAW_REWARD, next_state, True)
                elif Who_is_playing == WHITE_PLAYER:
                    Agent_White.remember(NN_state, Action, DRAW_REWARD, next_state, True)
                    Agent_Black.remember(BLACK_Last_State, BLACK_Last_Action, DRAW_REWARD, next_state, True)
                else:
                    print ("PANIC!!!!!!!!!!!!! Unexpected player, MAGIC:OWURADHAGDAD8918")
                    sys.exit()
                    
            ## no winner, game is ongoing.         
            if not done and Who_is_playing == WHITE_PLAYER and BLACK_Last_Action>-1:
                Agent_Black.remember(BLACK_Last_State, BLACK_Last_Action, 0, next_state, False)
            if not done and Who_is_playing == BLACK_PLAYER and WHITE_Last_Action>-1:
                Agent_White.remember(WHITE_Last_State, WHITE_Last_Action, 0, next_state, False)

            if Who_is_playing == BLACK_PLAYER:
                BLACK_Last_State = copy.deepcopy(NN_state)
                BLACK_Last_Action = Action                               
            if Who_is_playing == WHITE_PLAYER:
                WHITE_Last_State = copy.deepcopy(NN_state)
                WHITE_Last_Action = Action
           
            NN_state = copy.deepcopy(next_state) # update NN_state to new state

            if len(Agent_Black.memory) > BLACK_batch_size and done == True:
                Agent_Black.replay(BLACK_batch_size)
            if len(Agent_White.memory) > WHITE_batch_size and done == True:
                Agent_White.replay(WHITE_batch_size)

            if done == True:   # This round is finished.

                if reward>0:
                    if Who_is_playing == BLACK_PLAYER:
                        Win_Black_Count = Win_Black_Count+1
                    elif Who_is_playing == WHITE_PLAYER:
                        Win_White_Count = Win_White_Count+1
                    else:
                        print ("PANIC, unknow know who is winner")
                        
                elif reward == 0:
                    Win_Draw_Count = Win_Draw_Count + 1
                                
                if (Win_Black_Count+Win_White_Count)>0:
                    print("Action={} BlackWinRate={} Black Wins={} White Wins={} Draw={} episode: {}/{}, used_step: {}, epsilon: {:.2}"
                           .format(Action, Win_Black_Count/(Win_Black_Count+Win_White_Count), Win_Black_Count,Win_White_Count, Win_Draw_Count, e, EPISODES, time, Agent_Black.epsilon))
                    if e%2000==0:
                        Agent_Black.save("5x5_get_4_14layer_BLACK.h5")
                        Agent_White.save("5x5_get_4_14layer_WHITE.h5")                       
                        #Agent_Black.save("3x3_get_3_14layer_BLACK.h5")
                        #Agent_White.save("3x3_get_3_14layer_WHITE.h5")                    
                        #Agent_Black.save("5x5_get_4_BLACK.h5")
                        #Agent_White.save("5x5_get_4_WHITE.h5")
                        #Agent_Black.save("8x8_get_4_BLACK.h5")
                        #Agent_White.save("8x8_get_4_WHITE.h5")
                        #!cp "8x8_get_4_BLACK.h5" "/content/drive/My Drive/AI_DATA/8x8_get_4_BLACK.h5"
                        #!cp "8x8_get_4_WHITE.h5" "/content/drive/My Drive/AI_DATA/8x8_get_4_WHITE.h5"

                break

            Who_is_playing = -1 * Who_is_playing


