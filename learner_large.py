'''
This is the gridworld.py file that implements an agent that finds the goal state using a RL algorithm on a
a 2D 40 x 40 gridworld game and is part of the mid-term project in
the COMP4600/5500-Reinforcement Learning course - Fall 2021
'''

import numpy as np
import matplotlib.pyplot as plt

from gridworld_large import *

TS = 1
states = {}

for i in range(SIZE):
    state_i = SIZE * i
    for j in range(SIZE):
        state_j = SIZE * j
        states[i, j] = [state_i, state_j]

# START = states[np.random.randint(SIZE), np.random.randint(SIZE)]  # start state
START = [1000, 320]

GOAL = [560, 320]  # goal state
# Actions
RIGHT = 0
LEFT = 1
UP = 2
DOWN = 3

ACTIONS = [LEFT, RIGHT, UP, DOWN]
nA = len(ACTIONS)
nS = 400  # number of states


def learner_grid_obstacles(grid):
    y = int(200 / SIZE)
    for i in range(int(320 / SIZE), int(800 / SIZE), 1):
        grid[i][y] = 1
        y += 1

    #      Obstacle 2
    x = int(320 / SIZE)
    for j in range(int(640 / SIZE), int(160 / SIZE), -1):
        grid[x][j] = 1
        x += 1

    z = int(800 / SIZE)
    for i in range(int(880 / SIZE), int(1400 / SIZE), 1):
        grid[i][z] = 1
        z += 1

    x = int(880 / SIZE)
    for j in range(int(1280 / SIZE), int(760 / SIZE), -1):
        grid[x][j] = 1
        x += 1


    # new obstacle
    

    for stochastic in range(int(0), int(WIDTH / SIZE), 1):
        grid[stochastic][int(1560 / SIZE)] = 5

    #             Obstacle 4
    for stochastic in range(int(0), int(WIDTH / SIZE), 1):
        grid[stochastic][int(0)] = 5

    for teleport in TELEPORT.keys():
        grid[int(teleport[0] / SIZE), int(teleport[1] / SIZE)] = 2

    for punish in PUNISHMENT:
        grid[int(punish[0] / SIZE), int(punish[1] / SIZE)] = 3

    #             bad state 1
    for bad_state in range(int(1000 / SIZE), int(1080 / SIZE), int(40 / SIZE)):
        grid[bad_state][int(120 / SIZE)] = -1
        grid[bad_state][int(160 / SIZE)] = -1
        grid[bad_state][int(200 / SIZE)] = -1

    for bad_state in range(int(360 / SIZE), int(440 / SIZE), int(40 / SIZE)):
        grid[bad_state][int(320 / SIZE)] = -1
        grid[bad_state][int(360 / SIZE)] = -1
        grid[bad_state][int(400 / SIZE)] = -1

    for bad_state in range(int(0 / SIZE), int(80 / SIZE), int(40 / SIZE)):
        grid[bad_state][int(240 / SIZE)] = -1
        grid[bad_state][int(280 / SIZE)] = -1
        grid[bad_state][int(320 / SIZE)] = -1

    for bad_state in range(int(1200 / SIZE), int(1280 / SIZE), int(40 / SIZE)):
        grid[bad_state][int(480 / SIZE)] = -1
        grid[bad_state][int(520 / SIZE)] = -1
        grid[bad_state][int(560 / SIZE)] = -1

    for bad_state in range(int(880 / SIZE), int(960 / SIZE), int(40 / SIZE)):
        grid[bad_state][int(880 / SIZE)] = -1
        grid[bad_state][int(920 / SIZE)] = -1
        grid[bad_state][int(960 / SIZE)] = -1

    # new bad state

    for bad_state in range(int(480/SIZE),  int(560/SIZE), int(40/SIZE)):
        grid[bad_state][int(160/SIZE)] = -1
        grid[bad_state][int(200/SIZE)] = -1
        grid[bad_state][int(240/SIZE)] = -1

    return grid


learner_grid = np.zeros((SIZE, SIZE))
learner_grid = learner_grid_obstacles(learner_grid)


def is_learner_stochastic(x, y, learner_grid):
    x = int(x / SIZE)
    y = int(y / SIZE)

    if 0 <= x < 40 and 0 <= y < 40 and learner_grid[x, y] == 5:
        # print("You have been pushed by the wind")
        return True
    else:
        return False


def is_learner_obstacle(x, y, action, learner_grid):
    #         Check if within limit
    if action == 0 and 0 <= x + 1 < learner_grid.shape[0]:
        if learner_grid[x + 1, y] == 1:
            return True
        else:
            return False
    elif action == 1 and 0 <= x - 1 < learner_grid.shape[0]:
        if learner_grid[x - 1, y] == 1:
            return True
        else:
            return False
    elif action == 2 and 0 <= y - 1 < learner_grid.shape[1]:
        if learner_grid[x, y - 1] == 1:
            return True
        else:
            return False
    elif action == 3 and 0 <= y + 1 < learner_grid.shape[1]:
        if learner_grid[x, y + 1] == 1:
            return True
        else:
            return False
    else:
        return False


def learner_teleport(x_main, y_main):
    new_location = TELEPORT[x_main, y_main]
    x = new_location[0]
    y = new_location[1]
    # print("You have been teleported!")

    return x, y


def is_learner_teleport(x_main, y_main, learner_grid):
    x = int(x_main / SIZE)
    y = int(y_main / SIZE)

    if learner_grid[x, y] == 2:
        return True

    return False


def is_learner_bad_state(x_main, y_main, learner_grid):
    x = int(x_main / SIZE)
    y = int(y_main / SIZE)

    if learner_grid[x, y] == -1:
        return True

    return False


def is_learner_punishment(x_main, y_main, learner_grid):
    x = int(x_main / SIZE)
    y = int(y_main / SIZE)

    if learner_grid[x, y] == 3:
        # print('You have entered the Punishment state. Your next 5 actions will be penalized by 5 times.')
        return True

    return False


def transition(s, a):
    '''transition function'''
    x = s[0]
    y = s[1]
    #     UP = 2

    x_index = int(x / SIZE)
    y_index = int(y / SIZE)

    learner_obstacle = is_learner_obstacle(x_index, y_index, a, learner_grid)

    if a == UP:
        stochastic = is_learner_stochastic(x, y - 40, learner_grid)
    elif a == DOWN:
        stochastic = is_learner_stochastic(x, y + 40, learner_grid)

    if a == UP:
        if 0 <= y - SIZE and y - SIZE < HEIGHT and learner_obstacle == False:
            if stochastic == True:
                next_state = [x, y + 3 * SIZE]
            else:
                next_state = [x, y - SIZE]
        else:
            next_state = [x, y]

    #         Down  = 3
    elif a == DOWN:

        if 0 <= y + SIZE and y + SIZE < HEIGHT and learner_obstacle == False:
            if stochastic == True:
                next_state = [x, y - 3 * SIZE]
            else:
                next_state = [x, y + SIZE]
        else:
            next_state = [x, y]

    #         left = 1
    elif a == LEFT:
        if 0 <= x - SIZE and x - SIZE < WIDTH and learner_obstacle == False:
            next_state = [x - SIZE, y]
        else:
            next_state = [x, y]

    #         right = 0
    elif a == RIGHT:
        if 0 <= x + SIZE and x + SIZE < WIDTH and learner_obstacle == False:
            next_state = [x + SIZE, y]
        else:
            next_state = [x, y]
    else:
        next_state = [x, y]
    return next_state


def epsilon_greedy_action(Q, s1, s2, actions, epsilon = 0.05):
     
    s1 = int(s1 / SIZE)
    s2 = int(s2 / SIZE)
    
    if np.random.random() < epsilon:
        action = np.random.choice(actions)
        
    else:
        p = np.where(Q[s1,s2] == Q[s1,s2].max())
        p=p[0]
        if len(p)>1:
            action =  np.random.choice(p)
        else:
            action =  np.argmax(Q[s1, s2])
    return action


def reward_function(s, a):
    '''reward function'''
    x = s[0]
    y = s[1]
    
    x_index = int(s[0]/SIZE)
    y_index = int(s[1]/SIZE)
       
    if x == GOAL[0] - SIZE and y == GOAL[1] and a == RIGHT:
        return 100.0

    elif x == GOAL[0] + SIZE and y == GOAL[1] and a == LEFT:
        return 100.0

    elif x == GOAL[0] and y == GOAL[1] - SIZE and a == DOWN:
        return 100.0

    elif x == GOAL[0] and y == GOAL[1] + SIZE and a == UP:
        return 100.0
    
    elif is_learner_obstacle(x_index, y_index, a, learner_grid):
        return -0.25

    else:
        return -0.1

def evaluate_policy(Q, initial_state):

    states_tracker = []
    cumulative_reward = {}
    a = initial_state[0]
    b = initial_state[1]

    a_size = int(a/SIZE)
    b_size = int(b/SIZE)

    if (learner_grid[a_size, b_size] != 0):
        raise ValueError("The given initial state is part of the obstacle/challenge state. Try another state")

    
    for episode in range(1):
            
        # while True:
        #     a, b = generate_state()
        #     if ((a,b) not in states_tracker):
        #         states_tracker.append((a,b))
        #         break

        START = [a,b]
        start_s1 = START[0]
        start_s2 = START[1]

        current_s1 = start_s1
        current_s2 = start_s2
        states = [[current_s1, current_s2]]
        print("-------------------------------------------------------------")

        print("Starting Game at State:", [int(a/SIZE), int(b/SIZE)])
        counter=0
        time=0

        T = []
        R = []
        actions_taken = []
               
        while True:
            
            s1 = int(current_s1 / SIZE)
            s2 = int(current_s2 / SIZE)
            
            p = np.where(Q[s1,s2] == Q[s1,s2].max())
            p=p[0]
            
            if len(p)>1:
                action =  np.random.choice(p)
            else:
                action =  np.argmax(Q[s1, s2])
            
            re = reward_function([current_s1, current_s2], action)
            next_state= transition([current_s1, current_s2], action)
            

            next_s1 = next_state[0]
            next_s2 = next_state[1]
                            
             
            if is_learner_teleport(next_s1, next_s2, learner_grid):
                next_s1, next_s2 = learner_teleport(next_s1, next_s2)
                next_s1_index = int(next_s1/SIZE)
                next_s2_index = int(next_s2/SIZE)
                re = -0.25


            if (is_learner_punishment(next_s1, next_s2, learner_grid) or (counter>0 and counter<5)) and next_state!=GOAL:
                re *= 5
                counter+=1

            if counter==5:
                counter = 0

            if (is_learner_punishment(next_s1, next_s2, learner_grid) or (counter>0 and counter<5)) and next_state==GOAL:
                re = 1
                
                
            if is_learner_bad_state(next_s1, next_s2, learner_grid):
                re = -10
            
            # print("State", [int(current_s1/SIZE), int(current_s2/SIZE)], "--- Reward", re);
            
            actions_taken.append(action)
            R.append(re)
            T.append([next_s1, next_s2, action])
            states.append([current_s1, current_s2])
            
#             print("action:",action, "Reward:", re, "Q:", Q[current_s1_index, current_s2_index], current_s1_index, current_s2_index)
            
            if is_learner_bad_state(next_s1, next_s2, learner_grid):
                # print("BAD STATE!", next_s1, next_s2)
                break
            
            if next_s1 == GOAL[0] and next_s2 == GOAL[1]:
                # print("GOAL REACHED!", next_s1, next_s2)
                break
            
            current_s1 = next_s1
            current_s2 = next_s2
            
            
#             current_action = action
            time +=1 
        cumulative_reward[episode] = sum(R)
        animate(START, T, R, actions_taken)
        
    # return cumulative_reward

def generate_state():
    while True:
        a = np.random.randint(0,SIZE-1)
        b = np.random.randint(0,SIZE-1)

        if (learner_grid[a, b] == 0):
            if (a!=14 and b!=8):
                return a*SIZE, b*SIZE

def rl_agent(alpha=0.5, gamma=1, runs=50, episodes = 3000):
    
    
    Total_Reward = {}
    Total_Steps = {}
    
    for run in range(runs):
        
        print("Starting Run", run+1)
        Q = np.zeros((SIZE, SIZE, 4))
        
   
        for episode in range(episodes):

            T = []
            R = []
            actions_taken = []

            a, b = generate_state()
            START = [a,b]
            start_s1 = START[0]
            start_s2 = START[1]

    #         initial_action = epsilon_greedy_action(Q, start_s1, start_s2, ACTIONS, epsilon = 0.1)
    #         action = initial_action

            current_s1 = start_s1
            current_s2 = start_s2
            states = []
            print("Starting Game for Epsiode", episode+1, "at State:", [int(current_s1/SIZE), int(current_s2/SIZE)])

    #         actions_taken.append(action)

            counter = 0
            time = 0

            while True:

                action = epsilon_greedy_action(Q, current_s1, current_s2, ACTIONS, epsilon = 0.05)
                re = reward_function([current_s1, current_s2], action)
                next_state= transition([current_s1, current_s2], action)


                next_s1 = next_state[0]
                next_s2 = next_state[1]

                current_s1_index = int(current_s1/SIZE)
                current_s2_index = int(current_s2/SIZE)
                next_s1_index = int(next_s1/SIZE)
                next_s2_index = int(next_s2/SIZE)

                if is_learner_teleport(next_s1, next_s2, learner_grid):
                    next_s1, next_s2 = learner_teleport(next_s1, next_s2) 
                    next_s1_index = int(next_s1/SIZE)
                    next_s2_index = int(next_s2/SIZE)
                    re = -0.25


                if (is_learner_punishment(next_s1, next_s2, learner_grid) or (counter>0 and counter<5)) and next_state!=GOAL:
                    re *= 5
                    counter+=1

                if counter==5:
                    counter = 0

                if (is_learner_punishment(next_s1, next_s2, learner_grid) or (counter>0 and counter<5)) and next_state==GOAL:
                    re = 1


                if is_learner_bad_state(next_s1, next_s2, learner_grid):
                    re = -10

    #             print("Current State:", [current_s1, current_s2], "Action Taken:", action, "Reward Received:", re, "Next State", [next_s1, next_s2])
                actions_taken.append(action)
                R.append(re)
                T.append([next_s1, next_s2, action])
                states.append([current_s1, current_s2])

                max_q = np.max(Q[next_s1_index, next_s2_index])


                Q[current_s1_index, current_s2_index, action] +=  alpha * (re + gamma * max_q - Q[current_s1_index, current_s2_index, action])
    #             print("action:",action, "Reward:", re, "Q:", Q[current_s1_index, current_s2_index], current_s1_index, current_s2_index)

                if is_learner_bad_state(next_s1, next_s2, learner_grid):
                    print("BAD STATE!", next_s1, next_s2)
                    break

                if next_s1 == GOAL[0] and next_s2 == GOAL[1]:
                    print("GOAL REACHED!", next_s1, next_s2)
                    break

                current_s1 = next_s1
                current_s2 = next_s2

                time +=1
                
            if episode not in Total_Reward:
                Total_Reward[episode] = []
                Total_Reward[episode].append(sum(R))
            else:
                Total_Reward[episode].append(sum(R))
                
            if episode not in Total_Steps:
                Total_Steps[episode] = []
                Total_Steps[episode].append(time)
            else:
                Total_Steps[episode].append(time)
           
    return Q, T, R, actions_taken, states, START, Total_Reward, Total_Steps

def random_agent():
    '''this is a random walker
    your smart algorithm will replace this'''
    s = START
    T = [s]
    R = []
    actions_taken = []
    counter = 0
    while s != GOAL:
        a = np.random.choice(ACTIONS)
        sp = transition(s, a)
        re = reward_function(s, a)

        if is_learner_teleport(sp[0], sp[1], learner_grid):
            sp[0], sp[1] = learner_teleport(sp[0], sp[1])
            re = -0.25

        if (is_learner_punishment(sp[0], sp[1], learner_grid) or (counter > 0 and counter < 5)) and sp != GOAL:
            re *= 5

            counter += 1

        if counter == 5:
            counter = 0

        if (is_learner_punishment(sp[0], sp[1], learner_grid) or (counter > 0 and counter < 5)) and sp == GOAL:
            re = 1
            
        if is_learner_bad_state(sp[0], sp[1], learner_grid):
            re = -10

        R.append(re)

        T.append([s, a])
        actions_taken.append(a)

        if is_learner_bad_state(sp[0], sp[1], learner_grid):
            print('Bad State', sp[0], sp[1])
            break

        s = sp
    return T, R, actions_taken




def animate(START, Trajectory, Reward, Actions_taken):
    '''
    a function that can pass information to the
    pygame gridworld environment for visualizing
    agent's moves
    '''
    pg.init()  # initialize pygame
    screen = pg.display.set_mode((WIDTH , HEIGHT ))  # set up the screen
    pg.display.set_caption("Grid World 40 x 40 - Mamoon Habib")  # add a caption
    bg = pg.Surface(screen.get_size())  # get a background surface
    bg = bg.convert()
    bg.fill(bg_color)
    screen.blit(bg, (0, 0))
    clock = pg.time.Clock()

    x = START[0]
    y = START[1]

    grid_world = GridWorld(WIDTH, HEIGHT, NC, BLOCKSIZE, SIZE, GOAL)
    agent = Agent(screen, x, y, grid_world.grid, BLOCKSIZE, GOAL)
    agent.show(agent_color)

    pg.display.flip()
    run = True

    counter = 0
    checking = []

    while run:
        clock.tick(60)
        for event in pg.event.get():
            if event.type == pg.QUIT:
                run = False
            elif event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                run = False
            
        for act in Actions_taken:
            # print("action:", act)
            screen.blit(bg, (0, 0))
            grid_world.draw_grid(screen)

            agent.show(agent_color)
            #             print("Action", act)
            if act == 0 or act == 3:
                checking.append([agent.x, agent.y, act])
                agent.move(SIZE, act)
                counter += 1
            elif act == 1 or act == 2:
                checking.append([agent.x, agent.y, act])
                agent.move(-SIZE, act)
                counter += 1

            pg.display.flip()
            pg.display.update()
            pg.time.wait(200)
            # print(agent.x, agent.y)

            if agent.is_goal():
                run = False
                total_reward = round(sum(agent.reward), 3)
                print()
                print(
                    "Congratulations! You have reached the goal state. You receive a reward of 100 for reaching the Goal State")
                print("Total Reward Collected:", total_reward)
                print("Total Steps Taken:", len(Trajectory))
                print("-------------------------------------------------------------")
                print()

            if agent.is_bad_state():
                run = False
                #                 agent.reward = []
#                 agent.reward.append(-10)
                total_reward = round(sum(agent.reward), 3)
                print()
                print(
                    "Game Over! You have entered the bad state. You receive a reward of -10 for entering the bad state.")
                print("Total Reward Collected:", total_reward)
                print("Total Steps Taken:", len(Trajectory))
                print("-------------------------------------------------------------")
                print()

        screen.blit(bg, (0, 0))
        grid_world.draw_grid(screen)
        agent.show(agent_color)
        pg.display.flip()
        pg.display.update()
        pg.quit()

def plot_results(total_reward, total_steps, runs):
    fig1=plt.figure(figsize=(10,5)).add_subplot()
    fig1.set_xlabel('Episodes')
    fig1.set_ylabel('Reward-per-episode averaged over 50 runs')
    fig1.set_title("Number of episodes vs. Steps-per-episode")

    fig2= plt.figure(figsize=(10,5)).add_subplot()
    fig2.set_xlabel('Episodes')
    fig2.set_ylabel('Steps-per-episode averaged over 50 runs')
    fig2.set_title("Number of episodes vs. Steps-per-episode")

    avg_reward = {}
    for i in total_reward:
        avg_reward[i] = sum(total_reward[i])/runs    
    lists_reward = sorted(avg_reward.items())
    x_reward, y_reward = zip(*lists_reward)
    fig1.plot(x_reward, y_reward)

    avg_steps = {}
    for i in total_steps:
        avg_steps[i] = sum(total_steps[i])/runs    
    lists_steps = sorted(avg_steps.items())
    x_steps, y_steps = zip(*lists_steps)
    fig2.plot(x_steps, y_steps)



if __name__ == "__main__":
    # Trajectory, Reward, Actions_taken = random_agent()
    
# Below runs the code that an RL agent will use Q learning to learn the environment and visualize the last trajectory
    Q, Trajectory, Reward, Actions_taken, States, Start, Total_Reward, Total_Steps = rl_agent(runs=1, episodes=10000)
    animate(Start, Trajectory, Reward, Actions_taken)

    # animate(START, Trajectory, Reward, Actions_taken)
    # plot_results(Total_Reward, Total_Steps, 1)
    # evaluate_policy(Q)

#     pg.quit()
