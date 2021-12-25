import numpy as np
import pygame as pg

# Constants
WIDTH = 1600  # width of the environment (px)
HEIGHT = 1600  # height of the environment (px)
TS = 50  # delay in msec
NC = 40 * 40  # number of cells in the environment
SIZE = 40
GOAL = [560, 320]
BLOCKSIZE = 40
AGENT_START = [1000, 320]
AGENT_SIZE = BLOCKSIZE
TELEPORT = {
    (480,520): [800, 280],
    (840, 400) :[1560, 120],
    (600, 160): [680, 480],
    (800, 800) : [680,  360],
    (1200, 1440): [1560, 1240],
    (1280, 280): [720, 280],
    (1000, 1480):[1080, 1360],
    (1560, 1000):[1080, 520],
    (280, 880):[1160, 400],
    (0, 1400):[1560, 200],
    (1080, 1480):[1320, 1080],
    (1440, 880):[920, 1440],
    (40, 560):[720, 40],
    (120, 400):[200, 440]
}
PUNISHMENT = [[80, 80], [240, 240], [400, 120], [320, 240], [320, 1080], [1120, 40], [520, 640], [640, 1480],[1320, 920], [1000, 1000], [120, 1040],
              [720, 800],[880, 1560],[1560, 960],[480, 520],[1560, 800],[1440, 840],[680, 160],[1480, 640],[1520, 640],[1240, 560]]


# define colors
goal_color = pg.Color(0, 100, 0)
bad_color = pg.Color(100, 0, 0)
bg_color = pg.Color(0, 0, 0)
line_color = pg.Color(128, 128, 128)
agent_color = pg.Color(120, 120, 0)
WHITE = (200, 200, 200)

teleport_color = pg.Color('blue')
punishment_color = pg.Color('yellow')
wind_color = pg.Color('cyan')

class GridWorld:
    """The 2d Grid World Class"""

    def __init__(self, WIDTH, HEIGHT, NC, BLOCKSIZE, SIZE, GOAL):

        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT
        self.NC = NC
        self.BLOCKSIZE = BLOCKSIZE

        self.GOAL = GOAL
        self.SIZE = SIZE
        self.grid = np.zeros((self.SIZE, self.SIZE))
        self.TELEPORT = TELEPORT
        self.PUNISHMENT = PUNISHMENT
        self.grid = self.create_obstacles()


    def create_obstacles(self):
        #         Obstacle 1
        y = int(200 / self.SIZE)
        for i in range(int(320 / self.SIZE), int(800 / self.SIZE), 1):
            self.grid[i][y] = 1
            y += 1

        #      Obstacle 2
        x = int(320 / self.SIZE)
        for j in range(int(640 / self.SIZE), int(160 / self.SIZE), -1):
            self.grid[x][j] = 1
            x += 1

        z = int(800 / self.SIZE)
        for i in range(int(880 / self.SIZE), int(1400 / self.SIZE), 1):
            self.grid[i][z] = 1
            z += 1

        x = int(880 / self.SIZE)
        for j in range(int(1280 / self.SIZE), int(760 / self.SIZE), -1):
            self.grid[x][j] = 1
            x += 1

            # new obstacle

      
            


        #             Obstacle 3
        for stochastic in range(int(0), int(self.WIDTH / self.SIZE), 1):
            self.grid[stochastic][int(0)] = 5

        for stochastic in range(int(0), int(self.WIDTH / self.SIZE), 1):
            self.grid[stochastic][int(1560 / self.SIZE)] = 5

        for teleport in self.TELEPORT.keys():
            self.grid[int(teleport[0]/self.SIZE), int(teleport[1]/self.SIZE)] = 2

        for punish in self.PUNISHMENT:
            self.grid[int(punish[0]/self.SIZE), int(punish[1]/self.SIZE)] = 3

#             bad state 1
        for bad_state in range(int(1000/self.SIZE),  int(1080/self.SIZE), int(40/self.SIZE) ):
            self.grid[bad_state][int(120/self.SIZE)] = -1
            self.grid[bad_state][int(160/self.SIZE)] = -1
            self.grid[bad_state][int(200/self.SIZE)] = -1

        for bad_state in range(int(360/self.SIZE),  int(440/self.SIZE), int(40/self.SIZE) ):
            self.grid[bad_state][int(320/self.SIZE)] = -1
            self.grid[bad_state][int(360/self.SIZE)] = -1
            self.grid[bad_state][int(400/self.SIZE)] = -1

        for bad_state in range(int(0/self.SIZE),  int(80/self.SIZE), int(40/self.SIZE) ):

            self.grid[bad_state][int(240/self.SIZE)] = -1
            self.grid[bad_state][int(280/self.SIZE)] = -1
            self.grid[bad_state][int(320/self.SIZE)] = -1

        for bad_state in range(int(1200/self.SIZE),  int(1280/self.SIZE), int(40/self.SIZE) ):

            self.grid[bad_state][int(480/self.SIZE)] = -1
            self.grid[bad_state][int(520/self.SIZE)] = -1
            self.grid[bad_state][int(560/self.SIZE)] = -1

        for bad_state in range(int(880/self.SIZE),  int(960/self.SIZE), int(40/self.SIZE) ):

            self.grid[bad_state][int(880/self.SIZE)] = -1
            self.grid[bad_state][int(920/self.SIZE)] = -1
            self.grid[bad_state][int(960/self.SIZE)] = -1
# new bad state

        for bad_state in range(int(480/self.SIZE),  int(560/self.SIZE), int(40/self.SIZE) ):

            self.grid[bad_state][int(160/self.SIZE)] = -1
            self.grid[bad_state][int(200/self.SIZE)] = -1
            self.grid[bad_state][int(240/self.SIZE)] = -1






        return self.grid

    def draw_grid(self, scr):
        for x in range(self.WIDTH // self.BLOCKSIZE):
            for y in range(self.HEIGHT // self.BLOCKSIZE):
                rect = pg.Rect(x * self.BLOCKSIZE, y * self.BLOCKSIZE,
                               self.BLOCKSIZE, self.BLOCKSIZE)
                pg.draw.rect(scr, WHITE, rect, 2)

        # GOAL_state
        pg.draw.rect(scr, goal_color, (self.GOAL[0], self.GOAL[1], BLOCKSIZE, BLOCKSIZE))

        y = 200
        for i in range(320, 800, 40):
            pg.draw.rect(scr, WHITE, pg.Rect((i, y), (BLOCKSIZE, BLOCKSIZE)))
            y += 40

        x = 320
        for j in range(640, 160, -40):
            pg.draw.rect(scr, WHITE, pg.Rect((x, j), (BLOCKSIZE, BLOCKSIZE)))
            x += 40

        z = int(800)
        for i in range(int(880), int(1400), 40):
            pg.draw.rect(scr, WHITE, pg.Rect((i, z), (BLOCKSIZE, BLOCKSIZE)))
            z += 40

        x = int(880)
        for j in range(int(1280), int(760), -40):
            pg.draw.rect(scr, WHITE, pg.Rect((x, j), (BLOCKSIZE, BLOCKSIZE)))
            x += 40

            

#         Teleport State
        for teleport in self.TELEPORT.keys():
            pg.draw.rect(scr, teleport_color, pg.Rect(teleport[0], teleport[1], BLOCKSIZE, BLOCKSIZE))


        for punish in self.PUNISHMENT:
            pg.draw.rect(scr, punishment_color, pg.Rect((punish[0], punish[1]), (BLOCKSIZE, BLOCKSIZE)))


        pg.draw.rect(scr, bad_color, pg.Rect((1000,120), (80, 120)))

#     bad state 2
        pg.draw.rect(scr, bad_color, pg.Rect((360, 320), (80, 120)))

#     bad state 3
        pg.draw.rect(scr, bad_color, pg.Rect((0, 240), (80, 120)))


        pg.draw.rect(scr, bad_color, pg.Rect((1200,480 ), (80, 120)))

        pg.draw.rect(scr, bad_color, pg.Rect((880, 880 ), (80, 120)))

        # bad state new

        pg.draw.rect(scr, bad_color, pg.Rect((480, 160), (80, 120)))


#         stochastic
        pg.draw.rect(scr, wind_color, pg.Rect((0,0), (1600, 40)))
        pg.draw.rect(scr, wind_color, pg.Rect((0,1560), (1600, 40)))




class Agent:
    '''the agent class '''

    def __init__(self, scr, x, y, grid, agent_size, GOAL):

        self.scr = scr
        self.w = agent_size
        self.h = agent_size
        self.x = x
        self.y = y
        self.grid = grid
        self.GOAL = GOAL
        self.reward = []
        self.SIZE = SIZE
        self.TELEPORT = TELEPORT
        self.PUNISHMENT_COUNTER = 0

        self.my_rect = pg.Rect((self.x, self.y), (self.w, self.h))

    def show(self, color):
        self.my_rect = pg.Rect((self.x, self.y), (self.w, self.h))
        pg.draw.rect(self.scr, color, self.my_rect)

    def teleport(self):

        x = int(self.x/self.SIZE)
        y = int(self.y/self.SIZE)

        self.show(agent_color)
        pg.display.flip()
        pg.display.update()

        if self.grid[x,y] == 2:

            pg.time.wait(200)
            new_location = self.TELEPORT[self.x, self.y]
            self.x = new_location[0]
            self.y = new_location[1]
            print("You have been teleported!")

    def is_bad_state(self):

        x = int(self.x/self.SIZE)
        y = int(self.y/self.SIZE)

        if self.grid[x, y] == -1:
            return True

        return False

    def punishment(self):

        x = int(self.x/self.SIZE)
        y = int(self.y/self.SIZE)

        if self.grid[x,y] == 3:
            print('You have entered the Punishment state. Your next 5 actions will be penalized by 5 times!.')
            return True

        return False



    def is_stochastic(self, x, y):

        x = int(x / self.grid.shape[0])
        y = int(y / self.grid.shape[1])

        if self.grid[x, y] == 5:
            print("You have been pushed by the wind")
            return True
        else:
            return False

    def is_obstacle(self, x, y, action):
        #         Check if within limit

        if action == 0 and 0 <= x + 1 < self.grid.shape[0]:
            if self.grid[x + 1, y] == 1:
                return True
            else:
                return False
        elif action == 1 and 0 <= x - 1 < self.grid.shape[0]:
            if self.grid[x - 1, y] == 1:
                return True
            else:
                return False
        elif action == 2 and 0 <= y - 1 < self.grid.shape[1]:
            if self.grid[x, y - 1] == 1:
                return True
            else:
                return False
        elif action == 3 and 0 <= y + 1 < self.grid.shape[1]:
            if self.grid[x, y + 1] == 1:
                return True
            else:
                return False
        else:
            return False

    def is_move_valid(self, a, movement_dir):
        '''checking for the validity of moves'''
        x = int(self.x / self.grid.shape[0])
        y = int(self.y / self.grid.shape[1])

        obstacle = self.is_obstacle(x, y, movement_dir)

        if movement_dir == 0 or movement_dir == 1:
            if 0 <= self.x + a < WIDTH and obstacle == False:
                return True
            else:
                return False

        elif movement_dir == 2 or movement_dir == 3:
            if 0 <= self.y + a < HEIGHT and obstacle == False:
                return True
            else:
                return False

    def is_teleport(self):

        x = int(self.x/self.SIZE)
        y = int(self.y/self.SIZE)
    
        if self.grid[x,y] == 2:
            return True
    
        return False


    def keyboard_reward(self, movement_dir):

        '''reward function'''

        x_index = int(self.x/SIZE)
        y_index = int(self.y/SIZE)

        if self.x == self.GOAL[0] - self.SIZE and self.y == self.GOAL[1] and movement_dir == 0:
            rew =  100.0

        elif self.x == self.GOAL[0] + self.SIZE and self.y == self.GOAL[1] and movement_dir == 1:
            rew = 100.0

        elif self.x == self.GOAL[0] and self.y == self.GOAL[1] + self.SIZE and movement_dir == 2:
            rew = 100.0

        elif self.x == self.GOAL[0] and self.y == self.GOAL[1] - self.SIZE and movement_dir == 3:
            rew = 100.0

        elif self.is_obstacle(x_index, y_index, movement_dir):
            rew = -0.25

        else:
            rew = -0.1

        return rew

    def move(self, a, movement_dir):
        '''move the agent'''
        if movement_dir == 0 or movement_dir == 1:
            rew = self.keyboard_reward(movement_dir)

            if self.is_move_valid(a, movement_dir):

                pg.time.wait(TS)
                self.show(bg_color)

                self.x += a

                pg.time.wait(TS)
                self.show(bg_color)


                punishment_flag = self.punishment()

                if ((punishment_flag) or (self.PUNISHMENT_COUNTER>0 and self.PUNISHMENT_COUNTER<5) and self.is_goal()==False):
                    rew *= 5
                    self.PUNISHMENT_COUNTER += 1

                if self.PUNISHMENT_COUNTER == 5:
                    self.PUNISHMENT_COUNTER = 0



                if (((punishment_flag) or (self.PUNISHMENT_COUNTER>0 and self.PUNISHMENT_COUNTER<5)) and self.is_goal()==True):
                    rew = 1

                if self.is_bad_state():
                    rew =  -10

                if self.is_teleport():
                    rew = -0.25

                self.teleport()
                self.show(agent_color)
                print("State:", [int(self.x/self.SIZE), int(self.y/self.SIZE)], "--- Reward Received:", rew)


            else:
                pg.time.wait(TS)
                self.show(bg_color)

                punishment_flag = self.punishment()

                if ((punishment_flag) or (self.PUNISHMENT_COUNTER>0 and self.PUNISHMENT_COUNTER<5) and self.is_goal()==False):
                    rew *= 5
                    self.PUNISHMENT_COUNTER += 1

                if self.PUNISHMENT_COUNTER == 5:
                    self.PUNISHMENT_COUNTER = 0


                if (((punishment_flag) or (self.PUNISHMENT_COUNTER>0 and self.PUNISHMENT_COUNTER<5)) and self.is_goal()==True):
                    rew = 1

                if self.is_bad_state():
                    rew =  -10

                if self.is_teleport():
                    rew = -0.25

                self.teleport()
                self.show(agent_color)
                print("State:", [int(self.x/self.SIZE), int(self.y/self.SIZE)], "--- Reward Received:", rew)


            self.reward.append(rew)


        elif movement_dir == 2 or movement_dir == 3:

            rew = self.keyboard_reward(movement_dir)

            if self.is_move_valid(a, movement_dir):

                pg.time.wait(TS)
                self.show(bg_color)

                self.y += a

                pg.time.wait(TS)
                self.show(bg_color)


                punishment_flag = self.punishment()

                if ((punishment_flag) or (self.PUNISHMENT_COUNTER>0 and self.PUNISHMENT_COUNTER<5) and self.is_goal()==False):
                    rew *= 5
                    self.PUNISHMENT_COUNTER += 1

                if self.PUNISHMENT_COUNTER == 5:
                    self.PUNISHMENT_COUNTER = 0



                if (((punishment_flag) or (self.PUNISHMENT_COUNTER>0 and self.PUNISHMENT_COUNTER<5)) and self.is_goal()==True):
                    rew = 1

                if self.is_stochastic(self.x, self.y):
                    self.y -= 4 * a

                if self.is_bad_state():
                    rew =  -10

                if self.is_teleport():
                    rew = -0.25

                self.teleport()
                self.show(agent_color)
                print("State:", [int(self.x/self.SIZE), int(self.y/self.SIZE)], "--- Reward Received:", rew)


            else:

                pg.time.wait(TS)
                self.show(bg_color)


                punishment_flag = self.punishment()

                if ((punishment_flag) or (self.PUNISHMENT_COUNTER>0 and self.PUNISHMENT_COUNTER<5) and self.is_goal()==False):
                    rew *= 5
                    self.PUNISHMENT_COUNTER += 1

                if self.PUNISHMENT_COUNTER == 5:
                    self.PUNISHMENT_COUNTER = 0



                if (((punishment_flag) or (self.PUNISHMENT_COUNTER>0 and self.PUNISHMENT_COUNTER<5)) and self.is_goal()==True):
                    rew = 1

                if self.is_stochastic(self.x, self.y):
                    self.y -= 4 * a

                if self.is_bad_state():
                    rew =  -10

                if self.is_teleport():
                    rew = -0.25

                self.teleport()
                self.show(agent_color)
                print("State:", [int(self.x/self.SIZE), int(self.y/self.SIZE)], "--- Reward Received:", rew)



            self.reward.append(rew)
            

    def is_goal(self):
        if self.x == self.GOAL[0] and self.y == self.GOAL[1]:
            return True
        else:
            return False


def main():
    pg.init()  # initialize pygame
    screen = pg.display.set_mode((WIDTH, HEIGHT))  # set up the screen
    pg.display.set_caption("Grid World 40 x 40 - Mamoon Habib")  # add a caption
    bg = pg.Surface(screen.get_size())  # get a background surface
    bg = bg.convert()
    bg.fill(bg_color)
    screen.blit(bg, (0, 0))
    clock = pg.time.Clock()

    grid_world = GridWorld(WIDTH, HEIGHT, NC, BLOCKSIZE, SIZE, GOAL)
    agent = Agent(screen, AGENT_START[0], AGENT_START[1], grid_world.grid, BLOCKSIZE, GOAL)  # instantiate an agent
    agent.show(agent_color)

    pg.display.flip()
    run = True

    while run:
        clock.tick(60)
        for event in pg.event.get():
            if event.type == pg.QUIT:
                run = False
            elif event.type== pg.KEYDOWN and event.key == pg.K_ESCAPE:
                run = False
            elif event.type == pg.KEYDOWN and event.key == pg.K_RIGHT:
                agent.move(SIZE, 0)
            elif event.type == pg.KEYDOWN and event.key == pg.K_LEFT:
                agent.move(-SIZE, 1)
            elif event.type == pg.KEYDOWN and event.key == pg.K_UP:
                agent.move(-SIZE, 2)
            elif event.type == pg.KEYDOWN and event.key == pg.K_DOWN:
                agent.move(SIZE, 3)

        screen.blit(bg, (0, 0))
        grid_world.draw_grid(screen)
        agent.show(agent_color)
        pg.display.flip()
        pg.display.update()


        if agent.is_goal():
            run = False
            total_reward = round(sum(agent.reward), 3)
            print()
            print("Congratulations! You have reached the goal state. You receive 100 points for entering the Goal State")
            print("Total Reward Collected:", total_reward)

        if agent.is_bad_state():
            run = False
#             agent.reward = []
            # agent.reward.append(-10)
            total_reward = round(sum(agent.reward), 3)
            print()
            print("Game Over! You have entered the bad state. You receive a -10 reward for entering the bad state.")
            print("Total Reward Collected:", total_reward)


    pg.display.update()
    pg.quit()


if __name__ == "__main__":
    main()