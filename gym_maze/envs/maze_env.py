import numpy as np
import gymnasium as gym
from gymnasium import error, spaces, utils
from gymnasium.utils import seeding
from gym_maze.envs.maze_view_2d import MazeView2D

class MazeEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
    }

    ACTION = ["N", "S", "E", "W"]

    def __init__(self, maze_file=None, maze_size=None, mode=None, enable_render=True, seed=None, target=0):
        self.viewer = None
        self.enable_render = enable_render

        self.seed = seed
        self.maze_seed = seed  # Store the initial seed for maze generation
        self.target = target

        if maze_file:
            self.maze_view = MazeView2D(maze_name="OpenAI Gym - Maze (%s)" % maze_file,
                                        maze_file_path=maze_file,
                                        screen_size=(256, 256), #changed from 640
                                        enable_render=enable_render,
                                        seed=self.maze_seed)  # Use the stored maze_seed for maze generation
        elif maze_size:
            if mode == "plus":
                has_loops = True
                num_portals = int(round(min(maze_size) / 3))
            else:
                has_loops = False
                num_portals = 0

            self.maze_view = MazeView2D(maze_name="OpenAI Gym - Maze (%d x %d)" % maze_size,
                                        maze_size=maze_size, screen_size=(256, 256), #changed from 640
                                        has_loops=has_loops, num_portals=num_portals,
                                        enable_render=enable_render,
                                        seed=self.maze_seed)  # Use the stored maze_seed for maze generation
        else:
            raise AttributeError("One must supply either a maze_file path (str) or the maze_size (tuple of length 2)")

        self.maze_size = self.maze_view.maze_size

        # forward or backward in each dimension
        self.action_space = spaces.Discrete(2 * len(self.maze_size))

        # observation is the x, y coordinate of the grid
        low = np.zeros(len(self.maze_size), dtype=int)
        high = np.array(self.maze_size, dtype=int) - np.ones(len(self.maze_size), dtype=int)
        # self.observation_space = spaces.Box(low, high, dtype=np.int64) uint8?
        self.observation_space = spaces.Box(low=0, high=255, shape=(256, 256, 3), dtype=np.uint8)
        
        # initial condition
        self.state = None
        self.steps_beyond_done = None

        # Simulation related variables.
        self.reset(seed)

        # Just need to initialize the relevant attributes
        self.configure()

    def __del__(self):
        if self.enable_render is True:
            self.maze_view.quit_game()

    def configure(self, display=None):
        self.display = display

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        if isinstance(action, int):
            self.maze_view.move_robot(self.ACTION[action])
        else:
            self.maze_view.move_robot(action)

        # TARGET
        if self.target==0:
            if np.array_equal(self.maze_view.robot, self.maze_view.goal):
                reward = 1
                done = True
            else:
                reward = -0.1 / (self.maze_size[0] * self.maze_size[1])
                done = False
        if self.target==1:
            if np.array_equal(self.maze_view.robot, self.maze_view.object):
                reward = 1
                done = True
            else:
                reward = -0.1 / (self.maze_size[0] * self.maze_size[1])
                done = False

        self.state = self.maze_view.robot

        info = {}

        # NOTE: Altered for CNN
        observation = self.render(mode="rgb_array")
        return observation, reward, done, False, info
        # return self.state, reward, done, False, info  # NOTE: Changed original code here to account for new step API, adding False parameter

    def reset(self, seed=None, **kwargs): # Added kwargs
        if seed is not None:
            np.random.seed()

        self.maze_view.regenerate_maze()  # Reset the maze's structure
        self.maze_view.reset_robot()
        self.state = np.zeros(2)
        self.steps_beyond_done = None
        self.done = False

        # NOTE: Altered for CNN
        observation = self.render(mode="rgb_array").astype(np.uint8)
        # print(f'Reset Observation shape: {observation.shape}, Data type: {observation.dtype}')
        # import matplotlib.pyplot as plt

        # # Assuming 'observation' is your RGB image
        # plt.imshow(observation)
        # plt.show()

        info = {}
        return (observation, info)
        # return self.state

    def is_game_over(self):
        return self.maze_view.game_over

    def render(self, mode="human", close=False):
        if close:
            self.maze_view.quit_game()

        return self.maze_view.update(mode)


class MazeEnvSample5x5(MazeEnv):

    def __init__(self, enable_render=True):
        super(MazeEnvSample5x5, self).__init__(maze_file="maze2d_5x5.npy", enable_render=enable_render)


class MazeEnvRandom5x5(MazeEnv):

    def __init__(self, enable_render=True, seed=None, target=0):
        super(MazeEnvRandom5x5, self).__init__(maze_size=(10, 10), enable_render=enable_render, seed=seed, target=target)


class MazeEnvSample10x10(MazeEnv):

    def __init__(self, enable_render=True):
        super(MazeEnvSample10x10, self).__init__(maze_file="maze2d_10x10.npy", enable_render=enable_render)


class MazeEnvRandom10x10(MazeEnv):

    def __init__(self, enable_render=True, seed=None, target=0):
        super(MazeEnvRandom10x10, self).__init__(maze_size=(8, 8), enable_render=enable_render, seed=seed, target=target)


class MazeEnvSample3x3(MazeEnv):

    def __init__(self, enable_render=True):
        super(MazeEnvSample3x3, self).__init__(maze_file="maze2d_3x3.npy", enable_render=enable_render)


class MazeEnvRandom3x3(MazeEnv):

    def __init__(self, enable_render=True):
        super(MazeEnvRandom3x3, self).__init__(maze_size=(3, 3), enable_render=enable_render)


class MazeEnvSample100x100(MazeEnv):

    def __init__(self, enable_render=True):
        super(MazeEnvSample100x100, self).__init__(maze_file="maze2d_100x100.npy", enable_render=enable_render)


class MazeEnvRandom100x100(MazeEnv):

    def __init__(self, enable_render=True):
        super(MazeEnvRandom100x100, self).__init__(maze_size=(100, 100), enable_render=enable_render)


class MazeEnvRandom10x10Plus(MazeEnv):

    def __init__(self, enable_render=True):
        super(MazeEnvRandom10x10Plus, self).__init__(maze_size=(10, 10), mode="plus", enable_render=enable_render)


class MazeEnvRandom20x20Plus(MazeEnv):

    def __init__(self, enable_render=True):
        super(MazeEnvRandom20x20Plus, self).__init__(maze_size=(20, 20), mode="plus", enable_render=enable_render)


class MazeEnvRandom30x30Plus(MazeEnv):
    def __init__(self, enable_render=True):
        super(MazeEnvRandom30x30Plus, self).__init__(maze_size=(30, 30), mode="plus", enable_render=enable_render)
