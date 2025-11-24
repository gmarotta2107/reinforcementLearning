import sys
import argparse
import logging
import os
import random
import time
import math
import numpy as np
import torch
from client import *
from torch.utils.tensorboard import SummaryWriter
from utils.nets import *
from ddpg import DDPG
from utils.noise import OrnsteinUhlenbeckActionNoise
from utils.replay_memory import ReplayMemory, Transition

DEFAULT_PORT = 9543
logger = logging.getLogger('train')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Using {}".format(device))

# Define the joint ranges
joint_ranges = [
    (-3 * math.pi / 4, 3 * math.pi / 4)  # Joint 9: Rotation
]

def get_input_actor(state):
    paddle_position = state[11:14]  # Paddle center position (x, y, z)
    paddle_normal = state[14:17]  # Paddle normal vector (x, y, z)
    ball_position = state[17:20]  # Current ball position (x, y, z)
    ball_velocity = state[20:23]  # Current ball velocity (x, y, z)
    paddle_pitch_joint = [state[9]]  # Convert to list to concatenate

    relevant_state = np.concatenate((paddle_position,
                                     paddle_normal,
                                     ball_position,
                                     ball_velocity,
                                     paddle_pitch_joint))

    return relevant_state

def where_ball_touched(state):
    opponent_table_length = 1.2
    my_table_length = 1.2
    opponent_table_width = 1.4

    bx, by, bz = state[17:20]
    vx, vy, vz = state[20:23]
    g = 9.81  # Acceleration due to gravity
    d = 0.05  # Time step for simulation
    max_iterations = 1000  # Set a maximum number of iterations to prevent infinite loops

    iterations = 0

    while bz > 0 and iterations <= max_iterations:
        bx += vx * d
        by += vy * d
        bz += vz * d
        vz -= g * d
        iterations += 1

    logger.info("Ball coordinates: bx, by, bz: %f, %f, %f", bx, by, bz)

    if iterations == max_iterations:
        logger.warning("Trajectory overflow")

    return bx, by, bz


def get_neutral_joint_position():
    jp = [0.0] * JOINTS
    jp[0] = -0.3
    jp[2] = math.pi
    a = math.pi / 3.8
    jp[5] = a
    jp[7] = a
    jp[9] = math.pi / 3.5
    jp[10] = math.pi / 2
    return jp


def get_joint_position_supervised(ik_model, state):
    j = [0.0] * JOINTS

    bx, by, bz = final_ball_position(state)

    ball_pos = torch.tensor([bx, by, bz], dtype=torch.float32).unsqueeze(0).to(device)

    if state[31] < 0.9:
        j = ik_model(ball_pos)
        j = j.cpu().detach().numpy().flatten()
        j[10] = math.pi / 2
    else:
        j[0:11] = state[0:11]
    return j


def final_ball_position(state):
    bx, by, bz = state[17:20]
    vx, vy, vz = state[20:23]
    g = 9.81  # Gravity acceleration
    d = 0.01  # Simulation time interval
    restitution = 0.7  # Restitution coefficient for bounce
    min_height = 0.001  # Minimum height to consider a bounce
    max_bounces = 2  # Maximum number of bounces
    attrito = 0.3  # Lateral friction coefficient
    bounces = 0
    minz = 999

    while by > 0 and bounces < max_bounces:
        # Update position
        bx += vx * d
        by += vy * d
        bz += vz * d
        minz = min(minz, bz)
        
        vz -= g * d
        if bz <= 0:
            bz = -bz * restitution  # Bounce with energy loss
            vz = -vz * restitution  # Invert vertical velocity with energy loss
            vx *= (1 - attrito)  
            vy *= (1 - attrito) 
            bounces += 1
            if abs(vz) < min_height:
                break

    return bx, by, bz


def run(cli):
    checkpoint_dir = "checkpoint_RL"
    # Define reward
    seed = 123
    gamma = 0.99
    tau = 0.001
    hidden_size = (400, 300)
    replay_size = 1000000
    noise_stddev = 0.2
    batch_size = 16

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Define and build DDPG agent
    observation_space = 13
    action_space = np.array(joint_ranges)

    agent = DDPG(gamma,
                 tau,
                 hidden_size,
                 observation_space,
                 action_space,
                 checkpoint_dir=checkpoint_dir
                 )

    # Initialize replay memory
    memory = ReplayMemory(int(replay_size))

    # Initialize OU-Noise
    nb_actions = action_space.shape[0]
    ou_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions), sigma=float(noise_stddev) * np.ones(nb_actions))

    # Initialize IK model
    model = RegressionModel().to(device)
    model.load_state_dict(torch.load("model0.pth", map_location=device))
    model.eval()
        
    while True:
        cli_state = cli.get_state()  # Assume get_state returns the initial state
        raw_initial_state = cli_state
        state = get_input_actor(cli_state)
        state = torch.Tensor(state).unsqueeze(0)

        action_done = False
        done = False
        j = get_joint_position_supervised(model, cli_state)
        if min_distance_paddle_ball(cli_state) < 0.2:
            action = agent.calc_action(state, action_noise=ou_noise)
            new_joints = action.squeeze(0)
            j[9] = new_joints[0]
            action_done = True

        cli.send_joints(j)

            # Assuming you get the next state after performing the action
        raw_next_state = cli.get_state()  # Get the updated state from the environment
        next_state = get_input_actor(raw_next_state)
        next_state = torch.Tensor(next_state).unsqueeze(0)

        state = next_state
        cli_state = raw_next_state



def min_distance_paddle_ball(state):
    ball_position = state[17:20]
    paddle_position = state[11:14]
    return np.linalg.norm(paddle_position - ball_position)


def robot_paddle_touched_ball(state):
    vx, vy, vz = state[20:23]
    paddle_position = state[11:14]
    ball_position = state[17:20]
    ball_touched_your_robot = state[31]

    return ball_touched_your_robot > 0.9 and vy > 0


def opponent_field_is_touched(bx, by, bz):
    opponent_table_length = 1.2
    my_table_length = 1.2
    opponent_table_width = 1.4

    return by > my_table_length and by <= my_table_length + opponent_table_length and bx >= -opponent_table_width / 2 and bx <= opponent_table_width / 2

def my_field_is_touched(bx, by, bz):
    my_table_length = 1.2
    table_width = 1.4

    return 0 <= by <= my_table_length and -table_width / 2 <= bx <= table_width / 2

def missed_ball(state):
    bx, by, bz = state[17:20]  # Ball position
    vx, vy, vz = state[20:23]  # Ball velocity
    px, py, pz = state[11:14]  # Paddle position

    # Define a buffer for the paddle's reach in each direction
    buffer_x = 0.1
    buffer_y = 0.1
    buffer_z = 0.1

    # Check if the ball is past the paddle's y position
    if by < py:
        # Further check if the ball is within the buffer range in x and z direction
        if not (px - buffer_x <= bx <= px + buffer_x and pz - buffer_z <= bz <= pz + buffer_z):
            # Consider the ball missed if it's not within the paddle's reach
            return True

    # Additional check if the ball's velocity indicates it is moving away from the paddle
    if vy < 0:
        return True

    return False


def compute_reward(state):

    bx, by, bz = where_ball_touched(state)
        
    opponent_table_length = 1.2
    my_table_length = 1.2
    opponent_table_width = 1.4
    target_x = 0.0
    target_y = my_table_length + opponent_table_length / 2

    # Calculate distance from opponent field center
    center_opp_field = np.sqrt((bx - target_x) ** 2 + (by - target_y) ** 2)

    
    if my_field_is_touched(bx, by, bz):
        logger.info("Ball in your field")
        if by > 0.7:
            return -100
        return -500
    
    # Check if the ball landed in the opponent's field
    if opponent_field_is_touched(bx, by, bz):
        logger.info("Ball in")
        return max(0, 100 - center_opp_field)

    
    logger.info("Ball out")
    return -300


def check_done(state, next_state):
    current_score_robot = state[34]
    current_score_opponent = state[35]
    next_score_robot = next_state[34]
    next_score_opponent = next_state[35]

    if next_score_robot > current_score_robot or next_score_opponent > current_score_opponent:
        return True

    ball_in_field = next_state[29] == 1 or next_state[32] == 1
    if not ball_in_field:
        return True

    return False


def main():
    name = 'Group_08'
    if len(sys.argv) > 1:
        name = sys.argv[1]

    port = DEFAULT_PORT
    if len(sys.argv) > 2:
        port = sys.argv[2]

    host = 'localhost'
    if len(sys.argv) > 3:
        host = sys.argv[3]

    cli = Client(name, host, port)
    run(cli)


if __name__ == '__main__':
    main()
