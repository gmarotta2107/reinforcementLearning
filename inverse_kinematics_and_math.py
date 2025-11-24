import time
from utils.nets import *
from client import Client, DEFAULT_PORT, JOINTS
import sys
import torch
from torch import nn
import math

def get_neutral_joint_position():
    """Returns the neutral position of the joints."""
    jp = [0.0] * JOINTS
    jp[0] = -0.3
    jp[2] = math.pi
    a = math.pi / 3.8
    jp[5] = a
    jp[7] = a
    jp[9] = math.pi / 3.5
    jp[10] = math.pi / 2
    return jp

def final_ball_position(state):
    """Calculates the final position of the ball after simulating bounces."""
    bx, by, bz = state[17:20]
    vx, vy, vz = state[20:23]
    g = 9.81  # Acceleration due to gravity
    d = 0.01  # Time interval for the simulation
    restitution = 0.7  # Coefficient of restitution for the bounce
    min_height = 0.001  # Minimum height to consider a bounce
    max_bounces = 2  # Maximum number of bounces
    lateral_friction = 0.3  # Coefficient of lateral friction

    bounces = 0
    minz = 999

    while by > -0.5 and bounces < max_bounces:
        # Update position
        bx += vx * d
        by += vy * d
        bz += vz * d
        minz = min(minz, bz)

        # Update velocity
        vz -= g * d

        # Check if the ball touches the table
        if bz <= 0:
            bz = -bz * restitution  # Bounce with energy loss
            vz = -vz * restitution  # Invert vertical velocity with energy loss
            vx *= (1 - lateral_friction)  # Apply lateral friction
            vy *= (1 - lateral_friction)  # Apply lateral friction
            bounces += 1

            # Stop the simulation if the bounce energy is very low
            if abs(vz) < min_height:
                break

    return bx, by, bz

def choose_position(state):
    """Chooses the position to intercept the ball."""
    px, py, pz = state[11:14]
    bx, by, bz = state[17:20]
    vx, vy, vz = state[20:23]
    d = 0.05
    g = 9.81
    while vz > 0 or bz + d * vz >= pz:
        bx += d * vx
        by += d * vy
        bz += d * vz
        vz -= d * g

    return bx, by, bz

def calculate_dist(state):
    """Calculates the distance between the paddle and the ball."""
    px, py, pz = state[11:14]
    bx, by, bz = state[17:20]
    dist = math.hypot(px - bx, py - by, pz - bz)

    return dist

def ball_in_field(bx, by, bz):
    """Checks if the ball is within the field."""
    TABLE_LENGTH = 2.4
    TABLE_WIDTH = 1.4

    return bx >= -TABLE_WIDTH / 2 and bx <= TABLE_WIDTH / 2 and by >= 0 and by <= TABLE_LENGTH

def run(cli):
    """Runs the main loop of the simulation."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RegressionModel().to(device)
    model.load_state_dict(torch.load("model0.pth", map_location=device))
    model.eval()
    
    while True:
        state = cli.get_state()
        j = [0.0] * JOINTS

        bx, by, bz = final_ball_position(state)

        ballPos_tensor = torch.tensor([bx, by, bz], dtype=torch.float32).unsqueeze(0).to(device)

        if state[31] < 0.9:
            j = model(ballPos_tensor)
            j = j.cpu().detach().numpy().flatten()

            if calculate_dist(state) < 0.2:
                j[9] += max(0, 1.5 - calculate_dist(state))

        else:
            j[0:11] = state[0:11]

        # Send the joint positions to the client
        cli.send_joints(j)

def main():
    
    name = 'Wow math is useful!'
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

