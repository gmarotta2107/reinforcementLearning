import numpy as np
from client import Client, JOINTS, DEFAULT_PORT
import sys
import math
from pynput import keyboard

"""
Robot Control Guide

This program allows you to control a robot using the keyboard.
Here are the keys you can use to control the movements of the robot's joints:

Control Keys:
1  -> Increase joint 3
q  -> Decrease joint 3
2  -> Increase joint 5
w  -> Decrease joint 5
3  -> Increase joint 7
e  -> Decrease joint 7
4  -> Increase joint 9
r  -> Decrease joint 9

Each joint has predefined limits that prevent movements beyond a certain range.

"""

# Define limits for each joint
JOINT_LIMITS = [
    (-0.3, 0.3),                # Joint 0: Translation
    (-0.8, 0.8),                # Joint 1: Translation
    (-math.pi, math.pi),        # Joint 2: Rotation
    (-math.pi / 2, math.pi / 2),# Joint 3: Rotation
    (-math.pi, math.pi),        # Joint 4: Rotation
    (-3 * math.pi / 4, 3 * math.pi / 4), # Joint 5: Rotation
    (-math.pi, math.pi),        # Joint 6: Rotation
    (-3 * math.pi / 4, 3 * math.pi / 4), # Joint 7: Rotation
    (-math.pi, math.pi),        # Joint 8: Rotation
    (-3 * math.pi / 4, 3 * math.pi / 4), # Joint 9: Rotation
    (-math.pi, math.pi)         # Joint 10: Rotation
]

joints = [0] * JOINTS

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

def on_press(key):
    """Handles key press events to adjust joint positions."""
    try:
        increment = 0.1  # Increment/decrement for joint values
        if key.char == '1':
            joints[3] = min(joints[3] + increment, JOINT_LIMITS[3][1])
        elif key.char == 'q':
            joints[3] = max(joints[3] - increment, JOINT_LIMITS[3][0])
        elif key.char == '2':
            joints[5] = min(joints[5] + increment, JOINT_LIMITS[5][1])
        elif key.char == 'w':
            joints[5] = max(joints[5] - increment, JOINT_LIMITS[5][0])
        elif key.char == '3':
            joints[7] = min(joints[7] + increment, JOINT_LIMITS[7][1])
        elif key.char == 'e':
            joints[7] = max(joints[7] - increment, JOINT_LIMITS[7][0])
        elif key.char == '4':
            joints[9] = min(joints[9] + increment, JOINT_LIMITS[9][1])
        elif key.char == 'r':
            joints[9] = max(joints[9] - increment, JOINT_LIMITS[9][0])
    except AttributeError:
        pass

listener = keyboard.Listener(on_press=on_press)
listener.start()

def paddle_versor(state, correct_versor, tolerance_degrees=5):
    """Checks if the paddle's versor is within the specified tolerance."""
    versor_x, versor_y, versor_z = state[14:17]
    current_versor = np.array([versor_x, versor_y, versor_z])

    correct_versor = correct_versor / np.linalg.norm(correct_versor)
    current_versor = current_versor / np.linalg.norm(current_versor)

    dot_product = np.dot(correct_versor, current_versor)
    angle_rad = np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)

    return angle_deg <= tolerance_degrees

def generate_random_joints():
    """Generates random values for all joints."""
    return [np.random.uniform(low, high) for low, high in JOINT_LIMITS]

def run(cli):
    """Runs the simulation and collects data."""
    versor = np.array([0, 1, 0])
    global joints
    joints = get_neutral_joint_position()

    while True:
        # Send joint positions to the client
        cli.send_joints(joints)
        state = cli.get_state()

        # Check condition and collect data if valid
        if paddle_versor(state, versor):
            print(state[5], state[7], state[9], state[13])

def main():
    """Main entry point of the program."""
    name = 'Keyboard controlled'
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

