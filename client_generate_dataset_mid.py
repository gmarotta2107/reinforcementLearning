import numpy as np
from client import Client, JOINTS, DEFAULT_PORT
import sys
import math
import csv
import os

JOINT_LIMITS = [
    (-0.3, 0.3),
    (-0.8, 0.8),
    (-math.pi, math.pi),
    (-math.pi / 2, math.pi / 2),
    (-math.pi, math.pi),
    (-0.8, 0),
    (-math.pi, math.pi),
    (-0.7, 0),
    (-math.pi, math.pi),
    (-3 * math.pi / 4, 3 * math.pi / 4),
    (-math.pi, math.pi)
]


def get_neutral_joint_position():
    """Returns the neutral position for the joints."""
    neutral_position = [0.0] * JOINTS
    neutral_position[0] = -0.3  # Joint 0: Translation at -0.3 meters
    neutral_position[2] = math.pi  # Joint 2: Rotation at π radians
    angle = math.pi / 3.8
    neutral_position[5] = angle  # Joint 5: Rotation at π/3.8 radians
    neutral_position[7] = angle  # Joint 7: Rotation at π/3.8 radians
    neutral_position[9] = math.pi / 3.5  # Joint 9: Rotation at π/3.5 radians
    neutral_position[10] = math.pi / 2  # Joint 10: Rotation at π/2 radians
    return neutral_position


def generate_constrained_random_joints():
    """Generates random values for the joints within specified limits, with some constraints."""
    joint_values = [np.random.uniform(low, high) for low, high in JOINT_LIMITS]

    # Set specific rotational joints (roll) to 0
    joint_values[2] = 0  # Joint 2: Roll
    joint_values[4] = 0  # Joint 4: Roll
    joint_values[6] = 0  # Joint 6: Roll
    joint_values[8] = 0  # Joint 8: Roll

    # Set specific joints for the paddle

    joint_values[9] = -3 * math.pi / 4 + 0.5  # Joint 9: MID_POSITION

    joint_values[10] = math.pi / 2  # Joint 10: Paddle rotation

    return joint_values


def check_paddle_versor(state, target_versor, tolerance_degrees=10):
    """Checks if the paddle versor is within the specified tolerance."""
    versor_x, versor_y, versor_z = state[14:17]
    current_versor = np.array([versor_x, versor_y, versor_z])

    # Normalize the target and current versors
    target_versor = target_versor / np.linalg.norm(target_versor)
    current_versor = current_versor / np.linalg.norm(current_versor)

    # Calculate the angle between the target and current versors
    dot_product = np.dot(target_versor, current_versor)
    angle_rad = np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)

    # Check if the angle is within the tolerance
    return angle_deg <= tolerance_degrees


def run_simulation(cli, dataset_filename):
    """Runs the simulation and collects data."""
    target_versor = np.array([0, 1, 0])  # Target versor for the paddle


    # Check if the dataset file already exists
    file_exists = os.path.isfile(dataset_filename)

    with open(dataset_filename, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write the header row if the file does not exist
        if not file_exists:
            writer.writerow([
                'joint_0', 'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6',
                'joint_7', 'joint_8', 'joint_9', 'joint_10',
                'paddle_x', 'paddle_y', 'paddle_z'
            ])

        iteration_count = 0
        while iteration_count < 1750:  # Run 1750 iterations
            # Generate random values for the joints
            joint_positions = generate_constrained_random_joints()

            # Send the joint positions to the client
            cli.send_joints(joint_positions)
            current_state = cli.get_state()

            # Check the condition and collect data if valid
            if check_paddle_versor(current_state, target_versor):
                if current_state[26] < 0.1:
                    # Append the joint positions and paddle position to the dataset
                    data_row = joint_positions + list(current_state[11:14])
                    writer.writerow(data_row)
                    print(iteration_count)
                    iteration_count += 1


def main():
    """Punto di ingresso principale del programma."""

    #dataset_filename = "Mid_position_dataset.csv"

    dataset_filename = "datasets/Mid_position_dataset_test.csv"

    name = 'GENERATING DATASET'
    if len(sys.argv) > 1:
        name = sys.argv[1]

    port = DEFAULT_PORT
    if len(sys.argv) > 2:
        port = sys.argv[2]

    host = 'localhost'
    if len(sys.argv) > 3:
        host = sys.argv[3]

    cli = Client(name, host, port)
    run_simulation(cli, dataset_filename)


if __name__ == '__main__':
    main()
