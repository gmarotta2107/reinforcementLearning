import numpy as np
import math
from client import Client, JOINTS, DEFAULT_PORT
import sys
import torch
from utils.nets import RegressionModel


def transform_output(outputs):
    # Define the ranges for output transformation
    ranges = [
        (-0.3, 0.3),  # Index 0
        (-0.8, 0.8),  # Index 1
        (-math.pi / 2, math.pi / 2),  # Index 3
        (-3 * math.pi / 4, 3 * math.pi / 4),  # Index 5
        (-3 * math.pi / 4, 3 * math.pi / 4),  # Index 7
        (-3 * math.pi / 4, 3 * math.pi / 4),  # Index 9
    ]

    for i, (min_val, max_val) in enumerate(ranges):
        outputs[:, i] = (outputs[:, i] * (max_val - min_val) / 2) + (max_val + min_val) / 2

    return outputs


def run(cli):
    model = RegressionModel()
    model.load_state_dict(torch.load('model0.pth', map_location=torch.device('cpu')))
    model.eval()

    while True:
        state = [cli.get_state()]
        input_data = [state[0][17], state[0][18], state[0][19]]

        # Convert input_data to a PyTorch tensor and reshape it to the correct dimensions
        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            output = model(input_tensor)

        output_list = output.squeeze(0).tolist()
        cli.send_joints(output_list)


def main():
    name = 'Following this ball'

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
