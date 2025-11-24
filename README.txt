Project Scripts Description
----------------------------

This README provides an overview of the scripts included in the project. Each script plays a role in the development, training, or utility functions within the project focusing on robotics control and machine learning model interaction.

1. plot_dataset_coverage_x_z.py & plot_dataset_coverage_y_z.py
   Description: These scripts generate scatter plots for dataset coverage in different axes (X-Z and Y-Z planes respectively). They help visualize data distribution by plotting data points from CSV files.
   Usage: python plot_dataset_coverage_x_z.py <csv_file1> <csv_file2> ...

2. client_just_follow.py
   Description: This script uses a pretrained regression model to control robot joints based on real-time data inputs, following a specified target or trajectory.
   Usage: python client_just_follow.py [optional_name] [optional_port] [optional_host]

3. inverse_kinematics_and_math.py
   Description: Contains functionality for simulating joint positions and their impacts on a robot's interaction with a ball, utilizing inverse kinematics.
   Usage: python inverse_kinematics_and_math.py [optional_name] [optional_port] [optional_host]

4. train_inverse_kinematics.py
   Description: Script for training a neural network model on joint angle prediction for robotic inverse kinematics. It includes data loading, model definition, and training loop.
   Usage: Ensure that you have the dataset in the specified path and run python train_inverse_kinematics.py.

5. client_generate_dataset_high.py, client_generate_dataset_mid.py, client_generate_dataset_low.py
   Description: These scripts are used to generate datasets for different positions (high, mid, low) of the robot's joint movements. They simulate various joint configurations and record the results.
   Usage: python client_generate_dataset_high.py [optional_name] [optional_port] [optional_host]

6. train_reinforcement_learning.py
   Description: Implements a training environment for a reinforcement learning model focusing on robotic control. It uses DDPG (Deep Deterministic Policy Gradient) algorithm for training.
   Usage: python train_reinforcement_learning.py [optional_name] [optional_port] [optional_host]

7. client_keyboard_controllable.py
   Description: Allows manual control of robot joints via keyboard inputs, suitable for real-time interaction and control testing.
   Usage: python client_keyboard_controllable.py [optional_name] [optional_port] [optional_host]

8. converter_txt_to_csv.py
   Description: Converts data from a TXT file into a CSV format, setting up data for further processing or visualization.
   Usage: python converter_txt_to_csv.py

9. find_max_min_dataset.py
   Description: Analyzes a dataset to find the maximum and minimum values of specified columns, typically used for data normalization or analysis.
   Usage: python find_max_min_dataset.py <path_to_csv_file>

----------------------------
