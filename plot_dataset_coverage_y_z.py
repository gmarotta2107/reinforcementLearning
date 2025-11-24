import argparse
import pandas as pd
import matplotlib.pyplot as plt


def main():
    # Create a parser for command-line arguments
    parser = argparse.ArgumentParser(description='Create a combined scatter plot from one or more CSV files.')
    parser.add_argument('filenames', type=str, nargs='+', help='The name(s) of the CSV file(s) to load')

    # Parse command-line arguments
    args = parser.parse_args()

    # Create a scatter plot
    plt.figure()

    # Iterate over each provided file
    for filename in args.filenames:
        # Load the CSV file
        df = pd.read_csv(filename)

        # Extract data from columns
        x = df['paddle_y']
        y = df['paddle_z']

        # Add data to the scatter plot
        plt.scatter(x, y, label=filename)

    # Add labels, title, and legend
    plt.xlabel('Y Axis')
    plt.ylabel('Z Axis')
    plt.title('Combined Scatter Plot of Points')
    plt.legend()

    # Display the plot
    plt.show()


if __name__ == '__main__':
    main()