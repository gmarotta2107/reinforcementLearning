import pandas as pd
import sys


def find_min_max(csv_file):
    # Carica il dataset dal file CSV
    df = pd.read_csv(csv_file)

    # Trova i valori minimi e massimi per joint_5 e joint_7
    min_joint_3 = df['joint_3'].min()
    max_joint_3 = df['joint_3'].max()
    min_joint_5 = df['joint_5'].min()
    max_joint_5 = df['joint_5'].max()
    min_joint_7 = df['joint_7'].min()
    max_joint_7 = df['joint_7'].max()

    # Stampa i risultati
    print(f"Minimo joint_3: {min_joint_3}")
    print(f"Massimo joint_3: {max_joint_3}")
    print(f"Minimo joint_5: {min_joint_5}")
    print(f"Massimo joint_5: {max_joint_5}")
    print(f"Minimo joint_7: {min_joint_7}")
    print(f"Massimo joint_7: {max_joint_7}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python find_min_max.py <path_to_csv_file>")
        sys.exit(1)

    csv_file = sys.argv[1]
    find_min_max(csv_file)
