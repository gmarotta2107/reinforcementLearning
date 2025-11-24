import pandas as pd

# Specifica il percorso del file di input e di output
input_file_path = 'jointss_values.txt'
output_file_path = 'jointss_values.csv'

# Definisci l'intestazione
header = [
    'joint_0', 'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5',
    'joint_6', 'joint_7', 'joint_8', 'joint_9', 'joint_10',
    'paddle_x', 'paddle_y', 'paddle_z'
]

# Carica il file di testo
df = pd.read_csv(input_file_path, sep=' ', header=None)

# Assegna l'intestazione al dataframe
df.columns = header

# Salva il dataframe come file CSV
df.to_csv(output_file_path, index=False)

print(f"File CSV salvato come: {output_file_path}")
