import pandas as pd

smile_sequence = pd.read_csv('smile_sequence.csv')
print(len(set(smile_sequence['SEQUENCE'].tolist())))
print(set(smile_sequence['SEQUENCE'].tolist()[105:110]))
print(set(smile_sequence['SEQUENCE'].tolist()[110:115]))
print(set(smile_sequence['SEQUENCE'].tolist()[115:120]))
print(set(smile_sequence['SEQUENCE'].tolist()[120:125]))
