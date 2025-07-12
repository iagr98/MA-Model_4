import pandas as pd

df = pd.read_csv('Output\\simulation_results_parallel_evaluation_sozh_opt_2.csv')

df = df[df['V_dis_total'] > 0.03]
df = df[df['h_dis_0'] < 0.05]
df = df[df['dV_ges'] > 1000]

df.to_csv('Output\\utils_1.csv', index=False)

