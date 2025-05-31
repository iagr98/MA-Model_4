import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df_sim = pd.read_csv("Output/simulation_results_parallel_evaluation.csv")

df_lab = pd.read_excel("Input/data_main.xlsx", sheet_name="main")

V_dis_plus_25 = df_lab['V_dis'] * 1.25
V_dis_minus_25 = df_lab['V_dis'] * 0.75
V_dis_plus_50 = df_lab['V_dis'] * 1.50
V_dis_minus_50 = df_lab['V_dis'] * 0.50


mape_sim = (abs(df_sim['V_dis_total'] - df_lab['V_dis']) / df_lab['V_dis']).mean() * 100
mape_std_sim = (abs(df_sim['V_dis_total'] - df_lab['V_dis']) / df_lab['V_dis']).std() * 100

# elaborate parity plot
plt.figure(figsize=(8, 6))
plt.plot(df_lab['V_dis'], df_lab['V_dis'], label='V_dis vs V_dis', color='black', linestyle='--')
plt.scatter(df_lab['V_dis'], df_sim['V_dis_total'], label=f'Simulation. MAPE: {mape_sim:.2f}%. std. Abw.: {mape_std_sim:.2f}%', color='blue')
plt.plot(df_lab['V_dis'], V_dis_plus_25, label='+25%', color='gray', linestyle=':', alpha=0.5)
plt.plot(df_lab['V_dis'], V_dis_minus_25, label='-25%', color='gray', linestyle=':', alpha=0.5)
plt.plot(df_lab['V_dis'], V_dis_plus_50, label='+50%', color='gray', linestyle='--', alpha=0.5)
plt.plot(df_lab['V_dis'], V_dis_minus_50, label='-50%', color='gray', linestyle='--', alpha=0.5)
plt.xlabel('Experimentelle Ergebnisse (V_dis_exp)')
plt.ylabel('Simulierte Ergebnisse (V_dis_mod)')
plt.title('Vergleich der simulierten Ergebnisse mit experimentellen Daten')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()