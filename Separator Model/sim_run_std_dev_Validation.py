import sim_model as sim
import sim_parameters as sp
import constants
import numpy as np
import pandas as pd
import helper_functions as hf
import matplotlib.pyplot as plt


def init_sim(filename):
    Set = sp.Settings()
    SubSys = sp.Substance_System()
    SubSys.update(filename)
    return sim.input_simulation(Set, SubSys)


# Execution

filename = ['Paraffin_flut_20C.xlsx', 'niba_V1.xlsx', 'niba_V2.xlsx', 'niba_V3.xlsx', 'niba_V4.xlsx']
file_path = "c:\\Users\\Asus\\OneDrive\\Documentos\\RWTH\\Semester 4 (SS25)\\MA\\W10_Vergleichen_V_dis_alle_Daten\\std_dev_validation.xlsx"
sheet = 'Tabelle1'
include_ye = True
include_nida = False
s_values = np.array([1.0, 1.25, 1.5])

df_input = pd.read_excel(file_path, sheet_name=sheet) 
phi_in0 = df_input['d_32in'].to_numpy()
dV_ges = df_input['Q'].to_numpy()/(3.6 * 1e6)
eps_0 = df_input['varphi'].to_numpy()
V_dis_exp = df_input['V_dis_exp'].to_numpy()
V_dis_results = np.zeros((len(df_input), len(s_values)))

for i in range(len(df_input)):
    if (include_ye == True and i<len(df_input)-4):
        Sim = init_sim(filename[0])
        Sim.Set.L = 0.56
        Sim.Set.D = 0.15
        Sim.Set.h_c_0 = 0.055
        Sim.Set.h_dis_0 = 0.04
        print(filename[0])
    elif(i>=len(df_input)-4 and include_nida==True):
        if(i == len(df_input) - 4):
            Sim = init_sim(filename[1])
            Sim.Set.L = 1
            Sim.Set.D = 0.2
            Sim.Set.h_c_0 = 0.1
            Sim.Set.h_dis_0 = 0.03
            print(filename[1])
        if(i == len(df_input) - 3):
            Sim = init_sim(filename[2])
            Sim.Set.L = 1
            Sim.Set.D = 0.2
            Sim.Set.h_c_0 = 0.1
            Sim.Set.h_dis_0 = 0.03
            print(filename[2])
        if(i == len(df_input) - 2):
            Sim = init_sim(filename[3])
            Sim.Set.L = 1
            Sim.Set.D = 0.2
            Sim.Set.h_c_0 = 0.1
            Sim.Set.h_dis_0 = 0.03
            print(filename[3])
        if(i == len(df_input) - 1):
            Sim = init_sim(filename[4])
            Sim.Set.L = 1
            Sim.Set.D = 0.2
            Sim.Set.h_c_0 = 0.1
            Sim.Set.h_dis_0 = 0.03
            print(filename[4])
    else:
        continue

    for j, s in enumerate(s_values):
        try:
            Sim.Sub.phi_0 =  phi_in0[i]
            Sim.Sub.dV_ges = dV_ges[i]
            Sim.Sub.eps_0 = eps_0[i]
            Sim.initial_conditions(s=s)
            Sim.simulate_ivp()
            V_dis_results[i,j] = Sim.V_dis_total
        except Exception as e:
            print(f"Error in simulation for phi_in0={phi_in0[i]}: {e}")
            V_dis_results[i,j] = 0



plt.figure(figsize=(10, 6))
if (include_nida and include_ye):
    for i in range(len(df_input)):
        line, = plt.plot(s_values, V_dis_results[i, :], marker='o', label=f'{i+1}')
        plt.axhline(y=V_dis_exp[i], color=line.get_color(), linestyle='--', alpha=0.5, label=f'Exp. {i+1}')
    plt.legend(ncol=len(df_input), bbox_to_anchor=(0.5, -0.15), loc='upper center')
elif (include_ye and not include_nida):
    for i in range(len(df_input)-4):
        line, = plt.plot(s_values, V_dis_results[i, :], marker='o', label=f'{i+1}')
        plt.axhline(y=V_dis_exp[i], color=line.get_color(), linestyle='--', alpha=0.5, label=f'Exp. {i+1}')
    plt.legend(ncol=len(df_input)-4, bbox_to_anchor=(0.5, -0.15), loc='upper center')
elif (include_nida and not include_ye):
    for i in range(len(df_input)-4, len(df_input)):
        line, = plt.plot(s_values, V_dis_results[i, :], marker='o', label=f'{i+1}')
        plt.axhline(y=V_dis_exp[i], color=line.get_color(), linestyle='--', alpha=0.5, label=f'Exp. {i+1}')
    plt.legend(ncol=len(df_input)-8, bbox_to_anchor=(0.5, -0.15), loc='upper center')
    

plt.xlabel("Standardabweichung (s)")
plt.ylabel("V_dis_total (m³)")
plt.yscale('log')
plt.title("OAT-Sensitivitätsanalyse von s")

plt.grid(True)
plt.tight_layout()
plt.show()