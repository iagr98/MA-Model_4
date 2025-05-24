import sim_model as sm
import sim_parameters as sp
import numpy as np
import pandas as pd
import helper_functions as hf

def init_sim(filename):
    Set = sp.Settings()
    SubSys = sp.Substance_System()
    SubSys.update(filename)
    return sm.input_simulation(Set, SubSys)


file_path = "c:\\Users\\Asus\\OneDrive\\Documentos\\RWTH\\Semester 4 (SS25)\\MA\\W9_Vergleichen_Vdis_sep.Effi\\Versuche AVT.FVT\\Vdis_niba.xlsx"
file_path_results = "c:\\Users\\Asus\\OneDrive\\Documentos\\RWTH\\Semester 4 (SS25)\\MA\W9_Vergleichen_Vdis_sep.Effi\\Versuche AVT.FVT\\Vdis_Model_4_results.xlsx"
filename = ['niba_V1.xlsx', 'niba_V2.xlsx', 'niba_V3.xlsx', 'niba_V4.xlsx']

for i in range(len(filename)):
    Sim = init_sim(filename[i])
    A = np.pi*(Sim.Set.D**2)/4
    if (i==0):
        sheet = "V1"
    elif(i==1):
        sheet = "V2"
    elif(i==2):
        sheet = "V3"
    elif(i==3):
        sheet = "V4"
    df_input = pd.read_excel(file_path, sheet_name=sheet) 
    phi_in0 = df_input['d_32in'].to_numpy()
    dV_ges = df_input['V_ges'].to_numpy()/(3.6 * 1e6)
    eps_0 = df_input['varphi'].to_numpy()
    V_dis = np.zeros(len(phi_in0))

    for j in range(len(phi_in0)):
        try:
            Sim.Sub.phi_0 = phi_in0[j]
            Sim.Sub.dV_ges = dV_ges[j]
            Sim.Sub.eps_0 = eps_0[j]
            Sim.initial_conditions(s=1.5)
            Sim.simulate_ivp()
            V_dis[j] = Sim.V_dis_total
        except Exception as e:
            print(f"Error in simulation for phi_in0={phi_in0[j]}: {e}")
            V_dis[j] = 0

    df = pd.DataFrame({
        'd_32in': phi_in0,
        'V_ges': dV_ges * (3.6 * 1e6), # convert dV_ges into L/h
        'varphi': eps_0,
        'V_dis': V_dis
    })

    with pd.ExcelWriter(file_path_results, engine='openpyxl', mode='a') as writer:
        df.to_excel(writer, sheet_name=sheet, index=False)
