import sim_model as sim
import sim_parameters as sp
import numpy as np
import pandas as pd
import helper_functions as hf


def init_sim(filename):
    Set = sp.Settings()
    SubSys = sp.Substance_System()
    SubSys.update(filename)
    return sim.input_simulation(Set, SubSys)


# Execution

filename = "Paraffin_flut_20C.xlsx"
eSim = init_sim(filename)

########################################################### Daten Erfassung ############################################################################################
dV_ges = [160/3.6*1e-6, 200/3.6*1e-6, 240/3.6*1e-6, 280/3.6*1e-6]
file_path = "c:\\Users\\Asus\\OneDrive\\Documentos\\RWTH\\Semester 4 (SS25)\\MA\\W9_Vergleichen_Vdis_sep.Effi\\Vdis_Ye.xlsx"
file_path_results = "c:\\Users\\Asus\\OneDrive\\Documentos\\RWTH\\Semester 4 (SS25)\\MA\\W9_Vergleichen_Vdis_sep.Effi\\Vdis_Model_4_results.xlsx"
A = np.pi*(eSim.Set.D**2)/4

for i in range(len(dV_ges)):
    if (dV_ges[i] == 160/3.6*1e-6):
        sheet_input = "FilteredDataQ160"
    elif(dV_ges[i] == 200/3.6*1e-6):
        sheet_input = "FilteredDataQ200"
    elif(dV_ges[i] == 240/3.6*1e-6):
        sheet_input = "FilteredDataQ240"
    elif(dV_ges[i] == 280/3.6*1e-6):
        sheet_input = "FilteredDataQ280"

    # Daten einlesen
    df_input = pd.read_excel(file_path, sheet_name=sheet_input) 
    phi_in0 = df_input['d_32in'].to_numpy()
    V_dis = np.zeros(len(phi_in0))
    H_DPZ = np.zeros(len(phi_in0))
    L_DPZ = np.zeros(len(phi_in0))
    A_0 = np.zeros(len(phi_in0))

    for j in range(len(phi_in0)):
        try:
            eSim.Sub.phi_0 = phi_in0[j]
            eSim.Sub.dV_ges = dV_ges[i]
            eSim.initial_conditions(s=1.5)

            if (eSim.Sub.phi_0 == 0):
                V_dis[j] = 0
                H_DPZ[j] = 0
                L_DPZ[j] = 0
                A_0[j] = 0
            else:
                # Simulation durchführen
                eSim.simulate_ivp()

                V_dis[j] = eSim.V_dis_total
                H_DPZ[j] = eSim.H_DPZ
                L_DPZ[j] = eSim.L_DPZ
                A_0[j] = (A / 2) - hf.getArea((eSim.Set.D / 2) - H_DPZ[j], eSim.Set.D / 2)
        
        except Exception as e:
            print(f"Fehler in Simulation {i+1}, Iteration {j+1}: {e}. Diese Iteration wird übersprungen.")
            V_dis[j] = np.nan  # Optional: Setze den Wert als NaN oder 0
            H_DPZ[j] = np.nan
            L_DPZ[j] = np.nan
            A_0[j] = np.nan

    # Ergebnisse in DataFrame speichern
    df = pd.DataFrame({
        'H_DPZ': H_DPZ,
        'L_DPZ': L_DPZ,
        'V_dis': V_dis,
        'A_0': A_0
    })

    if (dV_ges[i] == 160/3.6*1e-6):
        sheet_output = "Results_Q160"
    elif(dV_ges[i] == 200/3.6*1e-6):
        sheet_output = "Results_Q200"
    elif(dV_ges[i] == 240/3.6*1e-6):
        sheet_output = "Results_Q240"
    elif(dV_ges[i] == 280/3.6*1e-6):
        sheet_output = "Results_Q280"

    # Ergebnisse in Excel-Datei schreiben
    with pd.ExcelWriter(file_path_results, engine='openpyxl', mode='a') as writer:
        df.to_excel(writer, sheet_name=sheet_output, index=False)

#######################################################################################################################################################