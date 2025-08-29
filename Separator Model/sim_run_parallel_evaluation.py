import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import helper_functions as hf
from sim_run import run_sim
import csv

N_CPU = 2

# experiment = "detail_V_dis" # "main" if ye + niba tests, "sozh" tests from AVT.FVT
# df = pd.read_excel("Input/data_main.xlsx", sheet_name=experiment)
experiment = 'testing_validation'
df = pd.read_csv('Input/df_te_dpz.csv')
exp = df['exp'].tolist()
phi_0 = df['phi_0'].tolist()
dV_ges = df['dV_ges'].tolist()
eps_0 = df['eps_0'].tolist()
if (experiment == "sozh" or experiment == 'testing_validation'):
    h_c_0 = df['h_c_0'].tolist()
    h_dis_0 = df['h_dis_0'].tolist()



def parallel_simulation(params):
    if (experiment == "main" or experiment == 'detail_V_dis'):    
        exp, phi_0, dV_ges, eps_0 = params
        print(f"Start simulation with exp={exp}, phi_0={phi_0}, dV_ges={dV_ges}, eps_0={eps_0}")
    elif(experiment == "sozh" or experiment == 'testing_validation'):
        exp, phi_0, dV_ges, eps_0, h_c_0, h_dis_0 = params
        print(f"Start simulation with exp={exp}, phi_0={phi_0}, dV_ges={dV_ges}, eps_0={eps_0}, h_c_0={h_c_0}, h_dis_0={h_dis_0}")
    try:
        if (experiment == "main" or experiment == 'detail_V_dis'):
            Sim = run_sim(exp, phi_0, dV_ges, eps_0)
            result = {'exp': exp, 'phi_0': phi_0, 'dV_ges': dV_ges, 'eps_0': eps_0, 'exponent':Sim.Set.exponent, 'u_0':Sim.u_0,
                    'V_dis_total': Sim.V_dis_total, 'Sep. Eff.': Sim.E, 'L_DPZ':Sim.L_DPZ,
                    'Vol_imbalance [%]': hf.calculate_volume_balance(Sim), 'status': 'success'}
        elif(experiment == "sozh" or experiment == 'testing_validation'):
            Sim = run_sim(exp, phi_0, dV_ges, eps_0, h_c_0, h_dis_0)
            result = {'exp': exp, 'phi_0': phi_0, 'dV_ges': dV_ges, 'eps_0': eps_0,
                'h_d_0': h_c_0, 'h_dis_0': h_dis_0, 'sim_status':Sim.status, 'dpz_flooded': Sim.dpz_flooded, 'u_0':Sim.u_0,
                'V_dis_total': Sim.V_dis_total, 'Sep. Eff.': Sim.E,'Vol_imbalance [%]': hf.calculate_volume_balance(Sim), 'status': 'success'}  
            
        # Expand h_d and h_dpz arrays into separate columns
        for i, val in enumerate(Sim.h_c):
            result[f'h_c_{i}'] = val
        for i, val in enumerate(Sim.h_dpz):
            result[f'h_dpz_{i}'] = val
            
         # Schreibe das Ergebnis sofort in die CSV-Datei
        # with open('simulation_results_final_validation_check.csv', mode='a', newline='') as file:
        #     writer = csv.DictWriter(file, fieldnames=result.keys())
        #     writer.writerow(result)
        return result 
    except Exception as e:
        if (experiment == "main" or experiment == 'detail_V_dis'):
            print(f"Simulation failed for exp={exp}, phi_0={phi_0}, dV_ges={dV_ges}, eps_0={eps_0}: {str(e)}")
            error_result = {'exp': exp, 'phi_0': phi_0, 'dV_ges': dV_ges, 'eps_0': eps_0, 'error': str(e), 'status': 'failed'}
        elif(experiment == "sozh" or experiment == 'testing_validation'):
            print(f"Simulation failed for exp={exp}, phi_0={phi_0}, dV_ges={dV_ges}, eps_0={eps_0}, h_c_0={h_c_0}, h_dis_0={h_dis_0}: {str(e)}")
            error_result = {'exp': exp, 'phi_0': phi_0, 'dV_ges': dV_ges, 'eps_0': eps_0, 'h_c_0': h_c_0, 'h_dis_0': h_dis_0, 'error': str(e), 'status': 'failed'}

        # with open('simulation_results_final_validation_check.csv', mode='a', newline='') as file:
        #     writer = csv.DictWriter(file, fieldnames=error_result.keys())
        #     writer.writerow(error_result)
        return error_result

if __name__ == "__main__":
    if (experiment == "main" or experiment == 'detail_V_dis'):
        parameters = [(exp[i], phi_0[i], dV_ges[i], eps_0[i]) for i in range(len(exp))]
    elif(experiment == "sozh" or experiment == 'testing_validation'):
        parameters = [(exp[i], phi_0[i], dV_ges[i], eps_0[i], h_c_0[i], h_dis_0[i]) for i in range(len(exp))]
        
    # n = 201  # Adjust to expected max length of h_d and h_dpz
    # base_fields = ['exp', 'phi_0', 'dV_ges', 'eps_0', 'h_c_0', 'h_dis_0', 'sim_status', 'dpz_flooded', 'u_0', 'V_dis_total', 'Sep. Eff.', 'Vol_imbalance [%]', 'status']
    # h_d_fields = [f'h_c_{i}' for i in range(1,n)]
    # h_dpz_fields = [f'h_dpz_{i}' for i in range(n)]
    # all_fields = base_fields + h_d_fields + h_dpz_fields
    # Header der CSV-Datei schreiben, falls sie noch nicht existiert
    # with open('simulation_results_final_validation_check.csv', mode='w', newline='') as file:
    #     writer = csv.DictWriter(file, fieldnames=all_fields)
    #     writer.writeheader()

    results = joblib.Parallel(n_jobs=N_CPU, backend='multiprocessing')(joblib.delayed(parallel_simulation)(param) for param in parameters)

    # Save results
    df_results = pd.DataFrame(results)
    # h_dpz_columns = pd.DataFrame(df_results['h_dpz'].tolist())   # Convert h_dpz (list of arrays) into separate columns
    # h_dpz_columns.columns = [f'h_dpz_{i}' for i in range(h_dpz_columns.shape[1])]
    # df_results = df_results.drop(columns=['h_dpz'])
    # df_results = pd.concat([df_results, h_dpz_columns], axis=1)  # Concatenate V_dis columns with the main result dataframe
    df_results.to_csv('simulation_results_te_dpz.csv', index=False)
    print("Alle Simulationen abgeschlossen. Ergebnisse gespeichert.")