import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import helper_functions as hf
from sim_run import run_sim
import csv

N_CPU = 8

experiment = "sozh" # "main" if ye + niba tests, "sozh" tests from AVT.FVT

df = pd.read_excel("Input/data_main.xlsx", sheet_name=experiment)
exp = df['exp'].tolist()
phi_0 = df['phi_0'].tolist()
dV_ges = df['dV_ges'].tolist()
eps_0 = df['eps_0'].tolist()
if (experiment == "sozh"):
    h_d_0 = df['h_c_0_DPZ_bot_mean'].tolist()   # CHANGE for option 1 or 2
    h_dis_0 = df['h_dis_max'].tolist()



def parallel_simulation(params):
    if (experiment == "main"):    
        exp, phi_0, dV_ges, eps_0 = params
        print(f"Start simulation with exp={exp}, phi_0={phi_0}, dV_ges={dV_ges}, eps_0={eps_0}")
    elif(experiment == "sozh"):
        exp, phi_0, dV_ges, eps_0, h_d_0, h_dis_0 = params
        print(f"Start simulation with exp={exp}, phi_0={phi_0}, dV_ges={dV_ges}, eps_0={eps_0}, h_d_0={h_d_0}, h_dis_0={h_dis_0}")
    try:
        if (experiment == "main"):
            Sim = run_sim(exp, phi_0, dV_ges, eps_0)
            result = {'exp': exp, 'phi_0': phi_0, 'dV_ges': dV_ges, 'eps_0': eps_0,
                    'V_dis_total': Sim.V_dis_total, 'Sep. Eff.': Sim.E,
                    'Vol_imbalance [%]': hf.calculate_volume_balance(Sim), 'status': 'success'}
        elif(experiment == "sozh"):
            Sim = run_sim(exp, phi_0, dV_ges, eps_0, h_d_0, h_dis_0)
            result = {'exp': exp, 'phi_0': phi_0, 'dV_ges': dV_ges, 'eps_0': eps_0,
                'h_d_0': h_d_0, 'h_dis_0': h_dis_0,
                'V_dis_total': Sim.V_dis_total, 'Sep. Eff.': Sim.E,
                'Vol_imbalance [%]': hf.calculate_volume_balance(Sim), 'status': 'success'}  
            
         # Schreibe das Ergebnis sofort in die CSV-Datei
        with open('simulation_results_parallel_evaluation_sozh_opt_2.csv', mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=result.keys())
            writer.writerow(result)
        return result 
    except Exception as e:
        if (experiment == "main"):
            print(f"Simulation failed for exp={exp}, phi_0={phi_0}, dV_ges={dV_ges}, eps_0={eps_0}: {str(e)}")
            error_result = {'exp': exp, 'phi_0': phi_0, 'dV_ges': dV_ges, 'eps_0': eps_0, 'error': str(e), 'status': 'failed'}
        elif(experiment == "sozh"):
            print(f"Simulation failed for exp={exp}, phi_0={phi_0}, dV_ges={dV_ges}, eps_0={eps_0}, h_d_0={h_d_0}, h_dis_0={h_dis_0}: {str(e)}")
            error_result = {'exp': exp, 'phi_0': phi_0, 'dV_ges': dV_ges, 'eps_0': eps_0, 'h_d_0': h_d_0, 'h_dis_0': h_dis_0, 'error': str(e), 'status': 'failed'}

        with open('simulation_results_parallel_evaluation_sozh_opt_2.csv', mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=error_result.keys())
            writer.writerow(error_result)
        return error_result

if __name__ == "__main__":
    if (experiment == "main"):
        parameters = [(exp[i], phi_0[i], dV_ges[i], eps_0[i]) for i in range(len(exp))]
    elif(experiment == "sozh"):
        parameters = [(exp[i], phi_0[i], dV_ges[i], eps_0[i], h_d_0[i], h_dis_0[i]) for i in range(len(exp))]
        
    # Header der CSV-Datei schreiben, falls sie noch nicht existiert
    with open('simulation_results_parallel_evaluation_sozh_opt_2.csv', mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['exp', 'phi_0', 'dV_ges', 'eps_0', 'h_d_0', 'h_dis_0', 'V_dis_total', 'Sep. Eff.', 'Vol_imbalance [%]', 'status', 'error'])
        writer.writeheader()

    results = joblib.Parallel(n_jobs=N_CPU)(joblib.delayed(parallel_simulation)(param) for param in parameters)

    # Save results
    # df_results = pd.DataFrame(results)
    # df_results.to_csv('simulation_results_parallel_evaluation_sozh_opt_2.csv', index=False)
    print("Alle Simulationen abgeschlossen. Ergebnisse gespeichert.")

   