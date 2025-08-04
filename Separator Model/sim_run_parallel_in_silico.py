import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import helper_functions as hf
from sim_run import run_sim
import csv

N_CPU = 8

experiment = "detail_V_dis" # "main" if ye + niba tests, "sozh" tests from AVT.FVT

df = pd.read_excel("Input/in_silico_dataset.xlsx")
exp = "in_silico"
dV_ges = df['dV_ges'].tolist()
eps_0 = df['eps_0'].tolist()
phi_0 = df['phi_0'].tolist()
N_D = 20 # Anzahl von Tropfenklassen


def parallel_simulation(params):
    
    phi_0, dV_ges, eps_0 = params
    print(f"Start simulation with exp={exp}, phi_0={phi_0}, dV_ges={dV_ges}, eps_0={eps_0}")
    try:
        Sim = run_sim(exp, phi_0, dV_ges, eps_0)
        result = {'exp': exp, 'phi_0': phi_0, 'dV_ges': dV_ges, 'eps_0': eps_0,
            'sim_status':Sim.status, 'dpz_flooded': Sim.dpz_flooded, 'u_0':Sim.u_0,
            'V_dis_total': Sim.V_dis_total, 'Sep. Eff.': Sim.E,
            'V_c':",".join(map(str,(Sim.V_c[:,-1]))), 'V_dis':",".join(map(str,(Sim.V_dis[:,-1]))),
            'V_d':",".join(map(str,(Sim.V_d[:,-1]))), 'phi_32':",".join(map(str,(Sim.phi_32[:,-1]))),
            'Vol_imbalance [%]': hf.calculate_volume_balance(Sim), 'status': 'success'}  
        for j in range(N_D):
            result[f'N_{j}'] = ",".join(map(str, Sim.N_j[j][:,-1]))
            
         # Schreibe das Ergebnis sofort in die CSV-Datei
        with open('simulation_results_parallel_in_silico.csv', mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=result.keys())
            writer.writerow(result)
        return result 
    except Exception as e:
        print(f"Simulation failed for exp={exp}, phi_0={phi_0}, dV_ges={dV_ges}, eps_0={eps_0}: {str(e)}")
        error_result = {'exp': exp, 'phi_0': phi_0, 'dV_ges': dV_ges, 'eps_0': eps_0, 'error': str(e), 'status': 'failed'}

        with open('simulation_results_parallel_in_silico.csv', mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=error_result.keys())
            writer.writerow(error_result)
        return error_result

if __name__ == "__main__":
    parameters = [(phi_0[i], dV_ges[i], eps_0[i]) for i in range(len(phi_0))]
        
    base_fields = ['exp', 'phi_0', 'dV_ges', 'eps_0', 'sim_status', 'dpz_flooded', 'u_0', 'V_dis_total', 'Sep. Eff.',
               'V_c', 'V_dis', 'V_d', 'phi_32', 'Vol_imbalance [%]', 'status']
    base_fields += [f'N_{j}' for j in range(N_D)]
    with open('simulation_results_parallel_in_silico.csv', mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=base_fields)
        writer.writeheader()

    results = joblib.Parallel(n_jobs=N_CPU, backend='multiprocessing')(joblib.delayed(parallel_simulation)(param) for param in parameters)

    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv('simulation_results_parallel_in_silico_1.csv', index=False)
    print("Alle Simulationen abgeschlossen. Ergebnisse gespeichert.")