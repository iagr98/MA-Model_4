import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import helper_functions as hf
from sim_run import run_sim
import csv

N_CPU = 8

dV_ges = [250, 500, 750, 1000, 1250, 1500, 1750, 2000]
phi_0 = [100e-6, 150e-6, 200e-6, 225e-6, 300e-6, 500e-6]

def parallel_simulation(params):    
    phi_0, dV_ges = params
    print(f"Start simulation with phi_0={phi_0}, dV_ges={dV_ges}")
    try:
        Sim = run_sim(exp='sensitivity', phi_0=phi_0, dV_ges=dV_ges, eps_0=0.25)
        result = {'phi_0': phi_0, 'dV_ges': dV_ges,'Sep. Eff.': Sim.E,'Vol_imbalance [%]': hf.calculate_volume_balance(Sim), 'status': 'success'}   
        #  Schreibe das Ergebnis sofort in die CSV-Datei
        with open('simulation_results_sensitivity_lambda_offset_eps_0.25.csv', mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=result.keys())
            writer.writerow(result)
        return result 
    except Exception as e:
        print(f"Simulation failed for phi_0={phi_0}, dV_ges={dV_ges}: {str(e)}")
        error_result = {'phi_0': phi_0, 'dV_ges': dV_ges, 'error': str(e), 'status': 'failed'}
        with open('simulation_results_sensitivity_lambda_offset_eps_0.25.csv', mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=error_result.keys())
            writer.writerow(error_result)
        return error_result

if __name__ == "__main__":

    parameters = [(phi, dV) for dV in dV_ges for phi in phi_0]
        
    # Header der CSV-Datei schreiben, falls sie noch nicht existiert
    with open('simulation_results_sensitivity_lambda_offset_eps_0.25.csv', mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['phi_0', 'dV_ges', 'Sep. Eff.', 'Vol_imbalance [%]', 'status'])
        writer.writeheader()

    results = joblib.Parallel(n_jobs=N_CPU, backend='multiprocessing')(joblib.delayed(parallel_simulation)(param) for param in parameters)

    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv('simulation_results_sensitivity_lambda_offset_eps_0.25_1.csv', index=False)
    print("Alle Simulationen abgeschlossen. Ergebnisse gespeichert.")

   