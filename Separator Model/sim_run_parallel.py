import numpy as np
import pandas as pd
import joblib
from sim_run import run_sim

N_CPU = 4
N_D_value = 15
filename = "Paraffin_flut_20C.xlsx"
s_value = 1.25 
N_x_values = [61, 71, 81, 91]


def parallel_simulation(params): # Quasi "wrapper function from sozh"
    N_D, filename, s, N_x = params
    print(f"Start simulation with N_x={N_x}")
    try:
        Sim = run_sim(filename, s=s, N_D=N_D, N_x=N_x)
        return {'N_x': N_x, 'V_dis_total': Sim.V_dis_total, 'status': 'success'}
    except Exception as e:
        print(f"Simulation failed by N_D={N_D}: {str(e)}")
        return {'N_x': N_x, 'error': str(e), 'status': 'failed'}

if __name__ == "__main__":
    parameters = [(N_D_value, filename, s_value, N_x) for N_x in N_x_values]
    
    results = joblib.Parallel(n_jobs=N_CPU)(joblib.delayed(parallel_simulation)(param) for param in parameters)
    
    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv('simulation_results_parallel.csv', index=False)
    print("Alle Simulationen abgeschlossen. Ergebnisse gespeichert.")