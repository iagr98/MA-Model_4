import numpy as np
import pandas as pd
import joblib
from sim_run import run_sim

N_CPU = 4
N_D = 15
filename = "Paraffin_flut_20C.xlsx"
N_x = 101
atol = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]

var = 'atol'


def parallel_simulation(params):
    N_D, filename, N_x, atol = params
    print(f"Start simulation with {var}={atol}")                                        # Update parameter in second {}
    try:
        Sim = run_sim(filename, N_D=N_D, N_x=N_x, a_tol=atol)
        return {f"{var}": atol, 'V_dis_total': Sim.V_dis_total, 'status': 'success'}    # Update parameter in second place
    except Exception as e:
        print(f"Simulation failed by {var}={atol}: {str(e)}")                           # Update parameter in second {}
        return {f"{var}": atol, 'error': str(e), 'status': 'failed'}                    # Update parameter in second place

if __name__ == "__main__":
    parameters = [(N_D, filename, N_x, atol_value) for atol_value in atol]              # Update parameter var_value, var_value & var 
    
    results = joblib.Parallel(n_jobs=N_CPU)(joblib.delayed(parallel_simulation)(param) for param in parameters)
    
    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv('simulation_results_parallel.csv', index=False)
    print("Alle Simulationen abgeschlossen. Ergebnisse gespeichert.")