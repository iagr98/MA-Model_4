import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import helper_functions as hf
from sim_run import run_sim

N_CPU = 8
N_D = [10, 13, 15, 17, 20, 30, 50, 60]
filename = "Paraffin_flut_20C.xlsx"
# filename = "niba_V2.xlsx"
N_x = 151
atol = 1e-6

var = 'N_D'             # Define


def parallel_simulation(params):
    N_D, filename, N_x, atol = params
    print(f"Start simulation with {var}={N_D}")                                        # Update parameter in second {}
    try:
        Sim = run_sim(filename, N_D=N_D, N_x=N_x, a_tol=atol)
        return {f"{var}": N_D, 'Sep. Eff.': Sim.E,'Vol_imbalance [%]': hf.calculate_volume_balance(Sim), 'status': 'success'}    # Update parameter in second place
    except Exception as e:
        print(f"Simulation failed by {var}={N_D}: {str(e)}")                           # Update parameter in second {}
        return {f"{var}": N_D, 'error': str(e), 'status': 'failed'}                    # Update parameter in second place

if __name__ == "__main__":
    parameters = [(N_D_value, filename, N_x, atol) for N_D_value in N_D]              # Update parameter var_value, var_value & var 
    
    results = joblib.Parallel(n_jobs=N_CPU)(joblib.delayed(parallel_simulation)(param) for param in parameters)
    
    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv('simulation_results_parallel.csv', index=False)
    print("Alle Simulationen abgeschlossen. Ergebnisse gespeichert.")

    # Plot results
    df = pd.read_csv("simulation_results_parallel.csv")
    df.columns = df.columns.str.strip()
    plt.figure(figsize=(8, 5))
    plt.plot(df['N_D'], df['Sep. Eff.'], marker='o')         # Update parameter in first place
    # plt.xscale('log')  # da atol logarithmisch skaliert ist
    plt.xlabel('N_D')                                          # Change x-label
    plt.ylabel('Sep. Eff.')
    plt.title(f'Gitterunabhängigkeitsanalyse ({var})')
    plt.grid(True)
    plt.tight_layout()
    plt.show()