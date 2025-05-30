import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import helper_functions as hf
from sim_run import run_sim

N_CPU = 8
N_D = 30
filename = "Paraffin_flut_20C.xlsx"
# filename = "niba_V2.xlsx"
N_x = 151
atol = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]

var = 'atol'             # Define


def parallel_simulation(params):
    N_D, filename, N_x, atol = params
    print(f"Start simulation with {var}={atol}")                                        # Update parameter in second {}
    try:
        Sim = run_sim(filename, N_D=N_D, N_x=N_x, a_tol=atol)
        return {f"{var}": atol, 'V_dis_total': Sim.V_dis_total, 'Sep. Eff.': Sim.E,'Vol_imbalance [%]': hf.calculate_volume_balance(Sim), 'status': 'success'}    # Update parameter in second place
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

    # Plot results
    df = pd.read_csv("simulation_results_parallel.csv")
    df.columns = df.columns.str.strip()
    plt.figure(figsize=(8, 5))
    # plt.plot(df['atol'], df['Sep. Eff.'], marker='o')         # Update parameter in first place
    plt.plot(df['atol'], df['V_dis_total'], marker='o')         # Update parameter in first place
    plt.xscale('log')  # da atol logarithmisch skaliert ist
    plt.yscale('log')  # da atol logarithmisch skaliert ist
    plt.xlabel('atol')                                          # Change x-label
    plt.ylabel('V_dis')                                         # Change output variable if needed
    plt.title(f'Gitterunabhängigkeitsanalyse ({var})')
    plt.grid(True)
    plt.tight_layout()
    plt.show()