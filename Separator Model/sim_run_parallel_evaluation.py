import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import helper_functions as hf
from sim_run import run_sim

N_CPU = 8


exp = ["ye", "niba1", "niba3"]
phi_0 = [325e-6, 700e-6, 635e-6]
dV_ges = [200 / 3.6 * 1e-6, 2350  / 3.6 * 1e-6, 1150 / 3.6 * 1e-6]
eps_0 = [0.2, 0.3, 0.5]

var = 'N_x'             # Define


def parallel_simulation(params):
    exp, phi_0, dV_ges, eps_0 = params
    print(f"Start simulation with exp={exp}, phi_0={phi_0}, dV_ges={dV_ges}, eps_0={eps_0}")
    try:
        Sim = run_sim(exp, phi_0, dV_ges, eps_0)
        return {'exp': exp, 'phi_0': phi_0, 'dV_ges': dV_ges, 'eps_0': eps_0,
                'V_dis_total': Sim.V_dis_total, 'Sep. Eff.': Sim.E,
                'Vol_imbalance [%]': hf.calculate_volume_balance(Sim), 'status': 'success'}
    except Exception as e:
        print(f"Simulation failed for exp={exp}, phi_0={phi_0}, dV_ges={dV_ges}, eps_0={eps_0}: {str(e)}")
        return {'exp': exp, 'phi_0': phi_0, 'dV_ges': dV_ges, 'eps_0': eps_0, 'error': str(e), 'status': 'failed'}

if __name__ == "__main__":
    parameters = [(exp[i], phi_0[i], dV_ges[i], eps_0[i]) for i in range(len(exp))]
    
    results = joblib.Parallel(n_jobs=N_CPU)(joblib.delayed(parallel_simulation)(param) for param in parameters)
    
    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv('simulation_results_parallel_evaluation.csv', index=False)
    print("Alle Simulationen abgeschlossen. Ergebnisse gespeichert.")

   