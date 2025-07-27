import sim_model as sim
import sim_parameters as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def init_sim(exp, phi_0, dV_ges, eps_0, h_d_0, h_dis_0, N_x):
    if (exp == "ye"):
        filename = "Paraffin_flut_20C.xlsx"
        Set = sp.Settings(N_x=N_x, L=0.56, D=0.15, h_d_0=0.055, h_dis_0=0.04)
    elif(exp == "niba1" or exp == "niba2" or exp == "niba3" or exp == "niba4"):
        Set = sp.Settings(N_x=N_x, L=1.0, D=0.2, h_d_0=0.1, h_dis_0=0.03)
        filename = "niba_V1.xlsx" if exp == "niba1" else \
        "niba_V2.xlsx" if exp == "niba2" else \
        "niba_V3.xlsx" if exp == "niba3" else \
        "niba_V4.xlsx" if exp == "niba4" else None
    elif(exp == "2mmol_21C" or exp == "2mmol_30C" or exp == "5mmol_30C" or exp == "10mmol_21C" or exp == "10mmol_30C" or exp == "15mmol_20C" or exp == "15mmol_30C"):
        h_dis_0 = min(h_dis_0, 0.05)
        Set = sp.Settings(N_x=N_x, L=1.3, D=0.2, h_d_0=h_d_0, h_dis_0=h_dis_0)
        filename = "2mmolNa2CO3_21C.xlsx" if exp == "2mmol_21C" else \
        "2mmolNa2CO3_30C.xlsx" if exp == "2mmol_30C" else \
        "5mmolNa2CO3_30C.xlsx" if exp == "5mmol_30C" else \
        "10mmolNa2CO3_21C.xlsx" if exp == "10mmol_21C" else \
        "10mmolNa2CO3_30C.xlsx" if exp == "10mmol_30C" else \
        "15mmolNa2CO3_20C.xlsx" if exp == "15mmol_20C" else \
        "15mmolNa2CO3_30C.xlsx" if exp == "15mmol_30C" else None
    else:
        print('Test does not belong to either Ye Niba or sozh. D=0.3 m, L=1.0 m and substance data from Butylacetat taken')
        filename = "Butylacetat_1_5_144.xlsx"
        Set = sp.Settings(N_x=N_x, L=1.0, D=0.3, h_d_0=0.1, h_dis_0=0.1)
    SubSys = sp.Substance_System()
    SubSys.update(filename)
    SubSys.phi_0 = phi_0
    SubSys.dV_ges = dV_ges / 3.6 * 1e-6
    SubSys.eps_0 = eps_0
    return sim.input_simulation(Set, SubSys)

def run_sim(exp="ye", phi_0=610e-6, dV_ges=240, eps_0=0.5, h_d_0=0.1, h_dis_0=0.05, N_D=20, N_x=201, a_tol=1e-6):
    Sim = init_sim(exp, phi_0, dV_ges, eps_0, h_d_0, h_dis_0, N_x)
    Sim.initial_conditions(N_D)
    Sim.simulate_ivp(atol=a_tol)
    if (exp == "2mmol_21C" or exp == "2mmol_30C" or exp == "5mmol_30C" or exp == "10mmol_21C" or exp == "10mmol_30C" or exp == "15mmol_20C" or exp == "15mmol_30C"):
        if (Sim.status == 1):
            h_dis_0 = h_dis_0 / Sim.factor
            Sim = init_sim(exp, phi_0, dV_ges, eps_0, h_d_0, h_dis_0, N_x)
            Sim.initial_conditions(N_D)
            Sim.simulate_ivp(atol=a_tol)
        else:
            print("No simulation coupling due to DPZ flooding")
    return Sim


if __name__ == "__main__":

    # filename = "Paraffin_flut_20C.xlsx"
    # filename = "niba_V2.xlsx"

    test = 11
    data = pd.read_excel("Input/data_main.xlsx", sheet_name="detail_lambda")
    exp = data['exp'][test]
    phi_0 = data['phi_0'][test]
    dV_ges = data['dV_ges'][test]
    eps_0 = data['eps_0'][test]
    # h_d_0 = data['h_c_0'][test]
    # h_dis_0 = data['h_dis_max'][test]
    h_d_0 = True
    h_dis_0 = True
    print('Simulation inputs: exp={}, phi_0={}, dV_ges={}, eps_0={}'.format(exp, phi_0, dV_ges, eps_0))

    Sim = run_sim(exp=exp, phi_0=phi_0, dV_ges=dV_ges, eps_0=eps_0, h_d_0=h_d_0, h_dis_0=h_dis_0)
    print(Sim.E)
    # Animationen

    # zu Phasenhöhe, Sauterdrchmesser und Hold-Up
    # plots = ['heights', 'phi_32', 'hold_up']
    # eSim.plot_anim(plots)

    # zu Tropfenanzahl
    # plots = ['N_j']
    # eSim.plot_anim(plots)

    # zu Tropfenanzahl + Phasenhöhe
    # plots = ['N_j','heights']
    # eSim.plot_anim(plots)

    # zu Phasenhöhe
    plots = ['heights']
    Sim.plot_anim(plots)