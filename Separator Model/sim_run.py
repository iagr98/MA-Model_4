import sim_model as sim
import sim_parameters as sp
import numpy as np


def init_sim(exp, phi_0, dV_ges, eps_0, N_x=101):
    if (exp == "ye"):
        filename = "Paraffin_flut_20C.xlsx"
        Set = sp.Settings(N_x=N_x, L=0.56, D=0.15, h_d_0=0.055, h_dis_0=0.04)
    elif(exp == "niba1" or exp == "niba2" or exp == "niba3" or exp == "niba4"):
        Set = sp.Settings(N_x=N_x, L=1.0, D=0.2, h_d_0=0.1, h_dis_0=0.03)
        filename = "niba_V1.xlsx" if exp == "niba1" else \
        "niba_V2.xlsx" if exp == "niba2" else \
        "niba_V3.xlsx" if exp == "niba3" else \
        "niba_V4.xlsx" if exp == "niba4" else None
    else:
        print('Test does not belong to either Ye or Niba.')
    SubSys = sp.Substance_System()
    SubSys.update(filename)
    SubSys.phi_0 = phi_0
    SubSys.dV_ges = dV_ges / 3.6 * 1e-6
    SubSys.eps_0 = eps_0
    return sim.input_simulation(Set, SubSys)

def run_sim(exp="ye", phi_0=610e-6, dV_ges=240, eps_0=0.2, N_D=20, N_x=201, a_tol=1e-6):
    Sim = init_sim(exp, phi_0, dV_ges, eps_0, N_x)
    Sim.initial_conditions(N_D)
    Sim.simulate_ivp(atol=a_tol)
    return Sim


if __name__ == "__main__":

    # filename = "Paraffin_flut_20C.xlsx"
    # filename = "niba_V2.xlsx"
    N_D = 20
    N_x = 201
    a_tol = 1e-6

    exp = "niba4"
    phi_0 = 630e-6
    dV_ges = 1500
    eps_0 = 0.3
    

    Sim = run_sim(exp, phi_0, dV_ges, eps_0, N_D, N_x, a_tol)

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
    plots = ['heights','phi_32', 'hold_up']
    Sim.plot_anim(plots)