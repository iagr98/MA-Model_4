import numpy as np
import fun
import helper_functions as hf
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
from helper_functions import getHeightArray


class input_simulation:

    def __init__(self, Settings, Substance_System):

        self.Set = Settings
        self.Sub = Substance_System

        self.y0 = []
        self.sol = []
        self.V_dis = []
        self.V_d = []
        self.V_c = []
        self.phi_32 = []
        self.N_j = []

        self.u_dis = []
        self.u_d = []
        self.u_c = []
        self.d_j = []
        
        self.H_DPZ = 0
        self.L_DPZ = 0
        self.V_dis_total = 0

    def initial_conditions(self, s=0.32):

        dl = self.Set.dl
        N_x = self.Set.N_x
        R = self.Set.D / 2
        h_dis_0 = self.Set.h_dis_0
        h_c_0 = self.Set.h_c_0

        # Einführung der Tropfenanzahl und Tropfendurchmesser
        hold_up_calc, n_in, d_in, N_in_total = fun.initialize_boundary_conditions(self.Sub.eps_0, self.Sub.phi_0, 2.5*self.Sub.phi_0, s , 'Output', plot=True)

        # Berechnungen von Querschnittsflächen
        h_d_0 = 2 * R - h_c_0 - h_dis_0
        A_d_0 = R**2 * np.arccos((R - h_d_0) / R) - (R - h_d_0) * np.sqrt(2 * R * h_d_0 - h_d_0**2)
        A_c_0 = R**2 * np.arccos((R - h_c_0) / R) - (R - h_c_0) * np.sqrt(2 * R * h_c_0 - h_c_0**2)
        A_dis_0 = self.Set.A - A_d_0 - A_c_0

        # Anfangsbedingungen für Volumina
        Vdis_0 = A_dis_0 * dl * np.ones(N_x)
        Vd_0 = A_d_0 * dl * np.ones(N_x)
        Vc_0 = A_c_0 * dl * np.ones(N_x)

        # Anfangsbedingung für phi_32
        phi32_0 = self.Sub.phi_0 * np.ones(N_x)

        # Anfangsbedingung für N_j
        factor = (self.Sub.eps_0*Vd_0[0])/((np.pi/6)*np.sum(n_in*d_in**3))
        N_j_0 = (factor*np.round(n_in,decimals=0)).tolist()
        print('B factor= ', factor)
        print('Total volume of droplets [m^3] =',(np.pi/6)*np.sum(n_in*d_in**3))
        print('eps_0 * V_d,0 [m^3] =',self.Sub.eps_0*Vd_0[0])
        print("Actual number of droplets for given feed Hold-up: ", round(np.sum(N_j_0),0))
        self.d_j = d_in
        for i in range(len(N_j_0)):
            N_j_0[i] = N_j_0[i] * np.ones(N_x)

        self.y0 = np.concatenate([Vdis_0, Vd_0, Vc_0, phi32_0, np.concatenate(N_j_0)])  # Array als Anfangsbedingung

    # Momentan nicht benutzt
    def getInitialConditions(self, old_Sim):
        Vdis_0 = old_Sim.V_dis[-1]
        Vd_0 = old_Sim.V_d[-1]
        Vc_0 = old_Sim.V_c[-1]
        phi32_0 = old_Sim.phi_32[-1]
        N_j_0 = old_Sim.N_j[:, -1]
        self.y0 = np.concatenate([np.array([Vdis_0, Vd_0, Vc_0, phi32_0])], N_j_0)

    def tau(self, h, d_32, ID, sigma, r_s_star):
        La_mod = (self.Sub.g * self.Sub.delta_rho / sigma) ** 0.6 * d_32 * h**0.2
        ra = 0.5 * d_32 * (1 - (1 - 4.7 / (4.7 + La_mod)) ** 0.5)
        if ID == "d":
            rf = d_32 * 0.3025 * (1 - (4.7 / (4.7 + La_mod))) ** 0.5
        else:
            rf = d_32 * 0.5239 * (1 - (4.7 / (4.7 + La_mod))) ** 0.5
        tau = (7.68* self.Sub.eta_c* (ra ** (7 / 3)/ (self.Sub.H_cd ** (1 / 6) * sigma ** (5 / 6) * rf * r_s_star)))

        return tau

    def henschke_input(self, V_dis, V_d, V_c, phi_32, sigma, r_s_star):

        D = self.Set.D
        dl = self.Set.dl
        dV = np.zeros_like(V_dis)
        tau_di = 9e9 * np.ones_like(V_dis)  # Koaleszenzzeit hoch gewählt, damit quasi keine stattfindet, wenn V_dis < 0
        tau_dd = tau_di
        h_dis = 0

        for i in range(len(V_dis)):
            if phi_32[i] < 0:
                phi_32[i] = self.Sub.phi_0 / 10

            if V_dis[i] > 0:
                h_c = hf.getHeight(V_c[i] / dl, D / 2)
                h_d = hf.getHeight(V_d[i] / dl, D / 2)
                Ay = 2 * dl * (2 * (D / 2) * h_c - h_c**2) ** 0.5
                h_dis = max(D - h_d - h_c , 0.0001)
                tau_di = self.tau(h_dis, phi_32[i], "I", sigma[i], r_s_star[i])
                tau_dd[i] = self.tau(h_dis, phi_32[i], "d", sigma[i], r_s_star[i])
                if (tau_di > 0):
                    dV[i] = 2 * Ay * phi_32[i] / (3 * tau_di * self.Sub.eps_p)
                else:
                    #dV[i] = 2 * Ay * phi_32[i] / (3 * 9e9 * self.Sub.eps_p)
                    dV[i] = 0
                if (tau_dd[i]==0):
                    tau_dd[i] = 9e9

        return dV, tau_dd

    def velocities(self):
        N_x = self.Set.N_x
        u_dis = (self.Sub.dV_ges / (np.pi * self.Set.D**2 / 4)) * np.ones(N_x)
        u_d = u_dis
        return u_dis, u_d
    
    def swarm_sedimenation_velocity(self, V_d, N_j):
        d_j = self.d_j
        v_sed = np.zeros((len(d_j), len(V_d)))
        eps = np.zeros(len(V_d))
        for i in range (len(V_d)): 
            if (V_d[i] > 0):
                eps[i] = np.sum(N_j[:,i] * (d_j**3) * (np.pi/6)) / V_d[i]
            else:
                eps[i] = self.Sub.eps_0
            for j in range(len(d_j)):
                v_sed[j][i] = ((self.Sub.g * self.Sub.delta_rho / (18 * self.Sub.eta_c))* (d_j[j] ** 2)* (1 - eps[i]))
        return v_sed

    def sedimentation_rate(self, V_d, N_j):
        d_j = self.d_j
        D = self.Set.D
        dl = self.Set.dl
        V_s = np.zeros(len(V_d))
        v_sed = self.swarm_sedimenation_velocity(V_d ,N_j)
        h_d = np.ones(len(V_d))
        for i in range(len(V_s)):
            if (V_d[i]>0):
                h_d[i] = hf.getHeight(V_d[i] / dl, D / 2)
                V_s[i] = np.sum((N_j[:,i] * v_sed[:,i] / h_d[i]) * (np.pi/6) * d_j**3)
            else:
                h_d[i] = h_d[i-1]
        return V_s
    
    def source_term_32(self, V_dis, V_d, phi_32, N_j):
        D = self.Set.D
        dl = self.Set.dl
        d_j = self.d_j
        S32=np.zeros(len(V_dis))
        v_sed = self.swarm_sedimenation_velocity(V_d ,N_j)
        h_d = np.ones(len(V_d))
        for i in range(len(V_dis)):
            if (V_dis[i]>0 and V_d[i]>0):
                h_d[i] = hf.getHeight(V_d[i] / dl, D / 2)
                S32[i] = ((np.pi/6) * phi_32[i] / (V_dis[i] * self.Sub.eps_p)) * np.sum(N_j[:,i] * v_sed[:,i] * d_j**2 * (d_j - phi_32[i]) / h_d[i])
            else:
                S32[i] = 0
        return S32
    
    def h_d_array(self, V_d):
        D = self.Set.D
        dl = self.Set.dl
        h_d_arr = np.zeros_like(V_d)
        for i in range(len(V_d)):
            if (V_d[i] > 0):
                h_d_arr[i] = hf.getHeight(V_d[i] / dl, D / 2)
            else:
                h_d_arr[i] = h_d_arr[i-1]
        return h_d_arr
    
    def simulate_ivp(self):

        start_time = time.time()
        y = []
        N_x = self.Set.N_x
        N_d = len(self.d_j)
        dl = self.Set.dl
        eps_p = self.Sub.eps_p
        sigma = self.Sub.sigma * np.ones(N_x)
        r_s_star = self.Sub.r_s_star * np.ones(N_x)
        D = self.Set.D
        N_j = np.zeros((N_d, N_x))
        dN_j_dt = np.zeros((N_d, N_x))


        def event(t, y):
            return np.min(y[:N_x])
        event.terminal = True   # event stops integration when V_dis<0

        def fun(t, y):

            V_dis = y[: N_x]
            V_d = y[N_x : 2*N_x]
            V_c = y[2*N_x : 3*N_x]
            phi_32 = y[3*N_x : 4*N_x]
            for j in range (N_d):
                N_j[j,:] = y[(j+4)*N_x : (j+5)*N_x]


            dV, tau_dd = self.henschke_input(V_dis, V_d, V_c, phi_32, sigma, r_s_star)
            u_dis, u_d = self.velocities()
            h_d = self.h_d_array(V_d)

            dVdis_dt = (u_dis / dl) * (np.roll(V_dis,1) - V_dis) + (1 / eps_p) * self.sedimentation_rate(V_d, N_j) - dV
            dVd_dt = (u_d / dl) * (np.roll(V_d,1) - V_d) - (1 / eps_p) * self.sedimentation_rate(V_d, N_j) + (1 - eps_p) * dV
            dVc_dt = -dVdis_dt - dVd_dt
            dphi32_dt = (u_dis / dl) * (np.roll(phi_32,1) - phi_32) + (phi_32 / (6 * tau_dd)) + self.source_term_32(V_dis, V_d, phi_32, N_j)
            for j in range(N_d):
                dN_j_dt[j,:] = (u_d / dl) * (np.roll(N_j[j,:],1) - N_j[j,:]) - N_j[j,:] * self.swarm_sedimenation_velocity(V_d, N_j)[j,:] / h_d

            dVdis_dt[0] = 0
            dVd_dt[0] = 0
            dVc_dt[0] = 0
            dphi32_dt[0] = 0
            for j in range(N_d):
                dN_j_dt[j,0] = 0
            
            # if (round(t,1) % 5 == 0):
            #     print(t)

            return np.concatenate([dVdis_dt, dVd_dt, dVc_dt, dphi32_dt, dN_j_dt.flatten()])
        

        # Lösung des GDGL-Systems

        self.sol = solve_ivp(fun, (0, self.Set.T), self.y0, t_eval=self.Set.t, method='RK45', events=event)
        

        y = self.sol.y
        self.V_dis = y[0 : N_x]
        self.V_d = y[N_x : 2*N_x]
        self.V_c = y[2*N_x : 3*N_x]
        self.phi_32 = y[3*N_x : 4*N_x]
        for j in range(N_d):
            self.N_j.append(list(y[(j+4)*N_x : (j+5)*N_x]))
        self.Set.t = self.sol.t

        end_time = time.time()
        u_dis, u_d = self.velocities()

        # Berechnung  der Extraktionseffizienz
        N_end = 0
        N_0 = 0
        for i in range(N_d):
            N_end += self.N_j[i][-1][-1]
            N_0 += self.N_j[i][0][0]
        E = 1 - (N_end/N_0)
        H = getHeightArray(self.V_d[:,-1] / dl, D/2)

        # print('\nSimulation ended successfully after: ', round(end_time-start_time,1), "s", "\nu_dis = u_d = ", u_dis[0], "[m/s]","\nExtraction efficiency",round(100*E,3), " %", "\nHeight of heavy phase: ", H[-1])
        # print('\nN_j(x=0,t=0)= ', N_0,'\nN_j(x=L, t_end)= ', N_end)

        h_d = getHeightArray(self.V_d[:, len(self.Set.t) - 1]/self.Set.dl, self.Set.D/2)
        h_d_dis = getHeightArray((self.V_d[:, len(self.Set.t) - 1] + self.V_dis[:, len(self.Set.t) - 1])/self.Set.dl, self.Set.D/2)
        h_dis = max(h_d_dis) - min(h_d)
        self.H_DPZ = h_dis
        # print('Height of the DPZ at the end of the simulation: ', 1000 * h_dis , ' mm')
        a = np.where(np.abs(h_d_dis - h_d) < 1e-3)[0][0] if np.any(np.abs(h_d_dis - h_d) < 1e-3) else -1
        self.L_DPZ = a * self.Set.dl
        # print('Length of the DPZ at the end of the simulation: ', 1000 * a * self.Set.dl, ' mm')
        self.V_dis_total = np.sum(self.V_dis[:,-1])


        print('dV_ges= ', self.Sub.dV_ges, 'phi_32,0= ', self.Sub.phi_0, 'V_dis= ', self.V_dis_total)
        print('')
    
    def plot_solution(self, N_i, N_t, ID):

        N_d = len(self.d_j)
        t = self.sol.t
        dl = self.Set.dl
        D = self.Set.D
        x = self.Set.x
        N_x = self.Set.N_x

        # Funktion zur Bestimmung des Gleichgewichtzistand
        
        # def steady_state_det(v):
        #     for i in range(len(v)):
        #         if ((np.mean(abs(abs(v[i:]) - abs(np.mean(v[i:]))))) < 1e-3 * max(v)):
        #             b = i
        #             break
        #     return b

        #Berechnung der Höhe 

        # b_dis = steady_state_det(self.V_dis[:,t_w])
        # b_d = steady_state_det(self.V_d[:,t_w])
        # b_c = steady_state_det(self.V_c[:,t_w])
        # b = max(b_dis, b_c, b_d)
        # A_dis = self.V_dis[0:b, t_w] / dl
        # A_d = self.V_d[0:b, t_w] / dl
        # A_c = self.V_c[0:b, t_w] / dl
        
        A_d = self.V_d[:, N_t] / dl
        A_c = self.V_c[:, N_t] / dl
        h_d = np.zeros(len(A_d))
        h_c = np.zeros(len(A_c))
        for i in range(len(A_d)):
            h_d[i] = hf.getHeight(A_d[i], D/2)
            h_c[i] = hf.getHeight(A_c[i], D/2)
        
        # Plotten der Volumina im ganzen Zeitraum im N_i Element
        if (ID == "vol"):
            plt.figure(1)
            plt.plot(t, self.V_dis[N_i,:], label='V_dis')
            plt.plot(t, self.V_d[N_i,:], label='V_d')
            plt.plot(t, self.V_c[N_i,:], label='V_c')
            plt.title("Volumina, Längenelement: " + str(N_i))
            plt.xlabel('t')
            plt.ylabel('[m^3]')
            plt.legend()
            plt.grid()
            plt.show()

        # Plotten des Sauterdurchmessers im ganzen Zeitraum im N_i Element
        if (ID == "phi32"):
            plt.figure(2)
            plt.plot(t,1000*self.phi_32[N_i,:])
            plt.title("Phi_32, Längenelement: " + str(N_i))
            plt.xlabel('t')
            plt.ylabel('[mm]')
            plt.grid()
            plt.show()

        # Plotten der Tropfenanzahle im ganzen Zeitraum im N_i Element
        if (ID == "Nj"):
            plt.figure(3)
            for j in range(N_d):
                plt.plot(t,self.N_j[j][N_i][:], label=(f'N_{j+1}, d_{j+1}= ' + str(round(1000*self.d_j[j],1)) + ' [mm]'))
            plt.xlabel('t')
            plt.ylabel('Tropfen')
            plt.title("Tropfenanzahl, Längenelement: " + str(N_i))
            plt.legend()
            plt.grid()
            plt.show()

        # Plotten der Phasenhöhe über die gesamte Einlaufbereichlänge an Zeitpunkt t_w
        if (ID == "heights"):
            plt.figure(4)
            plt.plot(1000*x,1000*(D-h_c), label='h_c')
            plt.plot(1000*x,1000*(h_d), label='h_d')
            plt.title("Phasenhöhe, t["+str(N_t)+"]= " + str(round(t[N_t],1)) + "s")
            plt.xlabel('Input area [mm]')
            plt.ylabel('Height [mm]')
            plt.legend()
            plt.grid()

            plt.figure(5)
            plt.plot(np.linspace(0, N_x, N_x),1000*(D-h_c), label='h_c')
            plt.plot(np.linspace(0, N_x, N_x),1000*(h_d), label='h_d')
            plt.title("Phasenhöhe, t["+str(N_t)+"]= " + str(round(t[N_t],1)) + "s")
            plt.xlabel('Längeelement [N_i]')
            plt.ylabel('Height [mm]')
            plt.legend()
            plt.grid()

            plt.show()



    ########################################################################################################################################################################################




    def plot_anim(self, plots): # consists of keys for plotting functions
        from helper_functions import getHeightArray

        # Figure erzeugen
        x = []
        y = []

        if (len(plots) > 1):
            fig, axes = plt.subplots(len(plots), 1, figsize=(9, 6))
            if ("N_j" in plots):
                for i in range(len(plots)):
                    if i==0 :
                        axes[i] = fig.add_subplot(len(plots),1,1, projection='3d')
                    else:  
                        axes[i].plot(x, y)
            else :
                for i in range(len(plots)):
                        axes[i].plot(x, y)


        if plots == ['N_j'] :
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

        if (plots == ['heights'] or plots == ['hold_up'] or plots == ['velo'] or plots == ['phi_32'] or plots == ['tau']):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            #fig, ax = plt.subplots()
            #ax.plot(x, y)

        # Variablen definieren
        V_dis = self.V_dis
        V_d = self.V_d
        V_c = self.V_c
        u_dis = self.u_dis
        u_d = self.u_d
        u_c = self.u_c
        dl = self.Set.dl
        D = self.Set.D
        t = self.Set.t
        x = self.Set.x
        N_d = len(self.d_j)
        h_p_star = self.Sub.h_p_star
        phi_32 = self.phi_32
        light_in_heavy = self.Sub.light_in_heavy
        N_j = self.N_j
        d_j = self.d_j
        eps = np.zeros((len(V_dis[:,0]), len(V_dis[0,:])))

        N = np.array(N_j)
        for i in range(len(V_dis[:,0])):
            for j in range(len(V_dis[0,:])):
                eps[i][j] = np.sum(N[:,i,j] * (d_j**3) * (np.pi/6)) / V_d[i,j]

        x*=1000


        ################################################ Plotting-Funktionen (teils sehr kompliziert geschrieben)

        # animierter Plot ausgewählter Variablen (funktioniert, indem eine Liste mit keys (strings) übergeben wird)
        def plot_anim_step(key, ax, frame, i):
            ax.cla()

            if i == 0:
                ax.set_title('Time = {:.2f}'.format(t[frame]) + 's')

            if key == 'velo':

                ax.plot(x, u_dis[:, frame], color='r', label='dpz')
                ax.plot(x, u_d[:, frame], color='b', label='disp phase')
                ax.plot(x, u_c[:, frame], color='g', label='conti phase')
                ax.plot(x, self.u0 * np.ones_like(u_dis[:, frame]), linestyle='--', color='black', label='u0')

                ax.set_xlabel('x in mm')
                ax.set_ylabel('Geschwindigkeit in m/s')
                ax.set_xlim(0, x[-1])
                ax.set_ylim(bottom=0)

            if key == 'phi_32':
                idx_no_dis = self.Set.N_x
                if min(V_dis[:,frame]) < 1e-8:
                    idx_no_dis = np.where(V_dis[:, frame] < 1e-8)[0][0]
                ax.plot(x[:idx_no_dis], phi_32[:idx_no_dis, frame] * 1000, label='phi_32', color='b')

                idx_no_dis = self.Set.N_x
                if min(V_dis[:, 0]) < 1e-8:
                    idx_no_dis = np.where(V_dis[:, 0] < 1e-8)[0][0]
                ax.plot(x[:idx_no_dis], phi_32[:idx_no_dis, 0] * 1000, label='phi_32 at t = 0', color='g', linestyle='--')

                idx_no_dis = self.Set.N_x
                if min(V_dis[:, len(t) - 2]) < 1e-8:
                    idx_no_dis = np.where(V_dis[:, len(t) - 2] < 1e-8)[0][0]
                ax.plot(x[:idx_no_dis], phi_32[:idx_no_dis, len(t) - 2] * 1000, label='phi_32 at t = {:.2f}'.format(t[len(t) - 1]),
                             color='r', linestyle='--')

                ax.set_xlabel('x in mm')
                ax.set_ylabel('Sauter mean diameter in mm')
                ax.set_ylim(0, np.ceil(1000 * np.max(phi_32)))
                ax.set_xlim(0, x[-1])

            if key == 'heights':

                V_tot = V_dis + V_d + V_c

                if light_in_heavy:
                    ax.plot(x, 1000 * (D - getHeightArray(V_c[:, 0] / dl, D / 2)), color='r', linestyle=':',
                                 label='Interface c, dis; t = 0')
                    ax.plot(x, 1000 * getHeightArray(V_d[:, 0] / dl, D / 2), color='g', linestyle=':',
                                 label='Interface dis, d; t = 0')

                    ax.plot(x, 1000 * (D - getHeightArray(V_c[:, len(t) - 1] / dl, D / 2)), color='r',
                                 linestyle='--',
                                 label='Interface c, dis; t = {:.2f}'.format(t[len(t) - 1]))
                    ax.plot(x, 1000 * getHeightArray(V_d[:, len(t) - 1] / dl, D / 2), color='g',
                                 linestyle='--',
                                 label='Interface dis, d; t = {:.2f}'.format(t[len(t) - 1]))

                    ax.plot(x, 1000 * (D - getHeightArray(V_c[:, frame] / dl, D / 2)), color='r',
                                 label='Interface c, dis')
                    ax.plot(x, 1000 * getHeightArray(V_d[:, frame] / dl, D / 2), color='g',
                                 label='Interface dis, d')  
                else:
                    ax.plot(x, 1000 * getHeightArray(V_d[:, 0] / dl, D / 2), color='r', linestyle=':',
                                 label='Interface d, dis; t = 0')
                    ax.plot(x, 1000 * ( D - getHeightArray(V_c[:, 0] / dl, D / 2)), color='g', linestyle=':',
                                 label='Interface dis, c; t = 0')

                    ax.plot(x, 1000 * getHeightArray(V_d[:, len(t) - 2] / dl, D / 2), color='r',
                                 linestyle='--',
                                 label='Interface d, dis; t = {:.2f}'.format(t[len(t) - 2]))
                    ax.plot(x, 1000 * ( D - getHeightArray(V_c[:, len(t) - 2] / dl, D / 2)), color='g',
                                 linestyle='--',
                                 label='Interface dis, c; t = {:.2f}'.format(t[len(t) - 2]))

                    ax.plot(x, 1000 * getHeightArray(V_d[:, frame] / dl, D / 2), color='r',
                                 label='Interface d, dis')
                    ax.plot(x, 1000 * ( D - getHeightArray(V_c[:, frame] / dl, D / 2)), color='g',
                                 label='Interface dis, c')

                ax.plot(x, 1000 * getHeightArray(V_tot[:, frame] / dl, D / 2), color='b', label='h_tot')

                ax.set_xlabel('x in mm')
                ax.set_ylabel('Height in mm')
                ax.set_xlim(0, x[-1])

            if key == 'tau':

                hp = D - getHeightArray(V_d[:, frame] / dl, D / 2) - getHeightArray(V_c[:, frame] / dl, D / 2)

                if np.where(hp[:-1] == hp[1:])[0].size > 0:
                    last_idx = np.where(hp[:-1] == hp[1:])[0][0]
                    for k in range(last_idx, len(x)):
                        hp[k] = hp[k-1]

                tau_di = np.zeros_like(x)
                tau_dd = np.zeros_like(x)
                for k in range(len(x)):
                    if hp[k] < D / 1e5: # Obergrenze, damit keine unendlich großen Koaleszenzzeiten auftreten
                        hp[k] = D / 1e5
                    tau_di[k] = self.tau(hp[k], phi_32[k, frame], 'i')
                    tau_dd[k] = self.tau(h_p_star*hp[k], phi_32[k, frame], 'd')

                ax.plot(x, tau_di, label='tau_di', color='b')
                ax.plot(x, tau_dd, label='tau_dd', color='r')

                ax.set_xlabel('x in mm')
                ax.set_ylabel('Koaleszenzzeit in s')
                ax.set_xlim(0, x[-1])
                ax.set_ylim(0, 10)

            if key == "hold_up" :

                ax.plot(x, eps[:,0], color='g', linestyle='--', label='Hold-up at = 0')
                ax.plot(x, eps[:,len(t) - 2], color='r',linestyle='--', label='Hold-up at t = {:.2f}'.format(t[len(t) - 1]))
                ax.plot(x, eps[:,frame], color='b',label='Hold-up')

                ax.set_xlabel('x in mm')
                ax.set_ylabel('Hold-up')
                ax.set_xlim(0, x[-1])

            
            if key == 'N_j' :

                # Convert list to numpy arrays
                y_length = x
                x_length = 1000*d_j
                z = np.array(N_j)


                # Construct arrays for the anchor positions of the bars
                xpos, ypos = np.meshgrid(x_length, y_length, indexing="ij")
                xpos = xpos.ravel()
                ypos = ypos.ravel()
                zpos = np.zeros_like(xpos)

                # Construct arrays with the dimensions for the bars
                dx = dy = 0.5 * np.ones_like(zpos)
                dz = z[:,:,frame].ravel()

                 # Plot the 3D histogram
                ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average', label='Tropfenanzahl(t)', color='magenta')

                # Set labels and title
                ax.set_xlabel('Droplet classes diameter [mm]')
                ax.set_ylabel('Separator length [mm]')
                ax.set_zlabel('Number of droplets')
                ax.view_init(elev=30, azim=45)
                ax.set_title('Time = {:.2f}'.format(t[frame]) + 's, '+' Frame = {:.2f}'.format(frame))

        def update(frame):

            if len(plots) > 1:
                for i in range(len(plots)):
                    plot_anim_step(plots[i], axes[i], frame, i)
                    axes[i].legend(loc='upper left', bbox_to_anchor=(1, 1))
            else:
                plot_anim_step(plots[0], ax, frame, 0)
                ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
            plt.tight_layout()

        anim = FuncAnimation(plt.gcf(), update, frames=range(len(t)), interval=10)


        plt.show()

        x /= 1000

    #################################################################################################



        
            



           



           