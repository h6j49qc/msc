import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define reaction rate constants (assumed reasonable values)
k_plus1 = 0.1  # Activation rate of BCR-ABL (s⁻¹)
k_minus1 = 0.02  # Inactivation rate of BCR-ABL (s⁻¹)
k_on1 = 1e2  # ATP binding rate (M⁻¹s⁻¹)
k_off1 = 0.1  # ATP unbinding rate (s⁻¹)

# Initial concentrations (M)
BcrAbl_active_0 = 5e-6  # 5 μM
BcrAbl_inactive_0 = 2e-6  # 2 μM
ATP_0 = 1.1e-3  # 1 mM (intracellular ATP concentration)
BcrAbl_ATP_0 = 0  # No bound ATP at start


# Define ODE system
def bcr_abl_kinetics(t, y):
    BcrAbl_active, BcrAbl_inactive, BcrAbl_ATP = y

    dBcrAbl_active = k_plus1 * BcrAbl_inactive - k_minus1 * BcrAbl_active - k_on1 * ATP_0 * BcrAbl_active + k_off1*BcrAbl_ATP
    dBcrAbl_inactive = k_minus1 * BcrAbl_active - k_plus1 * BcrAbl_inactive
    dBcrAbl_ATP = k_on1 * BcrAbl_active * ATP_0 - k_off1 * BcrAbl_ATP

    return [dBcrAbl_active, dBcrAbl_inactive, dBcrAbl_ATP]


t_end=100
# Time span for simulation (0 to 1000s)
t_span = (0, t_end)
t_eval = np.linspace(0, t_end, 500)

# Initial conditions
y0 = [BcrAbl_active_0, BcrAbl_inactive_0, BcrAbl_ATP_0]

plt.figure(figsize=(10, 6))


sol = solve_ivp(bcr_abl_kinetics, t_span, y0, t_eval=t_eval, method='LSODA')
# sol = solve_ivp(bcr_abl_kinetics, t_span, y0, t_eval=t_eval, method='RK45')

t = sol.t
BcrAbl_active, BcrAbl_inactive, dBcrAbl_ATP = sol.y

plt.plot(t, BcrAbl_active, label="Bcr-Abl (Active) k-1=%5.2f" % (k_minus1))
plt.plot(t, BcrAbl_inactive, label="Bcr-Abl (Inactive) k-1=%5.2f" % (k_minus1))
plt.plot(t, dBcrAbl_ATP, label="dBcrAbl_ATP k-1=%5.2f" % (k_minus1))
# plt.yscale("log")

plt.xlabel("Time (s)")
plt.ylabel("Concentration (M)")
plt.title("Kinetic Simulation of BCR-ABL Dynamics K+1=0.1")
plt.legend()
plt.grid()
plt.show()