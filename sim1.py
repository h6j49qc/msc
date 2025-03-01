import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define reaction rate constants (assumed reasonable values)
k_plus1 = 0.1  # Activation rate of BCR-ABL (s⁻¹)
k_minus1 = 0.02  # Inactivation rate of BCR-ABL (s⁻¹)
k_on1 = 1.0e6  # ATP binding rate (M⁻¹s⁻¹)
k_off1 = 0.1  # ATP unbinding rate (s⁻¹)
k_on2 = 1.0e6  # Substrate binding rate (M⁻¹s⁻¹)
k_off2 = 0.1  # Substrate unbinding rate (s⁻¹)
k_cat = 0.5  # Catalytic phosphorylation rate (s⁻¹)
k_on3 = 1.0e6  # Imatinib binding rate (M⁻¹s⁻¹)
k_off3 = 0.001  # Imatinib unbinding rate (s⁻¹)

# Initial concentrations (M)
BcrAbl_active_0 = 5e-6  # 5 μM
BcrAbl_inactive_0 = 2e-6  # 2 μM
ATP_0 = 1e-3  # 1 mM (intracellular ATP concentration)
Substrate_0 = 1e-5  # 10 μM
Phospho_Substrate_0 = 0  # Initially no phosphorylated substrate
Imatinib_0 = 1e-6  # 1 μM
BcrAbl_ATP_0 = 0  # No bound ATP at start
BcrAbl_Substrate_0 = 0  # No bound substrate at start
BcrAbl_Imatinib_0 = 0  # No Imatinib-bound BCR-ABL at start


# Define ODE system
def bcr_abl_kinetics(t, y):
    BcrAbl_active, BcrAbl_inactive, BcrAbl_ATP, BcrAbl_Substrate, Phospho_Substrate, BcrAbl_Imatinib = y

    dBcrAbl_active = k_minus1 * BcrAbl_inactive - k_plus1 * BcrAbl_active \
                     + k_off1 * BcrAbl_ATP - k_on1 * BcrAbl_active * ATP_0 \
                     + k_off2 * BcrAbl_Substrate - k_on2 * BcrAbl_active * Substrate_0

    dBcrAbl_inactive = k_plus1 * BcrAbl_active - k_minus1 * BcrAbl_inactive \
                       + k_off3 * BcrAbl_Imatinib - k_on3 * BcrAbl_inactive * Imatinib_0

    dBcrAbl_ATP = k_on1 * BcrAbl_active * ATP_0 - k_off1 * BcrAbl_ATP

    dBcrAbl_Substrate = k_on2 * BcrAbl_active * Substrate_0 - k_off2 * BcrAbl_Substrate - k_cat * BcrAbl_Substrate

    dPhospho_Substrate = k_cat * BcrAbl_Substrate

    dBcrAbl_Imatinib = k_on3 * BcrAbl_inactive * Imatinib_0 - k_off3 * BcrAbl_Imatinib

    return [dBcrAbl_active, dBcrAbl_inactive, dBcrAbl_ATP, dBcrAbl_Substrate, dPhospho_Substrate, dBcrAbl_Imatinib]

t_end=100000
# Time span for simulation (0 to 1000s)
t_span = (0, 100000)
t_eval = np.linspace(0, 10000, 500)

# Initial conditions
y0 = [BcrAbl_active_0, BcrAbl_inactive_0, BcrAbl_ATP_0, BcrAbl_Substrate_0, Phospho_Substrate_0, BcrAbl_Imatinib_0]

# Solve ODEs
# sol = solve_ivp(bcr_abl_kinetics, t_span, y0, t_eval=t_eval, method='RK45')
sol = solve_ivp(bcr_abl_kinetics, t_span, y0, t_eval=t_eval, method='LSODA')

# Extract solutions
t = sol.t
BcrAbl_active, BcrAbl_inactive, BcrAbl_ATP, BcrAbl_Substrate, Phospho_Substrate, BcrAbl_Imatinib = sol.y

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(t, BcrAbl_active, label="Bcr-Abl (Active)", linewidth=2)
plt.plot(t, BcrAbl_inactive, label="Bcr-Abl (Inactive)", linewidth=2)
plt.plot(t, Phospho_Substrate, label="Phosphorylated Substrate", linewidth=2)
plt.plot(t, BcrAbl_Imatinib, label="Bcr-Abl (Imatinib-bound)", linewidth=2, linestyle="--")
plt.plot(t, BcrAbl_ATP, label="BcrAbl_ATP", linewidth=2, linestyle="--")
plt.plot(t, BcrAbl_Substrate, label="BcrAbl_Substrate", linewidth=2, linestyle="--")

plt.xlabel("Time (s)")
plt.ylabel("Concentration (M)")
plt.title("Kinetic Simulation of BCR-ABL Dynamics")
plt.legend()
plt.grid()
plt.show()