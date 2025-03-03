import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.ticker as mticker

import time

from scipy.ndimage import minimum

# Define reaction rate constants (assumed reasonable values)
k_plus1 = 0.1  # Activation rate of BCR-ABL (s⁻¹)
k_minus1 = 0.0851  # Inactivation rate of BCR-ABL (s⁻¹)
k_on1 = 1e2  # ATP binding rate (M⁻¹s⁻¹)
k_off1 = 0.1  # ATP unbinding rate (s⁻¹)
k_on3 = 1e2  # Imatinib binding rate (M⁻¹s⁻¹)
k_off3 = 0.00  # Imatinib unbinding rate (s⁻¹)
Kintake = 3e-10 # Imatinib intake Ms⁻¹ (dosage 600mg or approx. 1.2 millimole daily)

k_on2 = 1.0e6  # Substrate binding rate (M⁻¹s⁻¹)
k_off2 = 0.1  # Substrate unbinding rate (s⁻¹)
k_cat = 0.5  # Catalytic phosphorylation rate (s⁻¹)
k_prod1 = 1e-8
k_deg = 1e-1


# Initial concentrations (M)
BcrAbl_active_0 = 5e-6  # 5 μM
BcrAbl_inactive_0 = 2e-6  # 2 μM
ATP_0 = 1.1e-3  # 1 mM (intracellular ATP concentration)
BcrAbl_ATP_0 = 0  # No bound ATP at start
Imatinib_0 = 1e-6  # 1 μM
BcrAbl_Imatinib_0 = 0  # No Imatinib-bound BCR-ABL at start

Substrate_0 = 1e-6  # 10 μM
BcrAbl_Substrate_0 = 0  # No bound substrate at start
Phospho_Substrate_0 = 0  # Initially no phosphorylated substrate


# Define ODE system
def bcr_abl_kinetics(t, y):

    # need to declare iter as global because we change it here, so python otherwise assumes it is local
    global iter
    BcrAbl_active, BcrAbl_inactive, BcrAbl_ATP, Imatinib, BcrAbl_Imatinib, Substrate, BcrAbl_Substrate, Phospho_Substrate = y

    dBcrAbl_active = k_plus1 * BcrAbl_inactive - k_minus1 * BcrAbl_active - k_on1 * ATP_0 * BcrAbl_active + k_off1*BcrAbl_ATP
    dBcrAbl_inactive = k_minus1 * BcrAbl_active - k_plus1 * BcrAbl_inactive - k_on3 * BcrAbl_inactive * Imatinib + k_prod1

    dBcrAbl_ATP = k_on1 * BcrAbl_active * ATP_0 - k_off1 * BcrAbl_ATP + k_off2 * BcrAbl_Substrate

    dImatinib = Kintake - k_on3 * Imatinib * BcrAbl_inactive + k_off3 * BcrAbl_Imatinib
    dBcrAbl_Imatinib = k_on3 * BcrAbl_inactive * Imatinib - k_off3 * BcrAbl_Imatinib - k_deg * BcrAbl_Imatinib

    dSubstrate =  k_off2 * BcrAbl_Substrate - k_on2 * Substrate * dBcrAbl_ATP
    # try treating this as a constant as we have no replenishment process
#    dSubstrate = 0
    dBcrAbl_Substrate = k_on2 * Substrate * dBcrAbl_ATP - k_off2 * BcrAbl_Substrate - k_cat * BcrAbl_Substrate
    dPhospho_Substrate = k_cat * BcrAbl_Substrate

    if debug>0:
        if (iter%10==0):
            print("+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-----------+")
            print("|   Time     |      dBcrAbl_active         |     BcrAbl_inactiv          |        BcrAbl_ATP           |        Imatinib             |      BcrAbl_Imatinib        |        Substrate            |       BcrAbl_Substrate      |      Phospho_Substrate      | Iteration |")
            print("-------------------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------|")
            print("|            |  Value           Delta      |   Value        Delta        |   Value       Delta         |     Value       Delta       |   Value       Delta         |     Value       Delta       |   Value       Delta         |   Value       Delta         |           |")
        print("| %10.3f | %13.3e %13.3e | %13.3e %13.3e | %13.3e %13.3e | %13.3e %13.3e | %13.3e %13.3e | %13.3e %13.3e | %13.3e %13.3e | %13.3e %13.3e |  %6i   |" %
              (t, BcrAbl_active, dBcrAbl_active, BcrAbl_inactive, dBcrAbl_inactive, BcrAbl_ATP, dBcrAbl_ATP, Imatinib, dImatinib,
                  BcrAbl_Imatinib, dBcrAbl_Imatinib, Substrate, dSubstrate, BcrAbl_Substrate, dBcrAbl_Substrate, Phospho_Substrate, dPhospho_Substrate, iter))
    iter+=1

    return [dBcrAbl_active, dBcrAbl_inactive, dBcrAbl_ATP, dImatinib, dBcrAbl_Imatinib, dSubstrate, dBcrAbl_Substrate, dPhospho_Substrate]


def drawPlot(xValues, y1Values, y1Labels, y2Values, y2Labels, title):

    # determine minima and maxima for scaling
    minY1, maxY1, minY2, maxY2 = 1e20, 0.0, 1e20, 0.0
    for y1 in y1Values:
        minY1=min(minY1, y1.min())
        maxY1=max(maxY1, y1.max())
    for y2 in y2Values:
        minY2=min(minY2, y2.min())
        maxY2=max(maxY2, y2.max())

    if debug!=0:
        print("y1 min,max = %10.2e, %10.2e"%(minY1, maxY1))
    # Use a secondary y-axis, set x-axis gridlines on
    fig, ax1 = plt.subplots(figsize=(10, 6))
    plt.grid(axis='x')

    ax1.set_xlim(0, xValues.max())
    ax1.set_ylim(minY1, maxY1*1.05)
#    ax1.set_ylim(2e-6, maxY1)

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Concentration (M)")
    i=0
    for y1 in y1Values:
        print(y1Labels[i])
        if (i==0):
            ax1.plot(t, y1, label=y1Labels[i], linestyle='dotted')
        else:
            ax1.plot(t, y1, label=y1Labels[i], linestyle='solid', lw=1)
        i+=1
    ax1.legend(loc='lower center')

    if len(y2Values)>0:
        ax2 = ax1.twinx()
        ax2.set_ylim(minY2, maxY2)
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("Proportion Active Bcr-Abl", color='green')
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x * 100:.0f}%'))
        i=0
        for y2 in y2Values:
            ax2.plot(t, y2, label=y2Labels[i], linestyle='dashed', lw=1)
            i+=1
        ax2.legend(loc='upper right')

    plt.title(title)
    plt.grid()
    plt.grid(axis='x')




# Only the last 3 "Radau","BDF","LSODA" seem to find stable solutions here,
# unless max_step is reduced from default. See below for documentation
algorithms=["RK45","RK23","DOP853","Radau","BDF","LSODA"]
algo=5

# a few global variables
iter=0
use_log_scale=1
debug=0

# Time span for simulation (0 to 100s)
t_end=80
t_span = (0, t_end)
t_eval = np.linspace(0, t_end, 500)

# Initial conditions
y0 = [BcrAbl_active_0, BcrAbl_inactive_0, BcrAbl_ATP_0, Imatinib_0, BcrAbl_Imatinib_0, Substrate_0, BcrAbl_Substrate_0, Phospho_Substrate_0]

y1Values=[]
y1Labels=[]
y2Values=[]
y2Labels=[]

k_plus1 = 0.1  # Activation rate of BCR-ABL (s⁻¹)
k_minus1 = 0.0851  # Inactivation rate of BCR-ABL (s⁻¹)
Imatinib_0 = 1e-6  # 1 μM
iter = 0
y0 = [BcrAbl_active_0, BcrAbl_inactive_0, BcrAbl_ATP_0, Imatinib_0, BcrAbl_Imatinib_0, Substrate_0, BcrAbl_Substrate_0, Phospho_Substrate_0]
sol = solve_ivp(bcr_abl_kinetics, t_span, y0, t_eval=t_eval, dense_output=True, method=algorithms[algo], max_step=t_end / 2000)
t = sol.t
BcrAbl_active, BcrAbl_inactive, BcrAbl_ATP, Imatinib, BcrAbl_Imatinib, Substrate, BcrAbl_Substrate, Phospho_Substrate = sol.y
y1Values.append(BcrAbl_Imatinib)
y1Labels.append("Wild BcrAbl (Imatinib = %iμM)" % (Imatinib_0*1e6))
y2Values.append(BcrAbl_active / (BcrAbl_active + BcrAbl_inactive))
y2Labels.append("Wild BcrAbl, proportion [BcrAbl_active]")

k_plus1 = 0.1  # Activation rate of BCR-ABL (s⁻¹)
k_minus1 = 0.014  # Inactivation rate of BCR-ABL (s⁻¹)

for Imatinib_0 in [1e-6, 2e-6, 3e-6, 4e-6, 5e-6]:
    iter = 0
    y0 = [BcrAbl_active_0, BcrAbl_inactive_0, BcrAbl_ATP_0, Imatinib_0, BcrAbl_Imatinib_0, Substrate_0, BcrAbl_Substrate_0, Phospho_Substrate_0]
    sol = solve_ivp(bcr_abl_kinetics, t_span, y0, t_eval=t_eval, dense_output=True, method=algorithms[algo], max_step=t_end / 2000)
    BcrAbl_active, BcrAbl_inactive, BcrAbl_ATP, Imatinib, BcrAbl_Imatinib, Substrate, BcrAbl_Substrate, Phospho_Substrate = sol.y
    y1Values.append(BcrAbl_Imatinib)
    y1Labels.append("Mutant BcrAbl (Imatinib = %iμM)" % (Imatinib_0*1e6))
    if len(y2Values)<2:
        y2Values.append(BcrAbl_active/(BcrAbl_active+BcrAbl_inactive))
        y2Labels.append("Mutant BcrAbl, proportion [BcrAbl_active]")

drawPlot(t, y1Values, y1Labels, y2Values, y2Labels, "[BcrAbl_Imatinib] by Dosage, Time Dependemcy")


'''
for k_on3 in [1e1, 5e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7]:
    iter = 0
    sol = solve_ivp(bcr_abl_kinetics, t_span, y0, t_eval=t_eval, dense_output=True, method=algorithms[algo], max_step=t_end/2000)
    t=sol.t
    BcrAbl_active, BcrAbl_inactive, BcrAbl_ATP, Imatinib, BcrAbl_Imatinib, Substrate, BcrAbl_Substrate, Phospho_Substrate = sol.y
#    print(BcrAbl_active)
    y1Values.append(BcrAbl_active)
    y1Labels.append("BcrAbl_active k_on3=%10.2e"%(k_on3))
drawPlot(t, y1Values, y1Labels, y2Values, y2Labels, "BcrAbl_active k_on3 dependency")
'''

plt.show()
