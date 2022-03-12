"""
建立 converter simulation 的自動化
使列出他的規格
Vo=5
Vi=3.3
Vd=0.5
Io=1.667
freq=300E3
delta_Vi=30E-3
delta_Vo=50E-3
L_coff=0.4#delata I(L) 佔 I(L)的比例
power_efficenicy=1.0
可以求得理論值 的輸出輸入電容 電感值
再帶進入 ltspice 模擬 出 實際的情形
根據使用者 使用的不同的 frequency 可以模擬出對於電路的影響
影響指標 power efficenicy ripple viltage 範圍 switching loss 的影響

"""

from PyLTSpice.LTSpiceBatch import SimCommander
import re
import os
import shutil
from itertools import permutations
from os import listdir
from os.path import isfile, isdir, join

LTC = SimCommander(r"C:\Users\user\Documents\LTspiceXVII\file\Montecarlo\LTC1871-7_F09.asc")
# set default arguments
LTC.set_parameters(res=0, cap=100e-6)
LTC.set_component_value('R2', '2k')
LTC.set_component_value('R1', '4k')
LTC.set_element_model('V3', "SINE(0 1 3k 0 0 0)")
# define simulation
LTC.add_instructions(
    "; Simulation settings",
    ".param run = 0"
)

for opamp in ('AD712', 'AD820'):
    LTC.set_element_model('XU1', opamp)
    for supply_voltage in (5, 10, 15):
        LTC.set_component_value('V1', supply_voltage)
        LTC.set_component_value('V2', -supply_voltage)
        # overriding he automatic netlist naming
        run_netlist_file = "{}_{}_{}.net".format(LTC.circuit_radic, opamp, supply_voltage)
        LTC.run(run_filename=run_netlist_file, callback=processing_data)


LTC.reset_netlist()
LTC.add_instructions(
    "; Simulation settings",
    ".ac dec 30 10 1Meg",
    ".meas AC Gain MAX mag(V(out)) ; find the peak response and call it ""Gain""",
    ".meas AC Fcut TRIG mag(V(out))=Gain/sqrt(2) FALL=last"
)

LTC.run()
LTC.wait_completion()

# Sim Statistics
print('Successful/Total Simulations: ' + str(LTC.okSim) + '/' + str(LTC.runno))