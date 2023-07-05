import numpy as np
from pygsti.baseobjs import Basis

np.set_printoptions(precision=1, linewidth=1000)

import pygsti
from pygsti.extras import idletomography as idt

n_qubits = 4
gates = ["Gi","Gx","Gy","Gcnot"]
max_lengths = [1,2,4,8]
pspec = pygsti.processors.QubitProcessorSpec(n_qubits, gates, geometry='line', nonstd_gate_unitaries={():1})
mdl_target = pygsti.models.create_crosstalk_free_model(pspec)
paulidicts = idt.determine_paulidicts(mdl_target)
idle_experiments = idt.make_idle_tomography_list(n_qubits, max_lengths, paulidicts, maxweight=1)
print(len(idle_experiments), "idle tomography experiments for %d qubits" % n_qubits)


from pygsti.baseobjs import Label
updated_ckt_list= []
for ckt in idle_experiments:
    new_ckt= ckt.copy(editable=True)
    for i,lbl in enumerate(ckt):
        print(i,lbl)
        if lbl == Label(()):
            new_ckt[i] = [Label(('Gi',i)) for i in range(n_qubits)]
            #new_ckt[i] = Label(('Gi', ))
    print(new_ckt)
    updated_ckt_list.append(new_ckt)

mdl_datagen = pygsti.models.create_crosstalk_free_model(pspec,
                                                        lindblad_error_coeffs={'Gi': {"SX": 0.01}})
# Error models! Random with right CP constraints from Taxonomy paper
ds = pygsti.data.simulate_data(mdl_datagen, updated_ckt_list,100000, seed=8675309)

results = idt.do_idle_tomography(n_qubits, ds, max_lengths, paulidicts, include_hamiltonian = True, include_stochastic=True, include_affine = True, maxweight=1, idle_string='[Gi:0Gi:1Gi:2Gi:3]')
idt.create_idletomography_report(results,"../IDTTestReport","Test idle tomography example report", auto_open=True)

results.error_list
ws = pygsti.report.Workspace()
#ws.init_notebook_mode(autodisplay=True)
print(results)
