import numpy as np
from pygsti.baseobjs import Basis

np.set_printoptions(precision=1, linewidth=1000)

import pygsti
from pygsti.extras import idletomography as idt

mode_flag = "us"
if "s" in mode_flag:
    from pygsti.extras.idletomography.pauliobjs import NQPauliState

    huh = [
        (NQPauliState("X", (1,)), NQPauliState("X", (1,))),
        (NQPauliState("X", (-1,)), NQPauliState("X", (1,))),
        (NQPauliState("Y", (1,)), NQPauliState("Y", (1,))),
        (NQPauliState("Y", (-1,)), NQPauliState("Y", (1,))),
        (NQPauliState("Z", (1,)), NQPauliState("Z", (1,))),
        (NQPauliState("Z", (-1,)), NQPauliState("Z", (1,))),
        (NQPauliState("X", (1,)), NQPauliState("Y", (1,))),
        (NQPauliState("X", (-1,)), NQPauliState("Y", (1,))),
        (NQPauliState("X", (1,)), NQPauliState("Z", (1,))),
        (NQPauliState("X", (-1,)), NQPauliState("Z", (1,))),
        (NQPauliState("Y", (1,)), NQPauliState("X", (1,))),
        (NQPauliState("Y", (-1,)), NQPauliState("X", (1,))),
        (NQPauliState("Y", (1,)), NQPauliState("Z", (1,))),
        (NQPauliState("Y", (-1,)), NQPauliState("Z", (1,))),
        (NQPauliState("Z", (1,)), NQPauliState("X", (1,))),
        (NQPauliState("Z", (-1,)), NQPauliState("X", (1,))),
        (NQPauliState("Z", (1,)), NQPauliState("Y", (1,))),
        (NQPauliState("Z", (-1,)), NQPauliState("Y", (1,))),
    ]

n_qubits = 4
if "1" in mode_flag or "s" in mode_flag:
    n_qubits = 1
gates = ["Gi", "Gx", "Gy", "Gcnot"]
max_lengths = [1, 2, 4, 8]
if "u" in mode_flag:
    pspec = pygsti.processors.QubitProcessorSpec(
        n_qubits, gates, geometry="line", nonstd_gate_unitaries={(): 1}
    )
elif "o" in mode_flag:
    pspec = pygsti.processors.QubitProcessorSpec(n_qubits, gates, geometry="line")
mdl_target = pygsti.models.create_crosstalk_free_model(pspec)
paulidicts = idt.determine_paulidicts(mdl_target)
if "s" in mode_flag:
    idle_experiments = idt.make_idle_tomography_list(
        n_qubits,
        max_lengths,
        paulidicts,
        maxweight=1,
        force_fid_pairs=huh,
    )
else:
    idle_experiments = idt.make_idle_tomography_list(
        n_qubits, max_lengths, paulidicts, maxweight=1
    )
print(len(idle_experiments), "idle tomography experiments for %d qubits" % n_qubits)

if mode_flag == "u":
    from pygsti.baseobjs import Label

    updated_ckt_list = []
    for ckt in idle_experiments:
        new_ckt = ckt.copy(editable=True)
        for i, lbl in enumerate(ckt):
            print(i, lbl)
            if lbl == Label(()):
                new_ckt[i] = [Label(("Gi", i)) for i in range(n_qubits)]
                # new_ckt[i] = Label(('Gi', ))
        print(new_ckt)
        updated_ckt_list.append(new_ckt)
elif mode_flag == "u1" or mode_flag == "us":
    from pygsti.baseobjs import Label

    updated_ckt_list = []
    for ckt in idle_experiments:
        new_ckt = ckt.copy(editable=True)
        for i, lbl in enumerate(ckt):
            print(i, lbl)
            if lbl == Label(()):
                # new_ckt[i] = [Label(("Gi", i)) for i in range(n_qubits)]
                new_ckt[i] = Label(("Gi", 0))
        print(new_ckt)
        updated_ckt_list.append(new_ckt)
elif "o" in mode_flag:
    updated_ckt_list = idle_experiments


mdl_datagen = pygsti.models.create_crosstalk_free_model(
    pspec, lindblad_error_coeffs={"Gi": {"SX": 0.01}}
)
# Error models! Random with right CP constraints from Taxonomy paper
ds = pygsti.data.simulate_data(mdl_datagen, updated_ckt_list, 100000, seed=8675309)
print(ds)

if mode_flag == "u":
    results = idt.do_idle_tomography(
        n_qubits,
        ds,
        max_lengths,
        paulidicts,
        include_hamiltonian=True,
        include_stochastic=True,
        include_affine=True,
        maxweight=1,
        advanced_options={"jacobian mode": "together"},
        idle_string="[Gi:0Gi:1Gi:2Gi:3]",
    )

elif mode_flag == "us":
    results = idt.do_idle_tomography(
        n_qubits,
        ds,
        max_lengths,
        paulidicts,
        include_hamiltonian=True,
        include_stochastic=True,
        include_affine=True,
        maxweight=1,
        advanced_options={"jacobian mode": "together", "pauli_fidpairs": huh},
        idle_string="Gi:0",
    )

elif "os" in mode_flag:
    results = idt.do_idle_tomography(
        n_qubits,
        ds,
        max_lengths,
        paulidicts,
        maxweight=1,
        include_hamiltonian=True,
        include_stochastic=True,
        include_affine=True,
        advanced_options={"jacobian mode": "together", "pauli_fidpairs": huh},
    )

elif "o" in mode_flag:
    results = idt.do_idle_tomography(
        n_qubits,
        ds,
        max_lengths,
        paulidicts,
        maxweight=1,
        include_hamiltonian=True,
        include_stochastic=True,
    )

elif mode_flag == "u1":
    results = idt.do_idle_tomography(
        n_qubits,
        ds,
        max_lengths,
        paulidicts,
        include_hamiltonian=True,
        include_stochastic=True,
        include_affine=True,
        maxweight=1,
        idle_string="Gi:0",
    )
idt.create_idletomography_report(
    results, "../IDTTestReport", "Test idle tomography example report", auto_open=True
)

results.error_list
ws = pygsti.report.Workspace()
# ws.init_notebook_mode(autodisplay=True)
print(results)
