import numpy as np
from pygsti.baseobjs import Basis
from itertools import product



# Commutator Helper Functions
def commute(mat1, mat2):
    return mat1@mat2 + mat2@mat1

def anti_commute(mat1, mat2):
    return mat1@mat2 - mat2@mat1

# Hamiltonian Error Generator
def hamiltonian_error_generator(initial_state, pauli_index, pauliDict):
    return -1j*pauliDict[pauli_index]@initial_state@pauliDict["I"] + 1j*pauliDict["I"]@initial_state@pauliDict[pauli_index]

# Stochastic Error Generator
def stochastic_error_generator(initial_state, pauli_index, pauliDict, numQubits):
    return pauliDict[pauli_index]@initial_state@pauliDict[pauli_index] - pauliDict["I"*numQubits]@initial_state@pauliDict["I"*numQubits]

# Pauli-correlation Error Generator
def pauli_correlation_error_generator(initial_state, pauli_index_1, pauli_index_2, pauliDict):
    return pauliDict[pauli_index_1]@initial_state@pauliDict[pauli_index_2] + pauliDict[pauli_index_2]@initial_state@pauliDict[pauli_index_1] - 0.5 * commute(commute(pauliDict[pauli_index_1], pauliDict[pauli_index_2]), initial_state)

# Anti-symmetric Error Generator
def anti_symmetric_error_generator(initial_state, pauli_index_1, pauli_index_2, pauliDict):
    return 1j*(pauliDict[pauli_index_1]@initial_state@pauliDict[pauli_index_2] - pauliDict[pauli_index_2]@initial_state@pauliDict[pauli_index_1] + 0.5 * commute(anti_commute(pauliDict[pauli_index_1], pauliDict[pauli_index_2]), initial_state))

# Convert basis
def convert_to_pauli(matrix, numQubits):
    pauliNames1Q = ["I", "X", "Y", "Z"]
    # Hard force to 1- or 2-qubit
    if numQubits == 1:
        pauliNames = pauliNames1Q
    elif numQubits == 2:
        pauliNames = [''.join(name) for name in product(pauliNames1Q, pauliNames1Q)]
    translationMatrix = pp.from_std_transform_matrix
    coefs = np.real_if_close(np.dot(translationMatrix,matrix.flatten()))
    return [(a,b) for (a,b) in zip(coefs,pauliNames) if abs(a) > 0.0001]

np.set_printoptions(precision=1, linewidth=1000)
numQubits = 2
pp1Q = Basis.cast('PP', dim = 4)
pp = Basis.cast('PP', dim=4**numQubits)
pauliNames1Q = ["I", "X", "Y", "Z"]
pauliStates1Q = ["X+", "X-", "Y+","Y-", "Z+", "Z-"]
# Temporary Hard Coding Error Gens to 1-, and 2-qubits
if numQubits == 1:
    pauliNames = pauliNames1Q
    initialStates = pauliStates1Q
elif numQubits == 2:
    pauliNames = [''.join(name) for name in product(pauliNames1Q, pauliNames1Q)]
    initialStates = [','.join((name)) for name in product(pauliStates1Q + ["I"],pauliStates1Q+["I"])][:-1]


# Compute the set of measurable effects of Hamiltonian error generators operating on two qubits in each of the specified eigenstates
hamiltonianIndices = pauliNames[1:]
hamiltonianErrorOutputs = dict()

for index in hamiltonianIndices:
    for state in initialStates:
        tempState = dict(enumerate(state.split(',')))
        tempIndex = dict(enumerate(index))
        print(tempIndex)
        
        errorState = tempState.copy()
        for (key, value) in tempState.items():
            print(key, value, value[:-1])
            if value[-1] == "+":
                errorState[key] = 0.5*hamiltonian_error_generator(pp1Q["I"], tempIndex[key], pp1Q) + 0.5*hamiltonian_error_generator(pp1Q[value[:-1]], tempIndex[key], pp1Q)
            elif value[-1] == "-":
                errorState[key] = 0.5*hamiltonian_error_generator(pp1Q["I"], tempIndex[key], pp1Q) - 0.5*hamiltonian_error_generator(pp1Q[value[:-1]], tempIndex[key], pp1Q)
            else:
                # I am pretty sure this is wrong?
                errorState[key] = hamiltonian_error_generator(pp1Q["I"], tempIndex[key], pp1Q)
        print(errorState)
        print(state)
        
        if numQubits == 1:
            hamiltonianErrorOutputs[(index,state)] = errorState[0]
        if numQubits == 2:
            hamiltonianErrorOutputs[(index,state)] = np.tensordot(errorState[0], errorState[1], 0)

for key in hamiltonianErrorOutputs:
    hamiltonianErrorOutputs[key] = convert_to_pauli(hamiltonianErrorOutputs[key], numQubits)

print(hamiltonianErrorOutputs)
sys.exit()



numQubits = 2
for index in hamiltonianIndices:
    for state in initialStates:
        if state[-1] == "+":
            hamiltonianErrorOutputs[(index, state)] = 0.5*hamiltonian_error_generator(pp["I"*numQubits], index, pp, numQubits) + 0.5*hamiltonian_error_generator(pp[state[:-1]], index, pp, numQubits)
        elif state[-1] == "-":
            hamiltonianErrorOutputs[(index, state)] = 0.5*hamiltonian_error_generator(pp["I"*numQubits], index, pp, numQubits) - 0.5*hamiltonian_error_generator(pp[state[:-1]], index, pp, numQubits)

# Convert measurable effects into coefficients
for key in hamiltonianErrorOutputs:
    hamiltonianErrorOutputs[key] = convert_to_pauli(hamiltonianErrorOutputs[key], numQubits)

for key in hamiltonianErrorOutputs:
    print(key, "\n", hamiltonianErrorOutputs[key])

import sys
sys.exit()


# Compute the set of measurable effects of stochastic error generators operating on a single qubit in each of the specified eigenstates
stochasticIndices = ["X", "Y", "Z"]
stochasticErrorOutputs = dict()
for index in stochasticIndices:
    for state in initialStates:
        if state[-1] == "+":
            stochasticErrorOutputs[(index, state)] = 0.5*stochastic_error_generator(pp["I"], index, pp,1) + 0.5*stochastic_error_generator(pp[state[0]], index, pp,1)
        elif state[-1] == "-":
            stochasticErrorOutputs[(index, state)] = 0.5*stochastic_error_generator(pp["I"], index, pp,1) - 0.5*stochastic_error_generator(pp[state[0]], index, pp,1)

# Convert measurable effects into coefficients
for key in stochasticErrorOutputs:
    stochasticErrorOutputs[key] = convert_to_pauli(stochasticErrorOutputs[key], numQubits)
for key in stochasticErrorOutputs:
    print(key, "\n", stochasticErrorOutputs[key])


# Compute the set of measurable effects of pauli-correlation error generators operating on a single qubit in each of the specified eigenstates
pauliCorrelationIndices = ["XY", "XZ", "YZ"]
pauliCorrelationErrorOutputs = dict()
for index in pauliCorrelationIndices:
    for state in initialStates:
        if state[-1] == "+":
            pauliCorrelationErrorOutputs[(index, state)] = 0.5*pauli_correlation_error_generator(pp["I"], index[0], index[1], pp) + 0.5*pauli_correlation_error_generator(pp[state[0]], index[0], index[1], pp)
        elif state[-1] == "-":
            pauliCorrelationErrorOutputs[(index, state)] = 0.5*pauli_correlation_error_generator(pp["I"], index[0], index[1], pp) - 0.5*pauli_correlation_error_generator(pp[state[0]], index[0], index[1], pp)

# Convert measurable effects into coefficients
for key in pauliCorrelationErrorOutputs:
    pauliCorrelationErrorOutputs[key] = convert_to_pauli(pauliCorrelationErrorOutputs[key], numQubits)
for key in pauliCorrelationErrorOutputs:
    print(key, "\n", pauliCorrelationErrorOutputs[key])

# Compute the set of measurable effects of pauli-correlation error generators operating on a single qubit in each of the specified eigenstates
antiSymmetricIndices = ["XY", "XZ", "YZ"]
antiSymmetricErrorOutputs = dict()
for index in antiSymmetricIndices:
    for state in initialStates:
        if state[-1] == "+":
            antiSymmetricErrorOutputs[(index, state)] = 0.5*anti_symmetric_error_generator(pp["I"], index[0], index[1], pp) + 0.5*anti_symmetric_error_generator(pp[state[0]], index[0], index[1], pp)
        elif state[-1] == "-":
            antiSymmetricErrorOutputs[(index, state)] = 0.5*anti_symmetric_error_generator(pp["I"], index[0], index[1], pp) - 0.5*anti_symmetric_error_generator(pp[state[0]], index[0], index[1], pp)

# Convert measurable effects into coefficients
for key in antiSymmetricErrorOutputs:
    antiSymmetricErrorOutputs[key] = convert_to_pauli(antiSymmetricErrorOutputs[key], numQubits)
for key in antiSymmetricErrorOutputs:
    print(key, "\n", antiSymmetricErrorOutputs[key])
