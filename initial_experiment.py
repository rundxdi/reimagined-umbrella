import numpy as np
from pygsti.baseobjs import Basis
from itertools import product, permutations


# Commutator Helper Functions
def commute(mat1, mat2):
    return mat1 @ mat2 + mat2 @ mat1


def anti_commute(mat1, mat2):
    return mat1 @ mat2 - mat2 @ mat1


# Hamiltonian Error Generator
def hamiltonian_error_generator(initial_state, indexed_pauli, identity):
    return (
        -1j * indexed_pauli @ initial_state @ identity
        + 1j * identity @ initial_state @ indexed_pauli
    )


# Stochastic Error Generator
def stochastic_error_generator(initial_state, indexed_pauli, identity):
    return (
        indexed_pauli @ initial_state @ indexed_pauli
        - identity @ initial_state @ identity
    )


# Pauli-correlation Error Generator
def pauli_correlation_error_generator(
    initial_state,
    pauli_index_1,
    pauli_index_2,
):
    return (
        pauli_index_1 @ initial_state @ pauli_index_2
        + pauli_index_2 @ initial_state @ pauli_index_1
        - 0.5 * commute(commute(pauli_index_1, pauli_index_2), initial_state)
    )


# Anti-symmetric Error Generator
def anti_symmetric_error_generator(initial_state, pauli_index_1, pauli_index_2):
    return 1j * (
        pauli_index_1 @ initial_state @ pauli_index_2
        - pauli_index_2 @ initial_state @ pauli_index_1
        + 0.5
        * commute(
            anti_commute(pauli_index_1, pauli_index_2),
            initial_state,
        )
    )


# Convert basis
def convert_to_pauli(matrix, numQubits):
    pauliNames1Q = ["I", "X", "Y", "Z"]
    # Hard force to 1- or 2-qubit
    if numQubits == 1:
        pauliNames = pauliNames1Q
    elif numQubits == 2:
        pauliNames = ["".join(name) for name in product(pauliNames1Q, pauliNames1Q)]
    translationMatrix = pp.from_std_transform_matrix
    coefs = np.real_if_close(np.dot(translationMatrix, matrix.flatten()))
    return [(a, b) for (a, b) in zip(coefs, pauliNames) if abs(a) > 0.0001]


np.set_printoptions(precision=1, linewidth=1000)
numQubits = 2
pp1Q = Basis.cast("PP", dim=4)
pp = Basis.cast("PP", dim=4**numQubits)
pauliNames1Q = ["I", "X", "Y", "Z"]
pauliStates1Q = ["X+", "X-", "Y+", "Y-", "Z+", "Z-"]
# Temporary Hard Coding Error Gens to 1-, and 2-qubits
if numQubits == 1:
    pauliNames = pauliNames1Q
    initialStates = pauliStates1Q
elif numQubits == 2:
    pauliNames = ["".join(name) for name in product(pauliNames1Q, pauliNames1Q)]
    initialStates = [
        ",".join((name))
        for name in product(pauliStates1Q + ["I"], pauliStates1Q + ["I"])
    ][:-1]


# Compute the set of measurable effects of Hamiltonian error generators operating on two qubits in each of the specified eigenstates
hamiltonianIndices = pauliNames[1:]
hamiltonianErrorOutputs = dict()

for index in hamiltonianIndices:
    for state in initialStates:
        tempState = dict(enumerate(state.split(",")))
        if numQubits == 2:
            if tempState[0][-1] == "-":
                mat1 = -1 * pp1Q[tempState[0][0]]
            elif tempState[0][-1] == "+":
                mat1 = pp1Q[tempState[0][0]]
            else:
                mat1 == pp1Q["I"]
            if tempState[1][-1] == "-":
                mat2 = -1 * pp1Q[tempState[1][0]]
            elif tempState[1][-1] == "+":
                mat2 = pp1Q[tempState[1][0]]
            else:
                mat2 == pp1Q["I"]
            ident = pp["II"]

            inputState = np.kron(mat1, mat2)
            inputState = ident / 2 + inputState / 2
            hamiltonianErrorOutputs[(index, state)] = hamiltonian_error_generator(
                inputState, pp[index], ident
            )

for key in hamiltonianErrorOutputs:
    hamiltonianErrorOutputs[key] = convert_to_pauli(
        hamiltonianErrorOutputs[key], numQubits
    )

for key in hamiltonianErrorOutputs:
    print(key, "\n", hamiltonianErrorOutputs[key])


# Compute the set of measurable effects of stochastic error generators operating on a single qubit in each of the specified eigenstates
stochasticIndices = hamiltonianIndices
stochasticErrorOutputs = dict()
for index in stochasticIndices:
    for state in initialStates:
        tempState = dict(enumerate(state.split(",")))
        if numQubits == 2:
            if tempState[0][-1] == "-":
                mat1 = -1 * pp1Q[tempState[0][0]]
            elif tempState[0][-1] == "+":
                mat1 = pp1Q[tempState[0][0]]
            else:
                mat1 == pp1Q["I"]
            if tempState[1][-1] == "-":
                mat2 = -1 * pp1Q[tempState[1][0]]
            elif tempState[1][-1] == "+":
                mat2 = pp1Q[tempState[1][0]]
            else:
                mat2 == pp1Q["I"]
            ident = pp["II"]

            inputState = np.kron(mat1, mat2)
            inputState = ident / 2 + inputState / 2
            stochasticErrorOutputs[(index, state)] = stochastic_error_generator(
                inputState, pp[index], ident
            )
# Convert measurable effects into coefficients
for key in stochasticErrorOutputs:
    stochasticErrorOutputs[key] = convert_to_pauli(
        stochasticErrorOutputs[key], numQubits
    )
for key in stochasticErrorOutputs:
    print(key, "\n", stochasticErrorOutputs[key])


# Compute the set of measurable effects of pauli-correlation error generators operating on a single qubit in each of the specified eigenstates
pauliCorrelationIndices = list(permutations(pauliNames[1:], 2))
pauliCorrelationErrorOutputs = dict()
for index in pauliCorrelationIndices:
    for state in initialStates:
        tempState = dict(enumerate(state.split(",")))
        if numQubits == 2:
            if tempState[0][-1] == "-":
                mat1 = -1 * pp1Q[tempState[0][0]]
            elif tempState[0][-1] == "+":
                mat1 = pp1Q[tempState[0][0]]
            else:
                mat1 == pp1Q["I"]
            if tempState[1][-1] == "-":
                mat2 = -1 * pp1Q[tempState[1][0]]
            elif tempState[1][-1] == "+":
                mat2 = pp1Q[tempState[1][0]]
            else:
                mat2 == pp1Q["I"]
            ident = pp["II"]

            inputState = np.kron(mat1, mat2)
            inputState = ident / 2 + inputState / 2
            pauliCorrelationErrorOutputs[
                (index, state)
            ] = pauli_correlation_error_generator(
                inputState, pp[index[0]], pp[index[1]]
            )

# Convert measurable effects into coefficients
for key in pauliCorrelationErrorOutputs:
    pauliCorrelationErrorOutputs[key] = convert_to_pauli(
        pauliCorrelationErrorOutputs[key], numQubits
    )
for key in pauliCorrelationErrorOutputs:
    print(key, "\n", pauliCorrelationErrorOutputs[key])


# Compute the set of measurable effects of pauli-correlation error generators operating on a single qubit in each of the specified eigenstates
antiSymmetricIndices = pauliCorrelationIndices
antiSymmetricErrorOutputs = dict()
for index in antiSymmetricIndices:
    for state in initialStates:
        tempState = dict(enumerate(state.split(",")))
        if numQubits == 2:
            if tempState[0][-1] == "-":
                mat1 = -1 * pp1Q[tempState[0][0]]
            elif tempState[0][-1] == "+":
                mat1 = pp1Q[tempState[0][0]]
            else:
                mat1 == pp1Q["I"]
            if tempState[1][-1] == "-":
                mat2 = -1 * pp1Q[tempState[1][0]]
            elif tempState[1][-1] == "+":
                mat2 = pp1Q[tempState[1][0]]
            else:
                mat2 == pp1Q["I"]
            ident = pp["II"]

            inputState = np.kron(mat1, mat2)
            inputState = ident / 2 + inputState / 2
            antiSymmetricErrorOutputs[(index, state)] = anti_symmetric_error_generator(
                inputState, pp[index[0]], pp[index[1]]
            )

# Convert measurable effects into coefficients
for key in antiSymmetricErrorOutputs:
    antiSymmetricErrorOutputs[key] = convert_to_pauli(
        antiSymmetricErrorOutputs[key], numQubits
    )
for key in antiSymmetricErrorOutputs:
    print(key, "\n", antiSymmetricErrorOutputs[key])
import sys
