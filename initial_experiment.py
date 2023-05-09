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
    pp = Basis.cast("PP", dim=4**numQubits)
    translationMatrix = pp.from_std_transform_matrix
    coefs = np.real_if_close(np.dot(translationMatrix, matrix.flatten()))
    return [(a, b) for (a, b) in zip(coefs, pauliNames) if abs(a) > 0.0001]


# Compute the set of measurable effects of Hamiltonian error generators operating on two qubits in each of the specified eigenstates
# Return: hamiltonian error and coef dictionary
def gather_hamiltonian_jacobian_coefs(
    pauliNames, initialStates, numQubits, printFlag=False
):
    hamiltonianIndices = pauliNames[1:]
    hamiltonianErrorOutputs = dict()

    for index in hamiltonianIndices:
        for state in initialStates:
            tempState = dict(enumerate(state.split(",")))
            # print(tempState)
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
            elif numQubits == 1:
                if state[-1] == "-":
                    mat1 = -1 * pp1Q[state[0]]
                elif tempState[0][-1] == "+":
                    mat1 = pp1Q[state[0]]
                else:
                    mat1 == pp1Q["I"]
                ident = pp1Q["I"]
                inputState = ident / 2 + mat1 / 2
                hamiltonianErrorOutputs[(index, state)] = hamiltonian_error_generator(
                    inputState, pp1Q[index], ident
                )

    for key in hamiltonianErrorOutputs:
        hamiltonianErrorOutputs[key] = convert_to_pauli(
            hamiltonianErrorOutputs[key], numQubits
        )
    if printFlag:
        for key in hamiltonianErrorOutputs:
            print(key, "\n", hamiltonianErrorOutputs[key])

    return hamiltonianErrorOutputs


# Compute the set of measurable effects of stochastic error generators operating on a single qubit in each of the specified eigenstates
def gather_stochastic_jacobian_coefs(
    pauliNames, initialStates, numQubits, printFlag=False
):
    stochasticIndices = pauliNames[1:]
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
            elif numQubits == 1:
                if state[-1] == "-":
                    mat1 = -1 * pp1Q[state[0]]
                elif tempState[0][-1] == "+":
                    mat1 = pp1Q[state[0]]
                else:
                    mat1 == pp1Q["I"]
                ident = pp1Q["I"]
                inputState = ident / 2 + mat1 / 2
                stochasticErrorOutputs[(index, state)] = stochastic_error_generator(
                    inputState, pp1Q[index], ident
                )
    # Convert measurable effects into coefficients
    for key in stochasticErrorOutputs:
        stochasticErrorOutputs[key] = convert_to_pauli(
            stochasticErrorOutputs[key], numQubits
        )
    if printFlag:
        for key in stochasticErrorOutputs:
            print(key, "\n", stochasticErrorOutputs[key])

    return stochasticErrorOutputs


# Compute the set of measurable effects of pauli-correlation error generators operating on a single qubit in each of the specified eigenstates
def gather_pauli_correlation_jacobian_coefs(
    pauliNames, initialStates, numQubits, printFlag=False
):
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
            elif numQubits == 1:
                if state[-1] == "-":
                    mat1 = -1 * pp1Q[state[0]]
                elif tempState[0][-1] == "+":
                    mat1 = pp1Q[state[0]]
                else:
                    mat1 == pp1Q["I"]
                ident = pp1Q["I"]
                inputState = ident / 2 + mat1 / 2
                pauliCorrelationErrorOutputs[
                    (index, state)
                ] = pauli_correlation_error_generator(
                    inputState, pp1Q[index[0]], pp1Q[index[1]]
                )

    # Convert measurable effects into coefficients
    for key in pauliCorrelationErrorOutputs:
        pauliCorrelationErrorOutputs[key] = convert_to_pauli(
            pauliCorrelationErrorOutputs[key], numQubits
        )

    if printFlag:
        for key in pauliCorrelationErrorOutputs:
            print(key, "\n", pauliCorrelationErrorOutputs[key])

    return pauliCorrelationErrorOutputs


# Compute the set of measurable effects of pauli-correlation error generators operating on a single qubit in each of the specified eigenstates
def gather_anti_symmetric_jacobian_coefs(
    pauliNames, initialStates, numQubits, printFlag=False
):
    antiSymmetricIndices = list(permutations(pauliNames[1:], 2))
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
                antiSymmetricErrorOutputs[
                    (index, state)
                ] = anti_symmetric_error_generator(
                    inputState, pp[index[0]], pp[index[1]]
                )
            elif numQubits == 1:
                if state[-1] == "-":
                    mat1 = -1 * pp1Q[state[0]]
                elif tempState[0][-1] == "+":
                    mat1 = pp1Q[state[0]]
                else:
                    mat1 == pp1Q["I"]
                ident = pp1Q["I"]
                inputState = ident / 2 + mat1 / 2
                antiSymmetricErrorOutputs[
                    (index, state)
                ] = anti_symmetric_error_generator(
                    inputState, pp1Q[index[0]], pp1Q[index[1]]
                )

    # Convert measurable effects into coefficients
    for key in antiSymmetricErrorOutputs:
        antiSymmetricErrorOutputs[key] = convert_to_pauli(
            antiSymmetricErrorOutputs[key], numQubits
        )

    if printFlag:
        for key in antiSymmetricErrorOutputs:
            print(key, "\n", antiSymmetricErrorOutputs[key])

    return antiSymmetricErrorOutputs


def build_class_jacobian(classification, numQubits):
    if numQubits == 1:
        pauliNames = pauliNames1Q
        initialStates = pauliStates1Q
    elif numQubits == 2:
        pauliNames = ["".join(name) for name in product(pauliNames1Q, pauliNames1Q)]
        initialStates = [
            ",".join((name))
            for name in product(pauliStates1Q + ["I"], pauliStates1Q + ["I"])
        ][:-1]

    # classification within ["H", "S", "C", "A"]
    if classification == "H":
        jacobian_coefs = gather_hamiltonian_jacobian_coefs(
            pauliNames=pauliNames,
            initialStates=initialStates,
            numQubits=numQubits,
            printFlag=False,
        )
        pauliIndexList = dict(enumerate(pauliNames[1:]))
        # print(jacobian)
    elif classification == "S":
        jacobian_coefs = gather_stochastic_jacobian_coefs(
            pauliNames=pauliNames,
            initialStates=initialStates,
            numQubits=numQubits,
            printFlag=False,
        )
        pauliIndexList = dict(enumerate(pauliNames[1:]))
    elif classification == "C":
        jacobian_coefs = gather_pauli_correlation_jacobian_coefs(
            pauliNames=pauliNames,
            initialStates=initialStates,
            numQubits=numQubits,
            printFlag=False,
        )
        pauliIndexList = dict(enumerate(permutations(pauliNames[1:], 2)))
    elif classification == "A":
        jacobian_coefs = gather_anti_symmetric_jacobian_coefs(
            pauliNames=pauliNames,
            initialStates=initialStates,
            numQubits=numQubits,
            printFlag=False,
        )
        pauliIndexList = dict(enumerate(permutations(pauliNames[1:], 2)))

    else:
        print(
            "Classification value must be 'H', 'S', 'C', or 'A'.  Please provide a valid argument."
        )
        quit()

    pauliIndexList = {v: k for k, v in pauliIndexList.items()}
    extrinsicErrorList = dict(enumerate(product(pauliNames[1:], initialStates)))
    extrinsicErrorList = {v: k for k, v in extrinsicErrorList.items()}
    jacobian = np.zeros((len(extrinsicErrorList), len(pauliIndexList)))
    for key, value in jacobian_coefs.items():
        # print(key, value)
        if len(value) > 0:
            for val in value:
                rowIdx = extrinsicErrorList.get((val[1], key[1]))
                # print((val[1], key[1]), rowIdx)
                colIdx = pauliIndexList.get(key[0])
                # print(key[0], colIdx)
                # print(val, value)
                # print(rowIdx, colIdx)
                jacobian[rowIdx][colIdx] = val[0]
    print(pauliIndexList)
    return jacobian


if __name__ == "__main__":
    np.set_printoptions(precision=1, linewidth=1000)
    pp1Q = Basis.cast("PP", dim=4)
    # hard forcing 2 qubits currently
    pp = Basis.cast("PP", dim=16)
    pauliNames1Q = ["I", "X", "Y", "Z"]
    pauliStates1Q = ["X+", "X-", "Y+", "Y-", "Z+", "Z-"]
    # Temporary Hard Coding Error Gens to 1-, and 2-qubits

    hamiltonian_jacobian = build_class_jacobian("H", 1)
    print(hamiltonian_jacobian)
    stochastic_jacobian = build_class_jacobian("S", 1)
    print(stochastic_jacobian)
    correlation_jacobian = build_class_jacobian("C", 1)
    print(correlation_jacobian)
    anti_symmetric_jacobian = build_class_jacobian("A", 1)
    print(anti_symmetric_jacobian)

    full_jacobian = np.hstack(
        (
            hamiltonian_jacobian,
            stochastic_jacobian,
            correlation_jacobian,
            anti_symmetric_jacobian,
        ),
    )
    print(full_jacobian)
    inverse_jacobian = np.linalg.pinv(full_jacobian)
    print(inverse_jacobian)


##### For Corey/Kenny/Robin 5/11:
##### 1) Help putting together paper draft!  Divide and conquer?  I will definitely work on results/conclusions and some of methodology?


# ok time to do bad things
