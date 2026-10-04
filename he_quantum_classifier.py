"""
HE (Highly Entangled) Quantum Classifier Module
================================================
Base paper: HyperKING (Lin & Young, IEEE TGRS 2025), Table IV / Fig. 4.
"""

import torch
import torch.nn as nn
import pennylane as qml

N_QUBITS = 4
N_GROUPS = 32
GROUP_DIM = 16


def _em(control_wire, target_wires, rx_params):
    for target, angle in zip(target_wires, rx_params):
        qml.CRX(angle, wires=[control_wire, target])


def he_quantum_circuit(state_vector, rx1, rz1, em_params, rx2, rz2):
    qml.AmplitudeEmbedding(state_vector, wires=range(N_QUBITS), normalize=True)

    for q in range(N_QUBITS):
        qml.RX(rx1[q], wires=q)
    for q in range(N_QUBITS):
        qml.RZ(rz1[q], wires=q)

    _em(3, [2, 1, 0], em_params[0])
    _em(2, [3, 1, 0], em_params[1])
    _em(1, [3, 2, 0], em_params[2])
    _em(0, [3, 2, 1], em_params[3])

    for q in range(N_QUBITS):
        qml.RX(rx2[q], wires=q)
    for q in range(N_QUBITS):
        qml.RZ(rz2[q], wires=q)

    return [qml.expval(qml.PauliX(q)) for q in range(N_QUBITS)]


class HEQuantumClassifier(nn.Module):
    def __init__(self, qdevice: str = "default.qubit"):
        super().__init__()
        diff_method = "backprop" if qdevice == "default.qubit" else "adjoint"
        self.dev = qml.device(qdevice, wires=N_QUBITS)
        self.circuit = qml.QNode(he_quantum_circuit, self.dev, interface="torch", diff_method=diff_method)

        self.rx1 = nn.Parameter(torch.randn(N_QUBITS) * 0.1)
        self.rz1 = nn.Parameter(torch.randn(N_QUBITS) * 0.1)
        self.em_params = nn.Parameter(torch.randn(4, 3) * 0.1)
        self.rx2 = nn.Parameter(torch.randn(N_QUBITS) * 0.1)
        self.rz2 = nn.Parameter(torch.randn(N_QUBITS) * 0.1)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, N_GROUPS, GROUP_DIM)

        outputs = []
        for b in range(batch_size):
            group_outputs = []
            for g in range(N_GROUPS):
                vec = x[b, g]
                measured = self.circuit(
                    vec, self.rx1, self.rz1, self.em_params, self.rx2, self.rz2
                )
                # Cast quantum measurement output (float64) to float32 so it
                # matches the rest of the network (fixes the Double/Float
                # dtype mismatch in the downstream Sigmoid/Linear layer).
                group_outputs.append(torch.stack(measured).float())
            outputs.append(torch.stack(group_outputs))
        out = torch.stack(outputs)

        return out.reshape(batch_size, 1, 128)


if __name__ == "__main__":
    model = HEQuantumClassifier()
    dummy = torch.randn(2, 2, 16, 16)
    out = model(dummy)
    print("Input shape: ", dummy.shape)
    print("Output shape:", out.shape)
    print("Output dtype:", out.dtype)
    assert out.shape == (2, 1, 128), f"Shape mismatch: {out.shape}"
    print("Shape check passed: (2, 1, 128)")
