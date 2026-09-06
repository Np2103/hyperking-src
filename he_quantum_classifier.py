"""
HE (Highly Entangled) Quantum Classifier Module
================================================
Base paper: HyperKING (Lin & Young, IEEE TGRS 2025), Table IV / Fig. 4.

Combines three of Table IV's rows into one PennyLane quantum circuit,
since the quantum state cannot be split across separate classical
modules (unlike the Reshape ops elsewhere, which are just tensor
reshuffling):

  Reshape:              2x16x16  -> 32x16
  Amplitude Embedding:  32x16    -> 32x4   (16 values packed into 4 qubits
                                             per group, via amplitude encoding)
  HE Quantum Classifier: RX -> RZ -> [EM(210|3), EM(310|2), EM(320|1), EM(321|0)]
                         -> RX -> RZ -> Pauli-X measurement
                         32x4 -> 32x4 (measured) -> reshaped to 1x128

Key differences from the Generator's Core Quantum FE Module:
  - Amplitude encoding (not angle encoding): a whole 16-value vector is
    packed into the amplitudes of one 4-qubit state, rather than one
    value per qubit. This needs the vector L2-normalized first (a
    quantum state's amplitudes must satisfy sum(|c_i|^2) = 1).
  - CRX entanglement (not Ising XX): a controlled gate, not a
    symmetric one -- one qubit's state directly gates another's
    rotation. EM(bcd|a) means qubit a sequentially controls b, c, d.
  - Pauli-X measurement (not Pauli-Z).

Input:  (batch, 2, 16, 16)  -- output of DS Module
Output: (batch, 1, 128)     -- fed into the Sigmoid Module
"""

import torch
import torch.nn as nn
import pennylane as qml

N_QUBITS = 4
N_GROUPS = 32          # 2x16x16 reshaped -> 32 groups of 16 values each
GROUP_DIM = 16          # 16 = 2^4, fits exactly into 4 qubits via amplitude encoding

dev = qml.device("default.qubit", wires=N_QUBITS)


def _em(control_wire, target_wires, rx_params):
    """EM(bcd|a): control_wire sequentially controls each of target_wires
    via a CRX gate. rx_params has one angle per target wire."""
    for target, angle in zip(target_wires, rx_params):
        qml.CRX(angle, wires=[control_wire, target])


@qml.qnode(dev, interface="torch", diff_method="backprop")
def he_quantum_circuit(state_vector, rx1, rz1, em_params, rx2, rz2):
    """One 4-qubit run of the HE Quantum Classifier circuit.

    state_vector: (16,) L2-normalized real vector -> amplitude-embedded
    rx1, rz1, rx2, rz2: (4,) learnable rotation angles, one per qubit
    em_params: (4, 3) learnable CRX angles -- 4 EM modules, 3 CRX gates each
    """
    qml.AmplitudeEmbedding(state_vector, wires=range(N_QUBITS), normalize=True)

    for q in range(N_QUBITS):
        qml.RX(rx1[q], wires=q)
    for q in range(N_QUBITS):
        qml.RZ(rz1[q], wires=q)

    # Four EM modules -- every qubit gets a turn as controller of the other three.
    _em(3, [2, 1, 0], em_params[0])   # EM(210|3)
    _em(2, [3, 1, 0], em_params[1])   # EM(310|2)
    _em(1, [3, 2, 0], em_params[2])   # EM(320|1)
    _em(0, [3, 2, 1], em_params[3])   # EM(321|0)

    for q in range(N_QUBITS):
        qml.RX(rx2[q], wires=q)
    for q in range(N_QUBITS):
        qml.RZ(rz2[q], wires=q)

    return [qml.expval(qml.PauliX(q)) for q in range(N_QUBITS)]


class HEQuantumClassifierModule(nn.Module):
    def __init__(self):
        super().__init__()
        # Learnable rotation angles, shared across all 32 groups (standard
        # weight-sharing choice, same pattern as the Generator's Core
        # Quantum FE Module applying one set of learned gates 128 times).
        self.rx1 = nn.Parameter(torch.randn(N_QUBITS) * 0.1)
        self.rz1 = nn.Parameter(torch.randn(N_QUBITS) * 0.1)
        self.em_params = nn.Parameter(torch.randn(4, 3) * 0.1)
        self.rx2 = nn.Parameter(torch.randn(N_QUBITS) * 0.1)
        self.rz2 = nn.Parameter(torch.randn(N_QUBITS) * 0.1)

    def forward(self, x):
        # x: (batch, 2, 16, 16) -> reshape to (batch, 32, 16)
        batch_size = x.shape[0]
        x = x.reshape(batch_size, N_GROUPS, GROUP_DIM)

        outputs = []
        for b in range(batch_size):
            group_outputs = []
            for g in range(N_GROUPS):
                vec = x[b, g]  # (16,)
                measured = he_quantum_circuit(
                    vec, self.rx1, self.rz1, self.em_params, self.rx2, self.rz2
                )
                group_outputs.append(torch.stack(measured))  # (4,)
            outputs.append(torch.stack(group_outputs))  # (32, 4)
        out = torch.stack(outputs)  # (batch, 32, 4)

        return out.reshape(batch_size, 1, 128)


if __name__ == "__main__":
    model = HEQuantumClassifierModule()
    dummy = torch.randn(2, 2, 16, 16)  # small batch -- this circuit runs per-group, so keep it small for a quick test
    out = model(dummy)
    print("Input shape: ", dummy.shape)
    print("Output shape:", out.shape)
    assert out.shape == (2, 1, 128), f"Shape mismatch: {out.shape}"
    print("Shape check passed: (2, 1, 128)")
