"""
core_quantum_fe.py — Core Quantum FE module (Theorem 1).
"""

import pennylane as qml
import torch
import torch.nn as nn

N_QUBITS = 4


def _circuit(features, alpha, theta_xx1, beta, theta_xx2, gamma):
    for k in range(N_QUBITS):
        qml.RY(features[k], wires=k)
    for k in range(N_QUBITS):
        qml.RZ(alpha[k], wires=k)
    qml.IsingXX(theta_xx1[0], wires=[1, 2])
    qml.IsingXX(theta_xx1[1], wires=[0, 3])
    for k in range(N_QUBITS):
        qml.RY(beta[k], wires=k)
    qml.IsingXX(theta_xx2[0], wires=[0, 1])
    qml.IsingXX(theta_xx2[1], wires=[2, 3])
    for k in range(N_QUBITS):
        qml.RZ(gamma[k], wires=k)
    qml.Toffoli(wires=[0, 1, 2])
    qml.Toffoli(wires=[1, 2, 3])
    qml.Toffoli(wires=[2, 3, 0])
    qml.Toffoli(wires=[3, 0, 1])
    return [qml.expval(qml.PauliZ(w)) for w in range(N_QUBITS)]


class CoreQuantumFE(nn.Module):
    def __init__(self, qdevice: str = "default.qubit"):
        super().__init__()
        diff_method = "backprop" if qdevice == "default.qubit" else "adjoint"
        self.dev = qml.device(qdevice, wires=N_QUBITS)
        self.circuit = qml.QNode(_circuit, self.dev, interface="torch", diff_method=diff_method)

        self.alpha = nn.Parameter(torch.randn(N_QUBITS) * 0.1)
        self.theta_xx1 = nn.Parameter(torch.randn(2) * 0.1)
        self.beta = nn.Parameter(torch.randn(N_QUBITS) * 0.1)
        self.theta_xx2 = nn.Parameter(torch.randn(2) * 0.1)
        self.gamma = nn.Parameter(torch.randn(N_QUBITS) * 0.1)

    def forward(self, x):
        batch_size, n_channels, vec_len = x.shape
        assert vec_len == 4, f"expected length-4 vectors, got {vec_len}"

        outputs = torch.zeros(batch_size, n_channels, 2, dtype=torch.float32)

        for b in range(batch_size):
            for c in range(n_channels):
                z0, z1, z2, z3 = self.circuit(
                    x[b, c], self.alpha, self.theta_xx1, self.beta,
                    self.theta_xx2, self.gamma,
                )
                outputs[b, c, 0] = (z0 + z1) / 2
                outputs[b, c, 1] = (z2 + z3) / 2

        return outputs.reshape(batch_size, 64, 2, 2)


if __name__ == "__main__":
    torch.manual_seed(0)

    print("Quick single-circuit check via a temporary module instance...")
    _probe = CoreQuantumFE()
    _test_vec = torch.tensor([0.1, 0.5, -0.3, 0.8])
    _result = _probe.circuit(_test_vec, _probe.alpha, _probe.theta_xx1, _probe.beta, _probe.theta_xx2, _probe.gamma)
    print(f"  4 Pauli-Z measurements: {[float(r) for r in _result]}")
    print()

    print("Testing CoreQuantumFE on a small dummy batch...")
    quantum_fe = CoreQuantumFE()
    small_input = torch.randn(1, 128, 4) * 0.5
    import time
    start = time.time()
    out = quantum_fe(small_input)
    elapsed = time.time() - start
    print(f"Input shape:  {tuple(small_input.shape)} (expected: (1, 128, 4))")
    print(f"Output shape: {tuple(out.shape)} (expected: (1, 64, 2, 2))")
    print(f"Time: {elapsed:.2f}s")
    loss = out.sum()
    loss.backward()
    print(f"Gradient check - alpha.grad is not None: {quantum_fe.alpha.grad is not None}")
    print("Done.")
