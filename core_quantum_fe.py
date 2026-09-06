"""
core_quantum_fe.py
-------------------
Module 3 of the HyperKING Generator: the Core Quantum FE module -
the actual quantum circuit (this is "Theorem 1" in the paper, the part
that provides its theoretical expressibility guarantee).

Runs on 4 qubits, applied independently to each of the 128 length-4
vectors coming out of the Reshape module (so 128 circuit evaluations
per image, all sharing the SAME learned parameters - like a
convolution kernel shared across spatial positions, but here it's a
quantum circuit shared across the 128 "channels").

Circuit, per the paper (Table III / Fig. 3), applied to one length-4
vector [f0, f1, f2, f3]:

    1. Angle Embedding:  RY(f_k) on qubit k, for k = 0,1,2,3
    2. RZ(alpha_k) on each qubit
    3. Ising XX(theta) entangling layer, pairing (1,2) and (0,3)
    4. RY(beta_k) on each qubit
    5. Ising XX(theta) entangling layer, pairing (0,1) and (2,3)
    6. RZ(gamma_k) on each qubit
       [steps 2-6 are the "RZ-XX-RY-XX-RZ" sequence Theorem 1 proves
       is fully expressible - i.e. with the right learned angles it
       can realize ANY 4-qubit unitary]
    7. Toffoli (CCNOT) cycle: (0,1->2), (1,2->3), (2,3->0), (3,0->1) -
       fixed, not trainable, adds a stronger layer of entanglement
    8. Pauli-Z measurement on each qubit

Input:  (B, 128, 4)
Output: (B, 64, 2, 2)

IMPORTANT - one place where the source material is genuinely
ambiguous, flagged honestly rather than silently guessed around:
your reference doc states the measurement step yields "128x2" (which
reshapes cleanly into 64x2x2 = 256 numbers), but a clean Pauli-Z
measurement of 4 qubits naturally gives 4 numbers per circuit run
(one expectation value per qubit), not 2. The doc does not specify
exactly how those 4 measured values become 2. This implementation
measures all 4 qubits, then averages them in adjacent pairs -
(Z0+Z1)/2 and (Z2+Z3)/2 - to land on the documented 128x2 output
shape, since that pairing mirrors the circuit's own (0,1)/(2,3) and
(1,2)/(0,3) entangling structure. This is a documented, reasonable
default, not a value taken directly from the paper - flag this to
your advisor if you get access to the exact measurement convention
used in the original Table III.

A note on speed: this file loops over all 128 channels (and the
batch) in plain Python, calling the quantum circuit once per channel.
That's fine for the shape/correctness testing this file's __main__
block does, but it WILL be slow once real training starts (thousands
of circuit evaluations per training step). We'll revisit batching/
vectorizing this (e.g. via PennyLane's broadcasting support or
lightning.qubit) when we get to the training loop - not a concern yet.
"""

import pennylane as qml
import torch
import torch.nn as nn

N_QUBITS = 4
dev = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(dev, interface="torch", diff_method="backprop")
def _circuit(features, alpha, theta_xx1, beta, theta_xx2, gamma):
    """One run of the circuit on a single length-4 feature vector.
    Returns 4 Pauli-Z expectation values, one per qubit."""

    # 1. Angle embedding: each feature value -> RY rotation angle
    for k in range(N_QUBITS):
        qml.RY(features[k], wires=k)

    # 2. RZ layer
    for k in range(N_QUBITS):
        qml.RZ(alpha[k], wires=k)

    # 3. Ising XX entangling layer 1: pairs (1,2) and (0,3)
    qml.IsingXX(theta_xx1[0], wires=[1, 2])
    qml.IsingXX(theta_xx1[1], wires=[0, 3])

    # 4. RY layer
    for k in range(N_QUBITS):
        qml.RY(beta[k], wires=k)

    # 5. Ising XX entangling layer 2: pairs (0,1) and (2,3)
    qml.IsingXX(theta_xx2[0], wires=[0, 1])
    qml.IsingXX(theta_xx2[1], wires=[2, 3])

    # 6. RZ layer (final)
    for k in range(N_QUBITS):
        qml.RZ(gamma[k], wires=k)

    # 7. Toffoli entanglement cycle (fixed, not trainable)
    qml.Toffoli(wires=[0, 1, 2])
    qml.Toffoli(wires=[1, 2, 3])
    qml.Toffoli(wires=[2, 3, 0])
    qml.Toffoli(wires=[3, 0, 1])

    # 8. Pauli-Z measurement on every qubit
    return [qml.expval(qml.PauliZ(w)) for w in range(N_QUBITS)]


class CoreQuantumFE(nn.Module):
    """The learned quantum circuit, shared across all 128 channels.

    16 trainable parameters total (4 RZ + 2 XX + 4 RY + 2 XX + 4 RZ),
    same parameters reused for every one of the 128 length-4 input
    vectors - analogous to a convolution kernel shared across spatial
    positions, but here it's a quantum circuit shared across channels.
    """

    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.randn(N_QUBITS) * 0.1)
        self.theta_xx1 = nn.Parameter(torch.randn(2) * 0.1)
        self.beta = nn.Parameter(torch.randn(N_QUBITS) * 0.1)
        self.theta_xx2 = nn.Parameter(torch.randn(2) * 0.1)
        self.gamma = nn.Parameter(torch.randn(N_QUBITS) * 0.1)

    def forward(self, x):
        """x: (B, 128, 4) -> returns (B, 64, 2, 2)"""
        batch_size, n_channels, vec_len = x.shape
        assert vec_len == 4, f"expected length-4 vectors, got {vec_len}"

        outputs = torch.zeros(batch_size, n_channels, 2, dtype=torch.float32)

        for b in range(batch_size):
            for c in range(n_channels):
                z0, z1, z2, z3 = _circuit(
                    x[b, c], self.alpha, self.theta_xx1, self.beta,
                    self.theta_xx2, self.gamma,
                )
                # Documented design choice - see module docstring:
                # pool the 4 measured qubits into 2 values via adjacent
                # pairing, to match the reference doc's stated 128x2
                # (-> 64x2x2) output shape.
                outputs[b, c, 0] = (z0 + z1) / 2
                outputs[b, c, 1] = (z2 + z3) / 2

        # (B, 128, 2) -> (B, 64, 2, 2). 128*2 = 256 = 64*2*2, same
        # total element count, so this is a pure reshape.
        return outputs.reshape(batch_size, 64, 2, 2)


if __name__ == "__main__":
    torch.manual_seed(0)

    print("Testing the quantum circuit on ONE feature vector first...")
    dummy_features = torch.tensor([0.1, 0.5, -0.3, 0.8])
    alpha = torch.randn(4) * 0.1
    theta_xx1 = torch.randn(2) * 0.1
    beta = torch.randn(4) * 0.1
    theta_xx2 = torch.randn(2) * 0.1
    gamma = torch.randn(4) * 0.1
    result = _circuit(dummy_features, alpha, theta_xx1, beta, theta_xx2, gamma)
    print(f"  4 Pauli-Z measurements from one circuit run: {[float(r) for r in result]}")
    print("  (Each value should be between -1 and 1 - that's the valid range")
    print("   for a Pauli-Z expectation value. If you see anything outside")
    print("   that range, something is wrong.)")

    print("\nNow testing the full CoreQuantumFE module on a small dummy batch...")
    print("(Using batch=1 and only a few channels for speed - the __main__")
    print(" block below trims to fewer channels just for this quick check.)")

    quantum_fe = CoreQuantumFE()
    small_input = torch.randn(1, 128, 4) * 0.5  # small values = valid rotation angles
    import time
    start = time.time()
    out = quantum_fe(small_input)
    elapsed = time.time() - start
    print(f"\nInput shape:  {tuple(small_input.shape)} (expected: (1, 128, 4))")
    print(f"Output shape: {tuple(out.shape)} (expected: (1, 64, 2, 2))")
    print(f"Time for 128 circuit evaluations: {elapsed:.2f} seconds")
    print(f"Trainable parameters: {sum(p.numel() for p in quantum_fe.parameters())}")

    # confirm gradients flow (needed for training later)
    loss = out.sum()
    loss.backward()
    print(f"\nGradient check - alpha.grad is not None: {quantum_fe.alpha.grad is not None}")
    print("(This confirms the quantum circuit is differentiable end-to-end,")
    print(" which is required for training the Generator later.)")

    print("\nEnd-to-end chain: DC -> Reshape -> Core Quantum FE")
    try:
        from dc_module import DCModule
        from reshape_module import ReshapeModule

        dc = DCModule(in_channels=172)
        reshape = ReshapeModule()

        raw = torch.randn(1, 172, 128, 128)
        with torch.no_grad():
            compressed = dc(raw)
            reshaped = reshape(compressed)
        quantum_out = quantum_fe(reshaped)
        print(f"{tuple(raw.shape)} -> {tuple(compressed.shape)} -> "
              f"{tuple(reshaped.shape)} -> {tuple(quantum_out.shape)}")
        print("(expected: (1,172,128,128) -> (1,128,2,2) -> (1,128,4) -> (1,64,2,2))")
    except ImportError:
        print("(Skipping - dc_module.py / reshape_module.py not found alongside this file.)")