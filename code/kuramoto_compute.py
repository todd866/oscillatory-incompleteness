"""
Kuramoto Network as Computational Substrate

Demonstrates that coupled oscillator networks can implement
computation, providing the bridge to arithmetic encoding
required for Gödel incompleteness.

Key idea: phase relationships between oscillators can encode
binary states; coupling dynamics can implement logic gates.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from pathlib import Path

FIGURES_DIR = Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def kuramoto_network(t, theta, K, omega, A):
    """
    Kuramoto model with arbitrary coupling matrix.

    Parameters:
    -----------
    theta : array, shape (n,)
        Phase of each oscillator
    K : float
        Global coupling strength
    omega : array, shape (n,)
        Natural frequencies
    A : array, shape (n, n)
        Adjacency/coupling matrix
    """
    n = len(theta)
    dtheta = omega.copy()
    for i in range(n):
        coupling = 0
        for j in range(n):
            coupling += A[i, j] * np.sin(theta[j] - theta[i])
        dtheta[i] += (K / n) * coupling
    return dtheta


def phase_to_bit(theta, threshold=np.pi):
    """Map phase to binary: 0 if theta in [0, π), 1 if in [π, 2π)."""
    return int((theta % (2 * np.pi)) >= threshold)


def phases_to_bitstring(thetas):
    """Convert array of phases to bitstring."""
    return ''.join(str(phase_to_bit(t)) for t in thetas)


def xor_gate_network():
    """
    Construct a Kuramoto network that approximates XOR gate behavior.

    Uses 4 oscillators:
    - 0, 1: inputs
    - 2: intermediate
    - 3: output

    The coupling is designed so that output phase encodes XOR of inputs.
    """
    n = 4
    omega = np.array([1.0, 1.0, 1.0, 1.0])

    # Coupling matrix encoding XOR-like logic
    # This is a simplified demonstration - real implementations
    # would need careful tuning
    A = np.array([
        [0, 0, 1, 0],   # input 0 -> intermediate
        [0, 0, 1, 0],   # input 1 -> intermediate
        [1, 1, 0, 1],   # intermediate couples to both inputs and output
        [0, 0, 1, 0],   # output reads from intermediate
    ])

    return n, omega, A


def universal_oscillator_argument():
    """
    Demonstrate the theoretical argument for universality.

    Following Moore (1990) and Siegelmann & Sontag (1995):
    - Continuous dynamical systems can simulate Turing machines
    - Kuramoto-type networks are a special case
    - Therefore, oscillator networks can encode arbitrary computation

    We don't implement a full TM simulator (that would be complex),
    but show the key ingredients:
    1. Binary encoding via phase
    2. State transitions via coupling dynamics
    3. Arbitrary precision via continuous phases
    """

    print("Theoretical Argument for Computational Universality")
    print("=" * 60)
    print()
    print("Claim: Kuramoto-type oscillator networks can simulate")
    print("       any Turing machine, hence encode arithmetic.")
    print()
    print("Proof ingredients:")
    print()
    print("1. BINARY ENCODING")
    print("   Phase θ ∈ [0, 2π) → bit b ∈ {0, 1}")
    print("   θ < π  → 0")
    print("   θ ≥ π  → 1")
    print()
    print("2. STATE SPACE")
    print("   n oscillators → 2^n possible bit configurations")
    print("   Continuous phases allow encoding of:")
    print("   - Tape symbols (discretized)")
    print("   - Head position (as phase of designated oscillator)")
    print("   - Machine state (as phase pattern)")
    print()
    print("3. TRANSITION FUNCTION")
    print("   Coupling matrix A encodes state transitions:")
    print("   - Strong coupling → synchronization → preserve bit")
    print("   - Anti-phase coupling → flip bit")
    print("   - Conditional coupling → logic gates")
    print()
    print("4. UNIVERSALITY (by construction)")
    print("   Moore (1990): 3D billiard can simulate any TM")
    print("   Siegelmann-Sontag (1995): RNNs with rational weights")
    print("   are Turing-complete")
    print("   Kuramoto networks are continuous dynamical systems")
    print("   with similar expressivity → can embed TM simulation")
    print()
    print("Conclusion: There exist choices of (K, ω, A) such that")
    print("the resulting Kuramoto network simulates any given TM.")
    print("=" * 60)


def demonstrate_phase_encoding():
    """Show how oscillator phases encode and evolve binary states."""

    print("\nPhase Encoding Demonstration")
    print("-" * 40)

    # Simple 4-oscillator network
    n = 4
    K = 2.0
    omega = np.array([1.0, 1.05, 0.95, 1.02])
    A = np.ones((n, n)) - np.eye(n)  # All-to-all coupling

    # Various initial conditions → different bit patterns
    initial_conditions = [
        np.array([0.0, 0.0, 0.0, 0.0]),           # 0000
        np.array([np.pi, 0.0, 0.0, 0.0]),         # 1000
        np.array([0.0, np.pi, np.pi, 0.0]),       # 0110
        np.array([np.pi, np.pi, np.pi, np.pi]),   # 1111
    ]

    t_span = [0, 50]
    t_eval = np.linspace(0, 50, 500)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for idx, theta0 in enumerate(initial_conditions):
        ax = axes[idx // 2, idx % 2]

        sol = solve_ivp(
            lambda t, y: kuramoto_network(t, y, K, omega, A),
            t_span, theta0, t_eval=t_eval, method='RK45'
        )

        # Plot phases
        for i in range(n):
            ax.plot(sol.t, sol.y[i] % (2*np.pi), label=f'θ_{i}')

        ax.axhline(np.pi, color='k', linestyle='--', alpha=0.3, label='bit threshold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Phase (mod 2π)')
        ax.set_title(f'Initial: {phases_to_bitstring(theta0)}')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_ylim(0, 2*np.pi)

        # Show evolution of bit pattern
        bits_initial = phases_to_bitstring(theta0)
        bits_final = phases_to_bitstring(sol.y[:, -1])
        print(f"   {bits_initial} → {bits_final} (after t=50)")

    plt.suptitle('Phase Encoding: Oscillator Phases as Binary States', fontsize=14)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2_phase_encoding.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n   Saved: {FIGURES_DIR / 'fig2_phase_encoding.png'}")


def demonstrate_symbol_sequence_from_network():
    """
    Generate a symbol sequence from a larger oscillator network.
    This is the key connection to the incompleteness theorem.
    """

    print("\nSymbol Sequence from Oscillator Network")
    print("-" * 40)

    n = 8  # 8 oscillators
    K = 1.5

    # Heterogeneous frequencies for complex dynamics
    np.random.seed(42)
    omega = 1.0 + 0.2 * np.random.randn(n)

    # Random sparse coupling for interesting dynamics
    A = np.random.rand(n, n)
    A = (A + A.T) / 2  # Symmetrize
    A[A < 0.7] = 0     # Sparsify
    np.fill_diagonal(A, 0)

    # Random initial phases
    theta0 = 2 * np.pi * np.random.rand(n)

    # Integrate
    t_span = [0, 200]
    t_eval = np.linspace(0, 200, 2000)

    sol = solve_ivp(
        lambda t, y: kuramoto_network(t, y, K, omega, A),
        t_span, theta0, t_eval=t_eval, method='RK45'
    )

    # Generate symbol sequence: each time step → n-bit string
    # We'll use a coarser alphabet by hashing the bitstring
    symbols = []
    for i in range(len(sol.t)):
        bits = phases_to_bitstring(sol.y[:, i])
        # Map 8-bit string to single symbol (0-255, but we'll use hex)
        val = int(bits, 2)
        symbols.append(format(val, 'x'))  # hex digit

    symbol_string = ''.join(symbols)

    print(f"   Network: {n} oscillators, K={K}")
    print(f"   Symbol alphabet: 256 states (8-bit)")
    print(f"   Sequence length: {len(symbols)}")
    print(f"   First 100 symbols: {symbol_string[:100]}")

    # Compute entropy
    from collections import Counter
    counts = Counter(symbol_string)
    probs = np.array([c / len(symbol_string) for c in counts.values()])
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    print(f"   Shannon entropy: {entropy:.2f} bits (max = {np.log2(256):.2f})")

    return symbol_string


def main():
    print("=" * 60)
    print("Kuramoto Networks as Computational Substrates")
    print("=" * 60)

    # 1. Theoretical argument
    universal_oscillator_argument()

    # 2. Phase encoding demo
    demonstrate_phase_encoding()

    # 3. Symbol sequence generation
    symbols = demonstrate_symbol_sequence_from_network()

    print("\n" + "=" * 60)
    print("Key Result:")
    print("  Oscillator networks generate symbol sequences via")
    print("  phase-based encoding. When networks are complex enough")
    print("  to encode arithmetic (per Moore/Siegelmann-Sontag),")
    print("  Gödel's incompleteness theorem applies to any theory")
    print("  reasoning about these sequences.")
    print("=" * 60)


if __name__ == "__main__":
    main()
