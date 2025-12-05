"""
Oscillatory Systems → Symbol Sequences Demo

Demonstrates the core concept: high-dimensional continuous dynamics
produce discrete symbol sequences via coarse-grained observation.

This is the foundation for the "Oscillatory Incompleteness" paper:
when such systems can encode arithmetic, Gödel applies.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from pathlib import Path

# Output directory
FIGURES_DIR = Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def van_der_pol(t, z, mu=1.0):
    """Van der Pol oscillator: classic nonlinear oscillator with limit cycle."""
    x, y = z
    dxdt = y
    dydt = mu * (1 - x**2) * y - x
    return [dxdt, dydt]


def coupled_oscillators(t, z, n=3, K=0.5, omega=None):
    """
    Kuramoto-like coupled oscillators.
    Higher-dimensional dynamics for richer symbol sequences.
    """
    if omega is None:
        omega = np.linspace(0.9, 1.1, n)

    theta = z
    dtheta = np.zeros(n)
    for i in range(n):
        coupling = (K / n) * np.sum(np.sin(theta - theta[i]))
        dtheta[i] = omega[i] + coupling
    return dtheta


def integrate_system(dynamics, z0, t_max=200.0, dt=0.1, **kwargs):
    """Integrate a dynamical system."""
    t_eval = np.arange(0, t_max, dt)
    sol = solve_ivp(
        dynamics, [0, t_max], z0,
        t_eval=t_eval,
        args=tuple(kwargs.values()) if kwargs else (),
        method='RK45'
    )
    return sol.t, sol.y.T


def quadrant_partition(point):
    """
    Simple quadrant-based partition for 2D systems.
    Maps continuous state to one of 4 symbols.
    """
    x, y = point[:2]
    if x >= 0 and y >= 0:
        return '0'
    elif x < 0 and y >= 0:
        return '1'
    elif x < 0 and y < 0:
        return '2'
    else:
        return '3'


def phase_partition(theta, n_bins=4):
    """
    Phase-based partition for oscillator networks.
    Bins phase into discrete symbols.
    """
    # Wrap to [0, 2π)
    theta_wrapped = theta % (2 * np.pi)
    bin_idx = int(theta_wrapped / (2 * np.pi) * n_bins)
    return str(min(bin_idx, n_bins - 1))


def trajectory_to_symbols(trajectory, partition_fn):
    """Convert continuous trajectory to symbol sequence."""
    return ''.join(partition_fn(p) for p in trajectory)


def symbol_statistics(symbols):
    """Compute basic statistics of symbol sequence."""
    from collections import Counter
    counts = Counter(symbols)
    total = len(symbols)

    # Shannon entropy
    probs = np.array([c / total for c in counts.values()])
    entropy = -np.sum(probs * np.log2(probs + 1e-10))

    # Transition matrix (bigram)
    transitions = {}
    for i in range(len(symbols) - 1):
        pair = (symbols[i], symbols[i+1])
        transitions[pair] = transitions.get(pair, 0) + 1

    return {
        'length': total,
        'unique_symbols': len(counts),
        'entropy': entropy,
        'symbol_counts': dict(counts),
        'n_transitions': len(transitions)
    }


def demonstrate_sensitivity(dynamics, z0_base, perturbations, t_max=100.0, **kwargs):
    """
    Show how small changes in initial conditions produce
    drastically different symbol sequences.

    This is key: the symbolic layer inherits sensitivity from
    the continuous dynamics, but in a coarse-grained way.
    """
    results = []
    for eps in perturbations:
        z0 = z0_base.copy()
        z0[0] += eps
        t, traj = integrate_system(dynamics, z0, t_max=t_max, **kwargs)
        symbols = trajectory_to_symbols(traj, quadrant_partition)
        results.append({
            'perturbation': eps,
            'symbols': symbols,
            'trajectory': traj
        })
    return results


def plot_oscillator_to_symbols(t, trajectory, symbols, title="Oscillator → Symbols"):
    """
    Create the key visualization: continuous dynamics + partition + symbol timeline.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Phase portrait with partition
    ax1 = axes[0, 0]
    ax1.plot(trajectory[:, 0], trajectory[:, 1], 'b-', alpha=0.5, linewidth=0.5)
    ax1.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax1.axvline(0, color='k', linestyle='--', alpha=0.3)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Phase Portrait with Partition')

    # Add quadrant labels
    ax1.text(0.7, 0.7, '0', transform=ax1.transAxes, fontsize=20, alpha=0.3)
    ax1.text(0.2, 0.7, '1', transform=ax1.transAxes, fontsize=20, alpha=0.3)
    ax1.text(0.2, 0.2, '2', transform=ax1.transAxes, fontsize=20, alpha=0.3)
    ax1.text(0.7, 0.2, '3', transform=ax1.transAxes, fontsize=20, alpha=0.3)

    # Time series
    ax2 = axes[0, 1]
    ax2.plot(t, trajectory[:, 0], 'b-', label='x(t)')
    ax2.plot(t, trajectory[:, 1], 'r-', alpha=0.7, label='y(t)')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('State')
    ax2.set_title('Time Series')
    ax2.legend()

    # Symbol sequence (as colored timeline)
    ax3 = axes[1, 0]
    symbol_to_color = {'0': 'C0', '1': 'C1', '2': 'C2', '3': 'C3'}
    colors = [symbol_to_color[s] for s in symbols]

    # Plot as colored bars
    n_show = min(500, len(symbols))
    for i in range(n_show):
        ax3.axvspan(i, i+1, color=colors[i], alpha=0.7)
    ax3.set_xlim(0, n_show)
    ax3.set_xlabel('Time step')
    ax3.set_title(f'Symbol Sequence (first {n_show} symbols)')
    ax3.set_yticks([])

    # Symbol statistics
    ax4 = axes[1, 1]
    stats = symbol_statistics(symbols)
    stats_text = f"""Symbol Sequence Statistics

Length: {stats['length']}
Unique symbols: {stats['unique_symbols']}
Shannon entropy: {stats['entropy']:.3f} bits

Symbol frequencies:
"""
    for sym, count in sorted(stats['symbol_counts'].items()):
        stats_text += f"  '{sym}': {count} ({100*count/stats['length']:.1f}%)\n"

    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace')
    ax4.axis('off')
    ax4.set_title('Statistics')

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig


def main():
    print("=" * 60)
    print("Oscillatory Systems → Symbol Sequences")
    print("=" * 60)

    # 1. Van der Pol oscillator
    print("\n1. Van der Pol Oscillator Demo")
    print("-" * 40)

    z0 = [1.0, 0.0]
    t, traj = integrate_system(van_der_pol, z0, t_max=200.0, dt=0.1, mu=1.0)
    symbols = trajectory_to_symbols(traj, quadrant_partition)

    print(f"   Trajectory length: {len(traj)}")
    print(f"   Symbol sequence length: {len(symbols)}")
    print(f"   First 100 symbols: {symbols[:100]}")

    stats = symbol_statistics(symbols)
    print(f"   Shannon entropy: {stats['entropy']:.3f} bits")

    # Plot
    fig = plot_oscillator_to_symbols(t, traj, symbols,
                                      "Van der Pol Oscillator → Symbol Sequence")
    fig.savefig(FIGURES_DIR / "fig1_vdp_symbols.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {FIGURES_DIR / 'fig1_vdp_symbols.png'}")

    # 2. Sensitivity to initial conditions
    print("\n2. Sensitivity Analysis")
    print("-" * 40)

    perturbations = [0, 1e-6, 1e-4, 1e-2]
    results = demonstrate_sensitivity(van_der_pol, np.array([1.0, 0.0]),
                                       perturbations, t_max=100.0, mu=1.0)

    for r in results:
        print(f"   ε = {r['perturbation']:.0e}: {r['symbols'][:50]}...")

    # 3. Coupled oscillators (higher-D)
    print("\n3. Coupled Oscillator Network")
    print("-" * 40)

    n_osc = 5
    z0_kuramoto = np.random.uniform(0, 2*np.pi, n_osc)
    t, traj = integrate_system(coupled_oscillators, z0_kuramoto,
                                t_max=100.0, dt=0.1, n=n_osc, K=0.5)

    # Multi-oscillator symbol: concatenate phase bins
    symbols_kuramoto = ''
    for point in traj:
        sym = ''.join(phase_partition(theta, n_bins=2) for theta in point)
        symbols_kuramoto += sym

    print(f"   {n_osc} coupled oscillators")
    print(f"   Symbol alphabet size: 2^{n_osc} = {2**n_osc}")
    print(f"   Sample symbols: {symbols_kuramoto[:100]}")

    print("\n" + "=" * 60)
    print("Demo complete. Key insight:")
    print("  Continuous oscillatory dynamics → discrete symbol sequences")
    print("  via coarse-grained observation (partition + sampling).")
    print("  When such systems encode arithmetic, Gödel applies.")
    print("=" * 60)


if __name__ == "__main__":
    main()
