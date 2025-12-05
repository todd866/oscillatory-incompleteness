"""
The Gödel Oscillator: Self-Reference in Dynamical Systems

Demonstrates how self-reference (the mechanism behind Gödel's theorem)
can be physically instantiated in an oscillatory system.

Key idea: an oscillator whose future dynamics depend on an internal
model of its own past symbol output. This creates a physical analog
of "This sentence is not provable."
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from pathlib import Path
from collections import deque

FIGURES_DIR = Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


class GodelOscillator:
    """
    An oscillator that modifies its dynamics based on predictions
    about its own symbol sequence.

    The system:
    1. Generates symbols via phase partition (as before)
    2. Maintains a model of its recent symbol history
    3. Predicts the next symbol using a simple n-gram model
    4. Adjusts its frequency to AVOID the predicted symbol

    This creates a self-referential loop: the system's behavior
    depends on its model of its own behavior.
    """

    def __init__(self, n_gram=3, history_length=100, base_freq=1.0):
        self.n_gram = n_gram
        self.history_length = history_length
        self.base_freq = base_freq

        # Symbol history
        self.history = deque(maxlen=history_length)

        # N-gram transition counts
        self.transitions = {}

        # Current phase
        self.phase = 0.0

        # Frequency modulation strength
        self.mod_strength = 0.3

    def phase_to_symbol(self, phase):
        """Map phase to symbol (4 symbols based on quadrant)."""
        p = phase % (2 * np.pi)
        if p < np.pi / 2:
            return '0'
        elif p < np.pi:
            return '1'
        elif p < 3 * np.pi / 2:
            return '2'
        else:
            return '3'

    def update_model(self, new_symbol):
        """Update n-gram model with new observation."""
        self.history.append(new_symbol)

        if len(self.history) >= self.n_gram:
            context = ''.join(list(self.history)[-self.n_gram:-1])
            outcome = new_symbol

            if context not in self.transitions:
                self.transitions[context] = {'0': 0, '1': 0, '2': 0, '3': 0}
            self.transitions[context][outcome] += 1

    def predict_next(self):
        """Predict most likely next symbol given recent history."""
        if len(self.history) < self.n_gram - 1:
            return None

        context = ''.join(list(self.history)[-(self.n_gram-1):])

        if context not in self.transitions:
            return None

        counts = self.transitions[context]
        total = sum(counts.values())
        if total == 0:
            return None

        # Return most likely symbol
        return max(counts.keys(), key=lambda k: counts[k])

    def compute_avoidance_frequency(self):
        """
        Compute frequency modulation to avoid predicted symbol.

        This is the self-referential mechanism: the oscillator
        tries to behave in a way its own model doesn't predict.
        """
        predicted = self.predict_next()

        if predicted is None:
            return self.base_freq

        # Current symbol
        current = self.phase_to_symbol(self.phase)

        # Target: a symbol different from predicted
        symbols = ['0', '1', '2', '3']
        alternatives = [s for s in symbols if s != predicted]

        # Compute target phase (center of target symbol's region)
        target_symbol = alternatives[0]  # Simple choice
        target_phases = {
            '0': np.pi / 4,
            '1': 3 * np.pi / 4,
            '2': 5 * np.pi / 4,
            '3': 7 * np.pi / 4
        }

        target_phase = target_phases[target_symbol]
        current_phase = self.phase % (2 * np.pi)

        # Adjust frequency to move toward target
        phase_diff = target_phase - current_phase
        if phase_diff > np.pi:
            phase_diff -= 2 * np.pi
        elif phase_diff < -np.pi:
            phase_diff += 2 * np.pi

        # Frequency modulation proportional to phase difference
        freq_mod = self.mod_strength * phase_diff
        return self.base_freq + freq_mod

    def step(self, dt=0.1):
        """Advance one time step."""
        # Compute frequency (with self-referential modulation)
        freq = self.compute_avoidance_frequency()

        # Update phase
        self.phase += freq * dt

        # Generate and record symbol
        symbol = self.phase_to_symbol(self.phase)
        self.update_model(symbol)

        return symbol, freq


def simulate_godel_oscillator(n_steps=1000, dt=0.1):
    """Run the Gödel oscillator and record its behavior."""

    osc = GodelOscillator(n_gram=3, history_length=100)

    symbols = []
    frequencies = []
    phases = []
    predictions = []

    for _ in range(n_steps):
        pred = osc.predict_next()
        predictions.append(pred)

        sym, freq = osc.step(dt)
        symbols.append(sym)
        frequencies.append(freq)
        phases.append(osc.phase % (2 * np.pi))

    return {
        'symbols': symbols,
        'frequencies': frequencies,
        'phases': phases,
        'predictions': predictions
    }


def analyze_self_reference(results):
    """Analyze how well the oscillator avoids its own predictions."""

    symbols = results['symbols']
    predictions = results['predictions']

    # Count prediction accuracy (which should be LOW for self-avoiding system)
    correct = 0
    total = 0

    for i in range(len(predictions)):
        if predictions[i] is not None:
            total += 1
            if predictions[i] == symbols[i]:
                correct += 1

    accuracy = correct / total if total > 0 else 0

    print(f"   Prediction accuracy: {100*accuracy:.1f}%")
    print(f"   (Low accuracy = successful self-avoidance)")

    # A random baseline would be 25% (4 symbols)
    # A good predictor of a regular oscillator might be >80%
    # Our self-avoiding oscillator should be <25%

    return accuracy


def plot_godel_oscillator(results, title="Gödel Oscillator"):
    """Visualize the self-referential oscillator."""

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    t = np.arange(len(results['symbols']))

    # Phase trajectory
    ax1 = axes[0]
    ax1.plot(t, results['phases'], 'b-', linewidth=0.5)
    ax1.set_ylabel('Phase (mod 2π)')
    ax1.set_title('Phase Evolution')

    # Add symbol boundaries
    for boundary in [np.pi/2, np.pi, 3*np.pi/2]:
        ax1.axhline(boundary, color='gray', linestyle='--', alpha=0.3)

    # Frequency modulation
    ax2 = axes[1]
    ax2.plot(t, results['frequencies'], 'r-', linewidth=0.5)
    ax2.axhline(1.0, color='k', linestyle='--', alpha=0.3, label='base freq')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Frequency Modulation (self-referential adjustment)')
    ax2.legend()

    # Symbol sequence
    ax3 = axes[2]
    symbol_to_num = {'0': 0, '1': 1, '2': 2, '3': 3}
    symbol_nums = [symbol_to_num[s] for s in results['symbols']]
    ax3.plot(t, symbol_nums, 'g-', linewidth=0.5, alpha=0.7)
    ax3.scatter(t[::10], [symbol_nums[i] for i in range(0, len(symbol_nums), 10)],
                c='green', s=2)
    ax3.set_ylabel('Symbol')
    ax3.set_xlabel('Time step')
    ax3.set_title('Symbol Sequence')
    ax3.set_yticks([0, 1, 2, 3])

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig


def compare_with_regular_oscillator():
    """
    Compare: regular oscillator (predictable) vs Gödel oscillator (self-avoiding).
    """

    print("\nComparison: Regular vs Gödel Oscillator")
    print("-" * 50)

    # Regular oscillator (no self-reference)
    print("\n1. Regular Oscillator (constant frequency):")
    phases_regular = []
    phase = 0.0
    freq = 1.0
    dt = 0.1

    for _ in range(1000):
        phase += freq * dt
        phases_regular.append(phase % (2 * np.pi))

    # Convert to symbols
    def phase_to_sym(p):
        if p < np.pi/2: return '0'
        elif p < np.pi: return '1'
        elif p < 3*np.pi/2: return '2'
        else: return '3'

    symbols_regular = [phase_to_sym(p) for p in phases_regular]

    # Measure predictability via compression
    from collections import Counter
    bigrams_regular = Counter(''.join(symbols_regular[i:i+2]) for i in range(len(symbols_regular)-1))
    print(f"   Unique bigrams: {len(bigrams_regular)}")
    print(f"   Most common: {bigrams_regular.most_common(3)}")

    # Gödel oscillator
    print("\n2. Gödel Oscillator (self-avoiding):")
    results = simulate_godel_oscillator(n_steps=1000)

    bigrams_godel = Counter(''.join(results['symbols'][i:i+2])
                            for i in range(len(results['symbols'])-1))
    print(f"   Unique bigrams: {len(bigrams_godel)}")
    print(f"   Most common: {bigrams_godel.most_common(3)}")

    analyze_self_reference(results)

    return results


def main():
    print("=" * 60)
    print("The Gödel Oscillator: Self-Reference in Dynamics")
    print("=" * 60)

    print("\nConcept:")
    print("  An oscillator that models its own symbol output and")
    print("  adjusts its dynamics to avoid predicted symbols.")
    print("  This is a physical instantiation of self-reference:")
    print("  'This system does not behave as its model predicts.'")
    print()

    # Run comparison
    results = compare_with_regular_oscillator()

    # Plot
    fig = plot_godel_oscillator(results)
    fig.savefig(FIGURES_DIR / "fig3_godel_oscillator.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n   Saved: {FIGURES_DIR / 'fig3_godel_oscillator.png'}")

    print("\n" + "=" * 60)
    print("Interpretation:")
    print("  The Gödel oscillator demonstrates how self-reference")
    print("  arises naturally in oscillatory systems with feedback.")
    print()
    print("  Just as Gödel's sentence G says 'G is not provable',")
    print("  our oscillator effectively says:")
    print("  'My next symbol is not what my model predicts.'")
    print()
    print("  This creates inherent unpredictability: any fixed model")
    print("  of the system is systematically evaded by the system")
    print("  itself. This is dynamical incompleteness.")
    print("=" * 60)


if __name__ == "__main__":
    main()
