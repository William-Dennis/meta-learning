"""
Create a visual chart comparing all 4 implementations
"""
import matplotlib.pyplot as plt
import numpy as np

# Data from benchmark
steps = [10, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
py_serial = [0.0149, 0.0507, 0.0999, 0.2015, 0.5141, 0.9784, 1.9428, 4.7767, 9.5679]
py_parallel = [0.0546, 0.0380, 0.0680, 0.1189, 0.2696, 0.4955, 1.0474, 2.3546, 4.7192]
rust_serial = [0.0002, 0.0006, 0.0011, 0.0024, 0.0058, 0.0106, 0.0198, 0.0482, 0.0956]
rust_parallel = [0.0004, 0.0005, 0.0006, 0.0010, 0.0027, 0.0046, 0.0087, 0.0217, 0.0436]

# Create figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Log-scale time comparison
ax1.plot(steps, py_serial, 'o-', label='Python Serial', linewidth=2, markersize=8)
ax1.plot(steps, py_parallel, 's-', label='Python Parallel', linewidth=2, markersize=8)
ax1.plot(steps, rust_serial, '^-', label='Rust Serial', linewidth=2, markersize=8)
ax1.plot(steps, rust_parallel, 'd-', label='Rust Parallel', linewidth=2, markersize=8)
ax1.set_xlabel('Number of SA Steps', fontsize=12, fontweight='bold')
ax1.set_ylabel('Execution Time (seconds, log scale)', fontsize=12, fontweight='bold')
ax1.set_title('Performance Comparison: Python vs Rust SA Implementations\n(100 runs per test)', 
              fontsize=14, fontweight='bold')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=11, loc='upper left')

# Plot 2: Speedup comparison
speedup_py_par = [py_serial[i] / py_parallel[i] for i in range(len(steps))]
speedup_rust_ser = [py_serial[i] / rust_serial[i] for i in range(len(steps))]
speedup_rust_par = [py_serial[i] / rust_parallel[i] for i in range(len(steps))]

x_pos = np.arange(len(steps))
width = 0.25

bars1 = ax2.bar(x_pos - width, speedup_py_par, width, label='Python Parallel', alpha=0.8)
bars2 = ax2.bar(x_pos, speedup_rust_ser, width, label='Rust Serial', alpha=0.8)
bars3 = ax2.bar(x_pos + width, speedup_rust_par, width, label='Rust Parallel', alpha=0.8)

ax2.set_xlabel('Number of SA Steps', fontsize=12, fontweight='bold')
ax2.set_ylabel('Speedup (vs Python Serial)', fontsize=12, fontweight='bold')
ax2.set_title('Speedup Analysis: All Implementations vs Python Serial', 
              fontsize=14, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(steps, rotation=45)
ax2.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='Baseline (1x)')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels on bars for key points
for i in [2, 5, 8]:  # Show values for 100, 1000, 10000 steps
    ax2.text(i - width, speedup_py_par[i] + 5, f'{speedup_py_par[i]:.1f}x', 
             ha='center', va='bottom', fontsize=9)
    ax2.text(i, speedup_rust_ser[i] + 5, f'{speedup_rust_ser[i]:.0f}x', 
             ha='center', va='bottom', fontsize=9)
    ax2.text(i + width, speedup_rust_par[i] + 5, f'{speedup_rust_par[i]:.0f}x', 
             ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('performance_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Saved performance_comparison.png")

# Create summary statistics image
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')

summary_text = """
SA ALGORITHM PERFORMANCE SUMMARY
═══════════════════════════════════════════════════════════════

AVERAGE SPEEDUPS (vs Python Serial):
────────────────────────────────────────────────────────────────
  Python Parallel:    1.69x
  Rust Serial:       91.39x  
  Rust Parallel:    170.75x  🚀

PARALLELIZATION EFFICIENCY:
────────────────────────────────────────────────────────────────
  Python (multiprocessing):  1.69x speedup
  Rust (native threads):     1.87x speedup

EXAMPLE: 10,000 SA STEPS
────────────────────────────────────────────────────────────────
  Python Serial:     9.57 seconds
  Python Parallel:   4.72 seconds  (2.0x faster)
  Rust Serial:       0.10 seconds  (99x faster)
  Rust Parallel:     0.04 seconds  (220x faster) ⚡

KEY FINDINGS:
────────────────────────────────────────────────────────────────
  ✓ Rust is 90-100x faster than Python even in serial mode
  ✓ Rust parallel achieves 170x average speedup
  ✓ Python multiprocessing limited by GIL and IPC overhead
  ✓ Rust threading scales better without GIL constraints
  ✓ All implementations produce equivalent results

TECHNICAL STACK:
────────────────────────────────────────────────────────────────
  • PyO3 0.22 for Python-Rust bindings
  • Maturin for building extension modules
  • UV for fast Python package management
  • Native Rust threads for parallelization
"""

ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
        fontsize=11, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig('performance_summary.png', dpi=150, bbox_inches='tight')
print("✓ Saved performance_summary.png")

print("\n✓ Created performance visualization charts")
