"""
Demonstration of GPU Speedup for TWA Simulations

This script compares CPU vs GPU performance for H2 and H2O TWA simulations.
Shows the dramatic speedup achieved by GPU acceleration.
"""

import numpy as np
import time
import matplotlib.pyplot as plt

def format_time(seconds):
    """Format time in human-readable form."""
    if seconds < 60:
        return f"{seconds:.2f} s"
    elif seconds < 3600:
        return f"{seconds/60:.2f} min"
    else:
        return f"{seconds/3600:.2f} hr"


def benchmark_h2_cpu_vs_gpu():
    """
    Benchmark H2 simulation: CPU vs GPU
    """
    print("=" * 80)
    print("H2 MOLECULE: CPU vs GPU BENCHMARK")
    print("=" * 80)

    # Parameters
    r = 0.74
    total_time = 10.0
    n_steps = 100
    n_trajectories = 1000

    print(f"\nSimulation parameters:")
    print(f"  Molecule: H2 (4 qubits)")
    print(f"  Bond distance: {r} Å")
    print(f"  Time steps: {n_steps}")
    print(f"  Total time: {total_time} a.u.")
    print(f"  Trajectories: {n_trajectories}")

    # CPU Version
    print(f"\n{'='*80}")
    print("RUNNING CPU VERSION")
    print("=" * 80)

    try:
        from rich_sim_h2_twa import H2_TWA_Simulator

        h2_cpu = H2_TWA_Simulator(n_trajectories=n_trajectories)

        start = time.time()
        results_cpu = h2_cpu.simulate_twa_dynamics(
            r=r,
            total_time=total_time,
            n_steps=n_steps,
            add_T1=True,
            add_T2=True
        )
        cpu_time = time.time() - start

        print(f"\n✓ CPU simulation complete")
        print(f"  Total time: {format_time(cpu_time)}")
        print(f"  Time per trajectory: {cpu_time/n_trajectories:.4f} s")
        print(f"  Time per step: {cpu_time/n_steps:.4f} s")
        print(f"  Final energy: {results_cpu['avg_energies'][-1]:.6f} ± {results_cpu['std_energies'][-1]:.6f} H")

    except Exception as e:
        print(f"\n⚠ CPU version failed: {e}")
        cpu_time = None
        results_cpu = None

    # GPU Version
    print(f"\n{'='*80}")
    print("RUNNING GPU VERSION")
    print("=" * 80)

    try:
        from cudaq_rich_sim_h2_twa import CUDAQ_H2_TWA_Simulator

        h2_gpu = CUDAQ_H2_TWA_Simulator(n_trajectories=n_trajectories, use_gpu=True)

        start = time.time()
        results_gpu = h2_gpu.simulate_twa_dynamics(
            r=r,
            total_time=total_time,
            n_steps=n_steps,
            add_T1=True,
            add_T2=True
        )
        gpu_time = time.time() - start

        print(f"\n✓ GPU simulation complete")
        print(f"  Total time: {format_time(gpu_time)}")
        print(f"  Time per trajectory: {gpu_time/n_trajectories:.4f} s")
        print(f"  Time per step: {gpu_time/n_steps:.4f} s")
        print(f"  Final energy: {results_gpu['avg_energies'][-1]:.6f} ± {results_gpu['std_energies'][-1]:.6f} H")

    except ImportError:
        print("\n⚠ CuPy not installed - GPU version unavailable")
        print("  Install with: pip install cupy-cuda12x")
        gpu_time = None
        results_gpu = None
    except Exception as e:
        print(f"\n⚠ GPU version failed: {e}")
        gpu_time = None
        results_gpu = None

    # Compare
    print(f"\n{'='*80}")
    print("PERFORMANCE COMPARISON")
    print("=" * 80)

    if cpu_time is not None and gpu_time is not None:
        speedup = cpu_time / gpu_time
        print(f"\n  CPU time:  {format_time(cpu_time)}")
        print(f"  GPU time:  {format_time(gpu_time)}")
        print(f"  Speedup:   {speedup:.2f}x")

        if speedup > 5:
            print(f"\n  🚀 Excellent speedup! GPU is {speedup:.1f}x faster")
        elif speedup > 2:
            print(f"\n  ✓ Good speedup! GPU is {speedup:.1f}x faster")
        else:
            print(f"\n  ⚠ Limited speedup. Consider increasing n_trajectories for better GPU utilization.")

        # Check result agreement
        if results_cpu is not None and results_gpu is not None:
            energy_diff = np.abs(results_cpu['avg_energies'][-1] - results_gpu['avg_energies'][-1])
            print(f"\n  Energy difference: {energy_diff:.2e} H (should be < 1e-3)")
            if energy_diff < 1e-3:
                print(f"  ✓ Results agree within statistical error")
            else:
                print(f"  ⚠ Large discrepancy - check implementation")

    elif cpu_time is not None:
        print(f"\n  CPU time: {format_time(cpu_time)}")
        print(f"  GPU not available")
    elif gpu_time is not None:
        print(f"\n  GPU time: {format_time(gpu_time)}")
        print(f"  CPU comparison not available")
    else:
        print("\n  ⚠ Neither CPU nor GPU version completed successfully")

    return {
        'cpu_time': cpu_time,
        'gpu_time': gpu_time,
        'speedup': cpu_time / gpu_time if (cpu_time and gpu_time) else None,
        'results_cpu': results_cpu,
        'results_gpu': results_gpu,
    }


def benchmark_h2o_cpu_vs_gpu():
    """
    Benchmark H2O simulation: CPU vs GPU
    """
    print("\n\n")
    print("=" * 80)
    print("H2O MOLECULE: CPU vs GPU BENCHMARK")
    print("=" * 80)

    # Parameters (smaller for H2O due to 10 qubits)
    total_time = 5.0
    n_steps = 200
    n_trajectories = 500  # Smaller for CPU version
    energy_scale = 1e15

    print(f"\nSimulation parameters:")
    print(f"  Molecule: H2O (10 qubits)")
    print(f"  Time steps: {n_steps}")
    print(f"  Total time: {total_time} a.u.")
    print(f"  Trajectories: {n_trajectories} (CPU), {n_trajectories*2} (GPU)")
    print(f"  Energy scale: {energy_scale:.0e}")

    # CPU Version
    print(f"\n{'='*80}")
    print("RUNNING CPU VERSION")
    print("=" * 80)

    try:
        from rich_sim_h2o_twa import H2O_TWA_Simulator

        h2o_cpu = H2O_TWA_Simulator(n_trajectories=n_trajectories, energy_scale=energy_scale)

        start = time.time()
        results_cpu = h2o_cpu.simulate_twa_dynamics(
            total_time=total_time,
            n_steps=n_steps,
            add_T1=True,
            add_T2=True
        )
        cpu_time = time.time() - start

        print(f"\n✓ CPU simulation complete")
        print(f"  Total time: {format_time(cpu_time)}")
        print(f"  Time per trajectory: {cpu_time/n_trajectories:.4f} s")
        print(f"  Final energy: {results_cpu['avg_energies'][-1]:.4f} ± {results_cpu['std_energies'][-1]:.4f} H")

    except Exception as e:
        print(f"\n⚠ CPU version failed: {e}")
        cpu_time = None
        results_cpu = None

    # GPU Version (use more trajectories)
    print(f"\n{'='*80}")
    print("RUNNING GPU VERSION (with 2x trajectories)")
    print("=" * 80)

    try:
        from cudaq_rich_sim_h2o_twa import CUDAQ_H2O_TWA_Simulator

        h2o_gpu = CUDAQ_H2O_TWA_Simulator(
            n_trajectories=n_trajectories*2,
            energy_scale=energy_scale,
            use_gpu=True
        )

        start = time.time()
        results_gpu = h2o_gpu.simulate_twa_dynamics(
            total_time=total_time,
            n_steps=n_steps,
            add_T1=True,
            add_T2=True
        )
        gpu_time = time.time() - start

        print(f"\n✓ GPU simulation complete")
        print(f"  Total time: {format_time(gpu_time)}")
        print(f"  Time per trajectory: {gpu_time/(n_trajectories*2):.4f} s")
        print(f"  Final energy: {results_gpu['avg_energies'][-1]:.4f} ± {results_gpu['std_energies'][-1]:.4f} H")

    except ImportError:
        print("\n⚠ CuPy not installed - GPU version unavailable")
        gpu_time = None
        results_gpu = None
    except Exception as e:
        print(f"\n⚠ GPU version failed: {e}")
        gpu_time = None
        results_gpu = None

    # Compare
    print(f"\n{'='*80}")
    print("PERFORMANCE COMPARISON")
    print("=" * 80)

    if cpu_time is not None and gpu_time is not None:
        # Effective speedup accounting for more trajectories
        effective_speedup = (cpu_time * 2) / gpu_time  # GPU did 2x work
        print(f"\n  CPU time:  {format_time(cpu_time)} ({n_trajectories} traj)")
        print(f"  GPU time:  {format_time(gpu_time)} ({n_trajectories*2} traj)")
        print(f"  Effective speedup: {effective_speedup:.2f}x")
        print(f"  (GPU ran 2x trajectories in {gpu_time/cpu_time:.2f}x the time)")

        if effective_speedup > 5:
            print(f"\n  🚀 Excellent! GPU gives {effective_speedup:.1f}x better throughput")
        elif effective_speedup > 2:
            print(f"\n  ✓ Good! GPU gives {effective_speedup:.1f}x better throughput")

    elif cpu_time is not None:
        print(f"\n  CPU time: {format_time(cpu_time)}")
        print(f"  GPU not available")
    elif gpu_time is not None:
        print(f"\n  GPU time: {format_time(gpu_time)}")
        print(f"  CPU comparison not available")

    return {
        'cpu_time': cpu_time,
        'gpu_time': gpu_time,
        'results_cpu': results_cpu,
        'results_gpu': results_gpu,
    }


def plot_speedup_scaling():
    """
    Plot speedup vs number of trajectories (theoretical).
    """
    print("\n\n")
    print("=" * 80)
    print("THEORETICAL SPEEDUP SCALING")
    print("=" * 80)

    # Theoretical model: speedup saturates due to overhead
    n_traj_range = np.logspace(1, 4, 20)  # 10 to 10,000 trajectories
    max_speedup = 50  # Theoretical maximum
    overhead = 2.0  # Seconds of fixed overhead

    # Model: speedup increases with trajectories but saturates
    cpu_time = 0.1 * n_traj_range + overhead  # Linear in trajectories
    gpu_time = overhead + 0.002 * n_traj_range  # Much lower per-trajectory cost
    speedup = cpu_time / gpu_time

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Time vs trajectories
    ax = axes[0]
    ax.loglog(n_traj_range, cpu_time, 'b-', linewidth=2, label='CPU')
    ax.loglog(n_traj_range, gpu_time, 'r-', linewidth=2, label='GPU')
    ax.set_xlabel('Number of Trajectories')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Execution Time vs Trajectory Count')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Speedup vs trajectories
    ax = axes[1]
    ax.semilogx(n_traj_range, speedup, 'g-', linewidth=3, label='GPU Speedup')
    ax.axhline(y=10, color='gray', linestyle='--', alpha=0.5, label='10x speedup')
    ax.axhline(y=max_speedup, color='gray', linestyle='--', alpha=0.5, label='Maximum')
    ax.fill_between(n_traj_range, 0, speedup, alpha=0.2, color='green')
    ax.set_xlabel('Number of Trajectories')
    ax.set_ylabel('Speedup (CPU time / GPU time)')
    ax.set_title('GPU Speedup vs Trajectory Count')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, max_speedup * 1.1])

    plt.suptitle('TWA GPU Acceleration: Theoretical Performance', fontsize=14, fontweight='bold')
    plt.tight_layout()

    print("\n  Theoretical model:")
    print(f"    Small systems (100 traj):   ~{speedup[5]:.1f}x speedup")
    print(f"    Medium systems (1000 traj):  ~{speedup[10]:.1f}x speedup")
    print(f"    Large systems (5000 traj):   ~{speedup[15]:.1f}x speedup")
    print("\n  Key insight: GPU advantage grows with trajectory count!")

    plt.show()


def main():
    """
    Run all benchmarks and display results.
    """
    print("\n")
    print("*" * 80)
    print("*" + " " * 78 + "*")
    print("*" + " " * 20 + "CUDA-Q TWA SPEEDUP DEMONSTRATION" + " " * 26 + "*")
    print("*" + " " * 78 + "*")
    print("*" * 80)
    print()
    print("This script demonstrates GPU acceleration for TWA simulations.")
    print("It compares CPU vs GPU performance for H2 and H2O molecules.")
    print()
    print("Note: First GPU run may be slow due to kernel compilation.")
    print("      Subsequent runs will be faster with cached kernels.")
    print()
    input("Press Enter to start benchmarks...")

    # Run H2 benchmark
    h2_results = benchmark_h2_cpu_vs_gpu()

    # Run H2O benchmark
    h2o_results = benchmark_h2o_cpu_vs_gpu()

    # Plot theoretical scaling
    try:
        plot_speedup_scaling()
    except:
        print("\n⚠ Could not generate scaling plot")

    # Final summary
    print("\n\n")
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print("\nH2 Molecule (4 qubits):")
    if h2_results['speedup'] is not None:
        print(f"  GPU speedup: {h2_results['speedup']:.2f}x")
    else:
        print(f"  Benchmark incomplete")

    print("\nH2O Molecule (10 qubits):")
    if h2o_results['cpu_time'] and h2o_results['gpu_time']:
        effective = (h2o_results['cpu_time'] * 2) / h2o_results['gpu_time']
        print(f"  GPU effective speedup: {effective:.2f}x (with 2x trajectories)")
    else:
        print(f"  Benchmark incomplete")

    print("\nRecommendations:")
    print("  - Use GPU for trajectory counts >= 1000")
    print("  - Larger systems (more qubits) benefit more from GPU")
    print("  - GPU enables better statistics with more trajectories")
    print("  - Consider batching for very large trajectory counts (>10,000)")

    print("\n" + "=" * 80)
    print("✓ Benchmark complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
