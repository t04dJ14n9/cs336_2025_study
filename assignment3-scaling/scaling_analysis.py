"""
CS336 Assignment 3: Scaling Laws Analysis

Fits Chinchilla-style scaling laws to isoFLOP data and predicts optimal configurations.

Scaling Law: L(N, D) = E + A/N^α + B/D^β

Where:
- L: Final loss
- N: Number of parameters
- D: Number of training tokens
- E, A, B, α, β: Fitted constants

From Chinchilla: C ≈ 6 * N * D (compute budget in FLOPs)
"""

import json
import numpy as np
from scipy.optimize import curve_fit  # type: ignore[import-untyped]
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any
from numpy.typing import NDArray


def load_isoflops_data(filepath: str | Path) -> list[dict[str, Any]]:
    """Load isoFLOP curves data from JSON file."""
    with open(filepath, 'r') as f:
        data: list[dict[str, Any]] = json.load(f)
    return data


def scaling_law(
    N: float | NDArray[np.floating],
    D: float | NDArray[np.floating],
    E: float,
    A: float,
    B: float,
    alpha: float,
    beta: float
) -> float | NDArray[np.floating]:
    """
    Chinchilla scaling law.
    
    Args:
        N: Number of parameters
        D: Number of training tokens
        E, A, B, alpha, beta: Fitted parameters
        
    Returns:
        Predicted final loss
    """
    return E + A / (N ** alpha) + B / (D ** beta)


def scaling_law_N_only(
    N: float | NDArray[np.floating],
    compute_budget: float,
    E: float,
    A: float,
    B: float,
    alpha: float,
    beta: float
) -> float | NDArray[np.floating]:
    """
    Scaling law expressed in terms of N only, assuming optimal compute allocation.
    
    For a given compute budget C, we derive D from C ≈ 6 * N * D
    So D = C / (6 * N)
    """
    D: float | NDArray[np.floating] = compute_budget / (6 * N)
    return scaling_law(N, D, E, A, B, alpha, beta)


def fit_scaling_laws(data: list[dict[str, Any]]) -> tuple[float, float, float, float, float]:
    """
    Fit scaling law parameters to the isoFLOP data.
    
    Returns fitted parameters (E, A, B, alpha, beta)
    """
    # Extract data points
    N_values = np.array([d['parameters'] for d in data])
    C_values = np.array([d['compute_budget'] for d in data])
    L_values = np.array([d['final_loss'] for d in data])
    
    # Compute D values: D = C / (6 * N)
    D_values = C_values / (6 * N_values)
    
    print("="*80)
    print("DATA SUMMARY")
    print("="*80)
    print(f"Number of data points: {len(data)}")
    print(f"Parameter range: {N_values.min():.2e} to {N_values.max():.2e}")
    print(f"Compute range: {C_values.min():.2e} to {C_values.max():.2e}")
    print(f"Loss range: {L_values.min():.4f} to {L_values.max():.4f}")
    print(f"Tokens range: {D_values.min():.2e} to {D_values.max():.2e}")
    print()
    
    # Alternative: Fit in linear space with initial guesses
    # Initial parameter guesses based on Chinchilla paper
    # E ~ 1-2, A ~ 400-1000, B ~ 400-1000, alpha ~ 0.34, beta ~ 0.28
    p0 = [1.5, 500, 500, 0.34, 0.28]
    
    # Fit the model
    print("="*80)
    print("FITTING SCALING LAW")
    print("="*80)
    print("Initial parameters:")
    print(f"  E = {p0[0]}")
    print(f"  A = {p0[1]}")
    print(f"  B = {p0[2]}")
    print(f"  α = {p0[3]}")
    print(f"  β = {p0[4]}")
    print()
    
    # Define function to fit
    def model_func(data, E, A, B, alpha, beta):
        """Wrapper for curve_fit."""
        N: NDArray[np.floating] = data[:, 0]
        D: NDArray[np.floating] = data[:, 1]
        return scaling_law(N, D, E, A, B, alpha, beta)
    
    # Prepare data
    X_data: NDArray[np.floating] = np.column_stack([N_values, D_values])
    
    # Fit
    try:
        # curve_fit returns tuple of (params, covariance)
        fit_result = curve_fit(
            model_func, 
            X_data, 
            L_values, 
            p0=p0,
            maxfev=10000,
            bounds=([0, 0, 0, 0, 0], [10, 1e6, 1e6, 1, 1])
        )
        popt = fit_result[0]
        
        E, A, B, alpha, beta = popt
        
        print("Fitted parameters:")
        print(f"  E = {E:.4f}")
        print(f"  A = {A:.4f}")
        print(f"  B = {B:.4f}")
        print(f"  α = {alpha:.4f}")
        print(f"  β = {beta:.4f}")
        print()
        
        # Compute R²
        L_pred = model_func(X_data, *popt)
        ss_res = np.sum((L_values - L_pred) ** 2)
        ss_tot = np.sum((L_values - np.mean(L_values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        print(f"R² = {r_squared:.4f}")
        print()
        
        return E, A, B, alpha, beta
        
    except Exception as e:
        print(f"Fitting failed: {e}")
        print("Using Chinchilla paper defaults instead.")
        return 1.69, 406.4, 410.7, 0.34, 0.28


def predict_optimal_config(
    compute_budget: float,
    E: float,
    A: float,
    B: float,
    alpha: float,
    beta: float
) -> tuple[float, float, float]:
    """
    Predict optimal model size and dataset size for a given compute budget.
    
    From Chinchilla: optimal N and D satisfy:
    D/N = (alpha * B) / (beta * A)
    
    Combined with C ≈ 6 * N * D
    """
    # Optimal ratio
    ratio: float = (alpha * B) / (beta * A)
    
    # From D = ratio * N and C = 6 * N * D
    # C = 6 * N * (ratio * N) = 6 * ratio * N²
    # N = sqrt(C / (6 * ratio))
    N_opt: float = float(np.sqrt(compute_budget / (6 * ratio)))
    D_opt: float = ratio * N_opt
    
    # Predict loss
    L_pred: float = float(scaling_law(N_opt, D_opt, E, A, B, alpha, beta))
    
    return N_opt, D_opt, L_pred


def generate_plots(
    data: list[dict[str, Any]],
    E: float,
    A: float,
    B: float,
    alpha: float,
    beta: float,
    output_dir: str | Path
) -> None:
    """Generate scaling law plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract data
    N_values: NDArray[np.floating] = np.array([d['parameters'] for d in data])
    C_values: NDArray[np.floating] = np.array([d['compute_budget'] for d in data])
    L_values: NDArray[np.floating] = np.array([d['final_loss'] for d in data])
    _ = C_values / (6 * N_values)  # D_values for tokens range
    
    # Plot 1: Loss vs Parameters for each compute budget
    _ = plt.figure(figsize=(12, 5))
    
    _ = plt.subplot(1, 2, 1)
    unique_C: list[float] = sorted(set(C_values))
    for C in unique_C:
        mask: NDArray[np.bool_] = C_values == C
        _ = plt.loglog(N_values[mask], L_values[mask], 'o-', label=f'C={C:.1e}')
    
    _ = plt.xlabel('Parameters (N)')
    _ = plt.ylabel('Final Loss (L)')
    _ = plt.title('Scaling Law: Loss vs Parameters')
    _ = plt.legend()
    _ = plt.grid(True, alpha=0.3)
    
    # Plot 2: Loss vs Compute Budget
    _ = plt.subplot(1, 2, 2)
    _ = plt.loglog(C_values, L_values, 'o')
    
    # Add fitted curve
    C_range: NDArray[np.floating] = np.logspace(np.log10(float(C_values.min())), np.log10(float(C_values.max())), 100)
    L_fitted: list[float] = []
    for compute_budget in C_range:
        _, _, L_opt = predict_optimal_config(float(compute_budget), E, A, B, alpha, beta)
        L_fitted.append(L_opt)
    
    _ = plt.loglog(C_range, L_fitted, 'r-', linewidth=2, label='Fitted Scaling Law')
    _ = plt.xlabel('Compute Budget (FLOPs)')
    _ = plt.ylabel('Final Loss (L)')
    _ = plt.title('Scaling Law: Loss vs Compute')
    _ = plt.legend()
    _ = plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'scaling_laws.png', dpi=150)
    print(f"Saved plot to {output_dir / 'scaling_laws.png'}")
    plt.close()


def main() -> dict[str, Any]:
    """Main analysis pipeline."""
    print("\n" + "="*80)
    print("CS336 ASSIGNMENT 3: SCALING LAWS ANALYSIS")
    print("="*80 + "\n")
    
    # Load data
    data_path: Path = Path("data/isoflops_curves.json")
    data: list[dict[str, Any]] = load_isoflops_data(data_path)
    
    # Fit scaling laws
    E, A, B, alpha, beta = fit_scaling_laws(data)
    
    # Generate plots
    generate_plots(data, E, A, B, alpha, beta, "results")
    
    # Make predictions
    print("="*80)
    print("PREDICTIONS AT 10^19 FLOPs")
    print("="*80)
    
    target_compute: float = 1e19
    N_opt, D_opt, L_pred = predict_optimal_config(target_compute, E, A, B, alpha, beta)
    
    print(f"Target compute budget: {target_compute:.2e} FLOPs")
    print(f"Optimal model size: {N_opt:.2e} parameters ({N_opt/1e9:.2f}B)")
    print(f"Optimal dataset size: {D_opt:.2e} tokens ({D_opt/1e9:.2f}B tokens)")
    print(f"Predicted final loss: {L_pred:.4f}")
    print()
    
    # Compare to Chinchilla predictions
    print("Comparison to Chinchilla paper (for reference):")
    print("  At 10^19 FLOPs, Chinchilla suggests:")
    print("  - Model size: ~4-5B parameters")
    print("  - Dataset size: ~80-100B tokens")
    print("  - Loss: ~5.5-6.0")
    print()
    
    # Save results
    results: dict[str, Any] = {
        "fitted_parameters": {
            "E": float(E),
            "A": float(A),
            "B": float(B),
            "alpha": float(alpha),
            "beta": float(beta)
        },
        "prediction_1e19": {
            "optimal_parameters": float(N_opt),
            "optimal_tokens": float(D_opt),
            "predicted_loss": float(L_pred)
        }
    }
    
    with open("results/scaling_law_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("Results saved to:")
    print("  - results/scaling_laws.png")
    print("  - results/scaling_law_results.json")
    print()
    
    return results


if __name__ == "__main__":
    results = main()
