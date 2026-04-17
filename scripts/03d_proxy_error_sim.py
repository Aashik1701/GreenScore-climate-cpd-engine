"""
GreenScore — Transition Proxy Monte Carlo Error Bounds
======================================================
Simulates the transition risk factor variance if the heuristic 'loan_purpose -> sector'
proxy is wrong X% of the time.
"""
import numpy as np

def simulate_proxy_error(baseline_pds, transition_factors, error_rates=[0.1, 0.2, 0.3], trials=100):
    print("====================================")
    print("MONTE CARLO PROXY ERROR SIMULATION")
    print("====================================")
    
    n = len(baseline_pds)
    all_factors = np.unique(transition_factors)
    
    for err in error_rates:
        mean_deviations = []
        for t in range(trials):
            # Randomly swap X% of transition factors
            mask = np.random.rand(n) < err
            corrupted_factors = transition_factors.copy()
            corrupted_factors[mask] = np.random.choice(all_factors, size=mask.sum())
            
            # Compute new average portfolio transition PD
            orig_cpd = baseline_pds * (1 + transition_factors)
            new_cpd = baseline_pds * (1 + corrupted_factors)
            
            # Record absolute deviation
            mean_deviations.append(np.abs(orig_cpd - new_cpd).mean())
            
        print(f"Error Rate: {err*100:.0f}% -> Mean CPD Error Bound: ±{np.mean(mean_deviations):.5f} (Std: ±{np.std(mean_deviations):.5f})")

if __name__ == '__main__':
    # Simulated approximation matching the real GreenScore distribution:
    # Baseline PD ~ 12% across 2 million records
    baseline_pds = np.random.beta(2, 10, 10000)
    # Transition factors (Orderly scenario ~ 9% premium)
    transition_factors = np.random.normal(0.096, 0.02, 10000)
    simulate_proxy_error(baseline_pds, transition_factors)
