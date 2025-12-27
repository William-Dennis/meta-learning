use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyList};
use rayon::prelude::*;
use std::f64::consts::PI;

/// 2D Rastrigin function
#[inline]
fn rastrigin_2d(x: f64, y: f64) -> f64 {
    let scale = 1.5;
    let x_scaled = x / scale;
    let y_scaled = y / scale;
    20.0 + x_scaled.powi(2) - 10.0 * (2.0 * PI * x_scaled).cos()
        + y_scaled.powi(2) - 10.0 * (2.0 * PI * y_scaled).cos()
}

/// Run single SA optimization using pre-generated random samples
/// NO RNG - uses provided starting point and random steps
fn run_single_sa_with_samples(
    start_x: f64,
    start_y: f64,
    random_steps: &[(f64, f64)],
    init_temp: f64,
    cooling_rate: f64,
    step_size: f64,
    num_steps: usize,
    bounds: (f64, f64),
) -> (f64, Vec<(f64, f64, f64)>) {
    let mut curr_x = start_x;
    let mut curr_y = start_y;
    let mut curr_cost = rastrigin_2d(curr_x, curr_y);
    let mut best_cost = curr_cost;
    
    let mut curr_temp = init_temp;
    
    let mut trajectory = Vec::with_capacity(num_steps + 1);
    trajectory.push((curr_x, curr_y, curr_cost));
    
    for step_idx in 0..num_steps {
        // Use pre-generated normal samples scaled by step_size (NO RNG)
        let (norm_dx, norm_dy) = random_steps[step_idx];
        let dx = norm_dx * step_size;
        let dy = norm_dy * step_size;
        
        let cand_x = (curr_x + dx).clamp(bounds.0, bounds.1);
        let cand_y = (curr_y + dy).clamp(bounds.0, bounds.1);
        let cand_cost = rastrigin_2d(cand_x, cand_y);
        
        // Accept?
        // NOTE: This uses DETERMINISTIC acceptance instead of probabilistic.
        // This is a FUNDAMENTAL algorithmic change from traditional SA:
        // - Traditional SA: accept with probability = exp(-delta/temp)
        // - This implementation: accept if exp(-delta/temp) > 0.5
        // This change ensures 100% reproducibility without RNG but may affect
        // optimization behavior. The threshold of 0.5 means we accept moves
        // that have >50% probability of acceptance in traditional SA.
        let delta = cand_cost - curr_cost;
        let accepted = if delta < 0.0 {
            true
        } else {
            let prob = if curr_temp > 1e-9 {
                (-delta / curr_temp).exp()
            } else {
                0.0
            };
            prob > 0.5  // Deterministic threshold (not probabilistic!)
        };
        
        if accepted {
            curr_x = cand_x;
            curr_y = cand_y;
            curr_cost = cand_cost;
            if curr_cost < best_cost {
                best_cost = curr_cost;
            }
        }
        
        trajectory.push((curr_x, curr_y, curr_cost));
        
        // Cool down
        curr_temp *= cooling_rate;
    }
    
    (curr_cost, trajectory)
}

/// Run Simulated Annealing algorithm (serial version)
/// 
/// Uses ONLY pre-generated random samples from Python UnifiedRandomSampler.
/// NO random number generation occurs in this function.
/// 
/// Args:
///     init_temp: Initial temperature
///     cooling_rate: Temperature decay rate per step
///     step_size: Standard deviation for random walk
///     num_steps: Total number of SA iterations
///     bounds: (min, max) bounds for search space
///     seed: Ignored - uses run index for deterministic sampling
///     num_runs: Number of SA runs to average over
/// 
/// Returns:
///     (avg_reward, costs, trajectory, median_idx)
#[pyfunction]
#[pyo3(signature = (init_temp, cooling_rate, step_size, num_steps, bounds, seed=None, num_runs=10))]
fn run_sa(
    py: Python,
    init_temp: f64,
    cooling_rate: f64,
    step_size: f64,
    num_steps: usize,
    bounds: (f64, f64),
    seed: Option<u64>,
    num_runs: usize,
) -> PyResult<(f64, Vec<f64>, Vec<(f64, f64, f64)>, usize)> {
    // Import Python's UnifiedRandomSampler (NO Rust RNG)
    let random_sampling = py.import_bound("random_sampling")?;
    let sampler_class = random_sampling.getattr("UnifiedRandomSampler")?;
    let sampler = sampler_class.call0()?;
    
    let mut total_reward = 0.0;
    let mut costs = Vec::with_capacity(num_runs);
    let mut trajectories: Vec<Vec<(f64, f64, f64)>> = Vec::with_capacity(num_runs);
    
    for run_idx in 0..num_runs {
        // Get pre-generated starting point (NO RNG)
        let start_point = sampler.call_method1("get_starting_point", (run_idx,))?;
        
        // Try extracting as tuple first, then as numpy array
        let (start_x, start_y) = match start_point.extract::<(f64, f64)>() {
            Ok(tuple) => tuple,
            Err(_) => {
                // Must be numpy array, convert via tolist
                let start_array = start_point.call_method0("tolist")?;
                let start_list: Vec<f64> = start_array.extract()?;
                (start_list[0], start_list[1])
            }
        };
        
        // Get pre-generated random steps (NO RNG)
        let steps_array = sampler.call_method1("get_random_steps", (run_idx, num_steps))?;
        let steps_list_py = steps_array.call_method0("tolist")?;
        let steps_list: Vec<Vec<f64>> = steps_list_py.extract()?;
        
        // Convert to vec of tuples
        let random_steps: Vec<(f64, f64)> = steps_list.iter()
            .map(|v| (v[0], v[1]))
            .collect();
        
        // Run SA with pre-generated samples (NO RNG)
        let (curr_cost, trajectory) = run_single_sa_with_samples(
            start_x,
            start_y,
            &random_steps,
            init_temp,
            cooling_rate,
            step_size,
            num_steps,
            bounds,
        );
        
        costs.push(curr_cost);
        trajectories.push(trajectory);
        total_reward += -curr_cost;
    }
    
    // Average reward WITHOUT step penalty (penalty moved to env)
    let avg_reward = total_reward / num_runs as f64;
    
    // Find trajectory with median cost (representative)
    let mut sorted_indices: Vec<usize> = (0..costs.len()).collect();
    sorted_indices.sort_by(|&a, &b| costs[a].partial_cmp(&costs[b]).unwrap());
    let median_idx = sorted_indices[costs.len() / 2];
    let last_trajectory = trajectories[median_idx].clone();
    
    Ok((avg_reward, costs, last_trajectory, median_idx))
}

/// Run Simulated Annealing algorithm (TRUE Rust parallel version with Rayon)
/// 
/// This implementation uses Rayon for TRUE parallel execution across CPU cores.
/// All random samples are pre-loaded into Rust data structures to avoid Python GIL
/// contention during parallel execution.
/// 
/// Uses ONLY pre-generated random samples from Python UnifiedRandomSampler.
/// NO random number generation occurs in this function.
/// 
/// Args:
///     init_temp: Initial temperature
///     cooling_rate: Temperature decay rate per step
///     step_size: Standard deviation for random walk
///     num_steps: Total number of SA iterations
///     bounds: (min, max) bounds for search space
///     seed: Ignored - uses run index for deterministic sampling
///     num_runs: Number of SA runs to average over
///     num_threads: Optional number of threads (defaults to CPU count)
/// 
/// Returns:
///     (avg_reward, costs, trajectory, median_idx)
#[pyfunction]
#[pyo3(signature = (init_temp, cooling_rate, step_size, num_steps, bounds, seed=None, num_runs=10, num_threads=None))]
fn run_sa_parallel(
    py: Python,
    init_temp: f64,
    cooling_rate: f64,
    step_size: f64,
    num_steps: usize,
    bounds: (f64, f64),
    seed: Option<u64>,
    num_runs: usize,
    num_threads: Option<usize>,
) -> PyResult<(f64, Vec<f64>, Vec<(f64, f64, f64)>, usize)> {
    // Import Python's UnifiedRandomSampler (NO Rust RNG)
    let random_sampling = py.import_bound("random_sampling")?;
    let sampler_class = random_sampling.getattr("UnifiedRandomSampler")?;
    let sampler = sampler_class.call0()?;
    
    // PRE-LOAD ALL random samples into Rust data structures to enable true parallelism
    // This avoids Python GIL contention during parallel execution
    let mut all_starting_points = Vec::with_capacity(num_runs);
    let mut all_random_steps = Vec::with_capacity(num_runs);
    
    for run_idx in 0..num_runs {
        // Get pre-generated starting point (NO RNG)
        let start_point = sampler.call_method1("get_starting_point", (run_idx,))?;
        
        // Try extracting as tuple first, then as numpy array
        let start_tuple = match start_point.extract::<(f64, f64)>() {
            Ok(tuple) => tuple,
            Err(_) => {
                // Must be numpy array, convert via tolist
                let start_array = start_point.call_method0("tolist")?;
                let start_list: Vec<f64> = start_array.extract()?;
                (start_list[0], start_list[1])
            }
        };
        all_starting_points.push(start_tuple);
        
        // Get pre-generated random steps (NO RNG)
        let steps_array = sampler.call_method1("get_random_steps", (run_idx, num_steps))?;
        let steps_list_py = steps_array.call_method0("tolist")?;
        let steps_list: Vec<Vec<f64>> = steps_list_py.extract()?;
        
        // Convert to vec of tuples
        let random_steps: Vec<(f64, f64)> = steps_list.iter()
            .map(|v| (v[0], v[1]))
            .collect();
        all_random_steps.push(random_steps);
    }
    
    // Set thread pool size if specified
    if let Some(threads) = num_threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .ok(); // Ignore error if already initialized
    }
    
    // Release GIL and run TRUE parallel execution with Rayon
    let results: Vec<(f64, Vec<(f64, f64, f64)>)> = py.allow_threads(|| {
        (0..num_runs)
            .into_par_iter()
            .map(|run_idx| {
                let (start_x, start_y) = all_starting_points[run_idx];
                let random_steps = &all_random_steps[run_idx];
                
                // Run SA with pre-generated samples (NO RNG)
                run_single_sa_with_samples(
                    start_x,
                    start_y,
                    random_steps,
                    init_temp,
                    cooling_rate,
                    step_size,
                    num_steps,
                    bounds,
                )
            })
            .collect()
    });
    
    // Collect results
    let mut total_reward = 0.0;
    let mut costs = Vec::with_capacity(num_runs);
    let mut trajectories = Vec::with_capacity(num_runs);
    
    for (cost, trajectory) in results {
        costs.push(cost);
        trajectories.push(trajectory);
        total_reward += -cost;
    }
    
    // Average reward WITHOUT step penalty (penalty moved to env)
    let avg_reward = total_reward / num_runs as f64;
    
    // Find trajectory with median cost (representative)
    let mut sorted_indices: Vec<usize> = (0..costs.len()).collect();
    sorted_indices.sort_by(|&a, &b| costs[a].partial_cmp(&costs[b]).unwrap());
    let median_idx = sorted_indices[costs.len() / 2];
    let last_trajectory = trajectories[median_idx].clone();
    
    Ok((avg_reward, costs, last_trajectory, median_idx))
}

/// Rastrigin function exposed to Python for testing
#[pyfunction]
fn rastrigin_2d_py(x: f64, y: f64) -> PyResult<f64> {
    Ok(rastrigin_2d(x, y))
}

/// A Python module implemented in Rust.
#[pymodule]
fn sa_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_sa, m)?)?;
    m.add_function(wrap_pyfunction!(run_sa_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(rastrigin_2d_py, m)?)?;
    Ok(())
}
