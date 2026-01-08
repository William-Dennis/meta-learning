use pyo3::prelude::*;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use std::sync::{Arc, Mutex};
use std::thread;

mod math;
use math::{rastrigin_2d, quadratic_2d};



/// Run single SA optimization
fn run_single_sa(
    mut rng: impl RngCore,
    init_temp: f64,
    cooling_rate: f64,
    step_size: f64,
    num_steps: usize,
    bounds: (f64, f64),
) -> (f64, Vec<(f64, f64, f64)>) {
    // Random start
    let mut curr_x = rng.gen_range(bounds.0..=bounds.1);
    let mut curr_y = rng.gen_range(bounds.0..=bounds.1);
    let mut curr_cost = rastrigin_2d(curr_x, curr_y);
    let mut best_cost = curr_cost;
    
    let mut curr_temp = init_temp;
    
    let mut trajectory = Vec::with_capacity(num_steps + 1);
    trajectory.push((curr_x, curr_y, curr_cost));
    
    for _ in 0..num_steps {
        // Generate neighbor using normal distribution
        let dx: f64 = rng.sample(rand_distr::Normal::new(0.0, step_size).unwrap());
        let dy: f64 = rng.sample(rand_distr::Normal::new(0.0, step_size).unwrap());
        
        let cand_x = (curr_x + dx).clamp(bounds.0, bounds.1);
        let cand_y = (curr_y + dy).clamp(bounds.0, bounds.1);
        let cand_cost = rastrigin_2d(cand_x, cand_y);
        
        // Accept?
        let delta = cand_cost - curr_cost;
        let accepted = if delta < 0.0 {
            true
        } else {
            let prob = if curr_temp > 1e-9 {
                (-delta / curr_temp).exp()
            } else {
                0.0
            };
            rng.gen::<f64>() < prob
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

/// Run Simulated Annealing algorithm (parallel version)
/// 
/// Uses multiple threads to run SA iterations in parallel for better performance
/// with large num_runs values.
/// 
/// Args:
///     init_temp: Initial temperature
///     cooling_rate: Temperature decay rate per step
///     step_size: Standard deviation for random walk
///     num_steps: Total number of SA iterations
///     bounds: (min, max) bounds for search space
///     seed: Random seed (optional)
///     num_runs: Number of SA runs to average over
///     num_threads: Number of parallel threads (optional, defaults to CPU count)
/// 
/// Returns:
///     (avg_reward, costs, trajectory, median_idx)
#[pyfunction]
#[pyo3(signature = (init_temp, cooling_rate, step_size, num_steps, bounds, seed=None, num_runs=10, num_threads=None))]
fn run_sa_parallel(
    init_temp: f64,
    cooling_rate: f64,
    step_size: f64,
    num_steps: usize,
    bounds: (f64, f64),
    seed: Option<u64>,
    num_runs: usize,
    num_threads: Option<usize>,
) -> PyResult<(f64, Vec<f64>, Vec<(f64, f64, f64)>, usize)> {
    let num_threads = num_threads.unwrap_or_else(|| thread::available_parallelism().map(|n| n.get()).unwrap_or(4));
    
    // Divide work among threads
    let runs_per_thread = num_runs / num_threads;
    let remainder = num_runs % num_threads;
    
    let results = Arc::new(Mutex::new(Vec::new()));
    let mut handles = vec![];
    
    for thread_id in 0..num_threads {
        let results = Arc::clone(&results);
        let runs_for_this_thread = runs_per_thread + if thread_id < remainder { 1 } else { 0 };
        
        let handle = thread::spawn(move || {
            // Create thread-local RNG with unique seed
            let thread_seed = seed.map(|s| s.wrapping_add(thread_id as u64));
            let mut rng: Box<dyn RngCore> = match thread_seed {
                Some(s) => Box::new(ChaCha8Rng::seed_from_u64(s)),
                None => Box::new(thread_rng()),
            };
            
            let mut local_costs = Vec::with_capacity(runs_for_this_thread);
            let mut local_trajectories = Vec::with_capacity(runs_for_this_thread);
            
            for _ in 0..runs_for_this_thread {
                let (curr_cost, trajectory) = run_single_sa(
                    &mut *rng,
                    init_temp,
                    cooling_rate,
                    step_size,
                    num_steps,
                    bounds,
                );
                
                local_costs.push(curr_cost);
                local_trajectories.push(trajectory);
            }
            
            // Store results
            let mut results = results.lock().unwrap();
            results.push((local_costs, local_trajectories));
        });
        
        handles.push(handle);
    }
    
    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }
    
    // Combine results from all threads
    let results = results.lock().unwrap();
    let mut costs = Vec::with_capacity(num_runs);
    let mut trajectories = Vec::with_capacity(num_runs);
    
    for (local_costs, local_trajectories) in results.iter() {
        costs.extend_from_slice(local_costs);
        trajectories.extend_from_slice(local_trajectories);
    }
    
    let total_reward: f64 = costs.iter().map(|&c| -c).sum();
    
    // Average reward (no penalty or bonus)
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

/// Quadratic function exposed to Python for testing
#[pyfunction]
fn quadratic_2d_py(x: f64, y: f64) -> PyResult<f64> {
    Ok(quadratic_2d(x, y))
}

/// A Python module implemented in Rust.
#[pymodule]
fn sa_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_sa_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(rastrigin_2d_py, m)?)?;
    m.add_function(wrap_pyfunction!(quadratic_2d_py, m)?)?;
    Ok(())
}

