use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use std::f64::consts::PI;
use std::sync::{Arc, Mutex};
use std::thread;

/// 2D Rastrigin function
#[inline]
fn rastrigin_2d(x: f64, y: f64) -> f64 {
    let scale = 1.5;
    let x_scaled = x / scale;
    let y_scaled = y / scale;
    20.0 + x_scaled.powi(2) - 10.0 * (2.0 * PI * x_scaled).cos()
        + y_scaled.powi(2) - 10.0 * (2.0 * PI * y_scaled).cos()
}

/// Extract f64 value from Python object
fn extract_f64(py: Python, obj: &Bound<PyAny>) -> PyResult<f64> {
    obj.extract::<f64>()
}

/// Get starting point from numpy array
fn get_starting_point(
    py: Python,
    starting_points: &Bound<PyAny>,
    run_idx: usize,
) -> PyResult<(f64, f64)> {
    let row = starting_points.call_method1("__getitem__", (run_idx,))?;
    let x = extract_f64(py, &row.call_method1("__getitem__", (0,))?)?;
    let y = extract_f64(py, &row.call_method1("__getitem__", (1,))?)?;
    Ok((x, y))
}

/// Get random step from numpy array
fn get_random_step(
    py: Python,
    random_steps: &Bound<PyAny>,
    step_idx: usize,
    run_idx: usize,
) -> PyResult<(f64, f64)> {
    let step = random_steps
        .call_method1("__getitem__", ((step_idx, run_idx),))?;
    let dx = extract_f64(py, &step.call_method1("__getitem__", (0,))?)?;
    let dy = extract_f64(py, &step.call_method1("__getitem__", (1,))?)?;
    Ok((dx, dy))
}

/// Get acceptance probability from numpy array
fn get_acceptance_prob(
    py: Python,
    acceptance_probs: &Bound<PyAny>,
    step_idx: usize,
    run_idx: usize,
) -> PyResult<f64> {
    let prob = acceptance_probs
        .call_method1("__getitem__", ((step_idx, run_idx),))?;
    extract_f64(py, &prob)
}

/// Run single SA optimization using pre-computed random samples from Python
fn run_single_sa(
    py: Python,
    run_idx: usize,
    init_temp: f64,
    cooling_rate: f64,
    step_size: f64,
    num_steps: usize,
    bounds: (f64, f64),
    starting_points: &Bound<PyAny>,
    random_steps: &Bound<PyAny>,
    acceptance_probs: &Bound<PyAny>,
) -> PyResult<(f64, Vec<(f64, f64, f64)>)> {
    // Get starting position from pre-computed samples
    let (mut curr_x, mut curr_y) = get_starting_point(py, starting_points, run_idx)?;
    let mut curr_cost = rastrigin_2d(curr_x, curr_y);
    let mut best_cost = curr_cost;
    
    let mut curr_temp = init_temp;
    
    let mut trajectory = Vec::with_capacity(num_steps + 1);
    trajectory.push((curr_x, curr_y, curr_cost));
    
    for step_idx in 0..num_steps {
        // Use pre-computed random steps
        let (dx_raw, dy_raw) = get_random_step(py, random_steps, step_idx, run_idx)?;
        let dx = dx_raw * step_size;
        let dy = dy_raw * step_size;
        
        let cand_x = (curr_x + dx).clamp(bounds.0, bounds.1);
        let cand_y = (curr_y + dy).clamp(bounds.0, bounds.1);
        let cand_cost = rastrigin_2d(cand_x, cand_y);
        
        // Acceptance criterion
        let delta = cand_cost - curr_cost;
        let accepted = if delta < 0.0 {
            true
        } else {
            let prob = if curr_temp > 1e-9 {
                (-delta / curr_temp).exp()
            } else {
                0.0
            };
            // Use pre-computed acceptance probability
            let acceptance_prob = get_acceptance_prob(py, acceptance_probs, step_idx, run_idx)?;
            acceptance_prob < prob
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
    
    Ok((curr_cost, trajectory))
}

/// Run Simulated Annealing algorithm (serial version)
/// 
/// Args:
///     init_temp: Initial temperature
///     cooling_rate: Temperature decay rate per step
///     step_size: Standard deviation for random walk
///     num_steps: Total number of SA iterations
///     bounds: (min, max) bounds for search space
///     seed: Random seed (ignored - uses pre-computed samples from Python)
///     num_runs: Number of SA runs to average over
///     starting_points: Pre-computed starting points from Python (numpy array)
///     random_steps: Pre-computed random steps from Python (numpy array)
///     acceptance_probs: Pre-computed acceptance probabilities from Python (numpy array)
/// 
/// Returns:
///     (avg_reward, costs, trajectory, median_idx)
#[pyfunction]
#[pyo3(signature = (init_temp, cooling_rate, step_size, num_steps, bounds, _seed=None, num_runs=10, starting_points, random_steps, acceptance_probs))]
fn run_sa(
    py: Python,
    init_temp: f64,
    cooling_rate: f64,
    step_size: f64,
    num_steps: usize,
    bounds: (f64, f64),
    _seed: Option<u64>,
    num_runs: usize,
    starting_points: &Bound<PyAny>,
    random_steps: &Bound<PyAny>,
    acceptance_probs: &Bound<PyAny>,
) -> PyResult<(f64, Vec<f64>, Vec<(f64, f64, f64)>, usize)> {
    let mut total_reward = 0.0;
    let mut costs = Vec::with_capacity(num_runs);
    let mut trajectories: Vec<Vec<(f64, f64, f64)>> = Vec::with_capacity(num_runs);
    
    for run_idx in 0..num_runs {
        let (curr_cost, trajectory) = run_single_sa(
            py,
            run_idx,
            init_temp,
            cooling_rate,
            step_size,
            num_steps,
            bounds,
            starting_points,
            random_steps,
            acceptance_probs,
        )?;
        
        costs.push(curr_cost);
        trajectories.push(trajectory);
        total_reward += -curr_cost;
    }
    
    // Average reward (no penalty - penalty is handled in environment)
    let avg_reward = total_reward / num_runs as f64;
    
    // Find trajectory with median cost (representative)
    let mut sorted_indices: Vec<usize> = (0..costs.len()).collect();
    sorted_indices.sort_by(|&a, &b| costs[a].partial_cmp(&costs[b]).unwrap());
    let median_idx = sorted_indices[costs.len() / 2];
    let last_trajectory = trajectories[median_idx].clone();
    
    Ok((avg_reward, costs, last_trajectory, median_idx))
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
///     seed: Random seed (ignored - uses pre-computed samples from Python)
///     num_runs: Number of SA runs to average over
///     num_threads: Number of parallel threads (optional, defaults to CPU count)
///     starting_points: Pre-computed starting points from Python (numpy array)
///     random_steps: Pre-computed random steps from Python (numpy array)
///     acceptance_probs: Pre-computed acceptance probabilities from Python (numpy array)
/// 
/// Returns:
///     (avg_reward, costs, trajectory, median_idx)
#[pyfunction]
#[pyo3(signature = (init_temp, cooling_rate, step_size, num_steps, bounds, _seed=None, num_runs=10, num_threads=None, starting_points, random_steps, acceptance_probs))]
fn run_sa_parallel(
    py: Python,
    init_temp: f64,
    cooling_rate: f64,
    step_size: f64,
    num_steps: usize,
    bounds: (f64, f64),
    _seed: Option<u64>,
    num_runs: usize,
    num_threads: Option<usize>,
    starting_points: &Bound<PyAny>,
    random_steps: &Bound<PyAny>,
    acceptance_probs: &Bound<PyAny>,
) -> PyResult<(f64, Vec<f64>, Vec<(f64, f64, f64)>, usize)> {
    let num_threads = num_threads.unwrap_or_else(|| thread::available_parallelism().map(|n| n.get()).unwrap_or(4));
    
    // Divide work among threads
    let runs_per_thread = num_runs / num_threads;
    let remainder = num_runs % num_threads;
    
    // For parallel execution, we need to extract data from numpy arrays once
    // and share it across threads (avoiding Python GIL issues)
    let mut starting_pts = Vec::with_capacity(num_runs);
    let mut random_step_data = vec![vec![(0.0, 0.0); num_runs]; num_steps];
    let mut acceptance_prob_data = vec![vec![0.0; num_runs]; num_steps];
    
    // Extract starting points
    for run_idx in 0..num_runs {
        starting_pts.push(get_starting_point(py, starting_points, run_idx)?);
    }
    
    // Extract random steps
    for step_idx in 0..num_steps {
        for run_idx in 0..num_runs {
            random_step_data[step_idx][run_idx] = get_random_step(py, random_steps, step_idx, run_idx)?;
            acceptance_prob_data[step_idx][run_idx] = get_acceptance_prob(py, acceptance_probs, step_idx, run_idx)?;
        }
    }
    
    let starting_pts = Arc::new(starting_pts);
    let random_step_data = Arc::new(random_step_data);
    let acceptance_prob_data = Arc::new(acceptance_prob_data);
    
    let results = Arc::new(Mutex::new(Vec::new()));
    let mut handles = vec![];
    
    for thread_id in 0..num_threads {
        let results = Arc::clone(&results);
        let starting_pts = Arc::clone(&starting_pts);
        let random_step_data = Arc::clone(&random_step_data);
        let acceptance_prob_data = Arc::clone(&acceptance_prob_data);
        
        let runs_for_this_thread = runs_per_thread + if thread_id < remainder { 1 } else { 0 };
        let start_run_idx = thread_id * runs_per_thread + thread_id.min(remainder);
        
        let handle = thread::spawn(move || -> PyResult<()> {
            let mut local_costs = Vec::with_capacity(runs_for_this_thread);
            let mut local_trajectories = Vec::with_capacity(runs_for_this_thread);
            
            for i in 0..runs_for_this_thread {
                let run_idx = start_run_idx + i;
                
                // Get starting position
                let (mut curr_x, mut curr_y) = starting_pts[run_idx];
                let mut curr_cost = rastrigin_2d(curr_x, curr_y);
                let mut best_cost = curr_cost;
                
                let mut curr_temp = init_temp;
                let mut trajectory = Vec::with_capacity(num_steps + 1);
                trajectory.push((curr_x, curr_y, curr_cost));
                
                for step_idx in 0..num_steps {
                    // Use pre-computed random steps
                    let (dx_raw, dy_raw) = random_step_data[step_idx][run_idx];
                    let dx = dx_raw * step_size;
                    let dy = dy_raw * step_size;
                    
                    let cand_x = (curr_x + dx).clamp(bounds.0, bounds.1);
                    let cand_y = (curr_y + dy).clamp(bounds.0, bounds.1);
                    let cand_cost = rastrigin_2d(cand_x, cand_y);
                    
                    // Acceptance criterion
                    let delta = cand_cost - curr_cost;
                    let accepted = if delta < 0.0 {
                        true
                    } else {
                        let prob = if curr_temp > 1e-9 {
                            (-delta / curr_temp).exp()
                        } else {
                            0.0
                        };
                        // Use pre-computed acceptance probability
                        let acceptance_prob = acceptance_prob_data[step_idx][run_idx];
                        acceptance_prob < prob
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
                
                local_costs.push(curr_cost);
                local_trajectories.push(trajectory);
            }
            
            // Store results
            let mut results = results.lock().unwrap();
            results.push((local_costs, local_trajectories));
            Ok(())
        });
        
        handles.push(handle);
    }
    
    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap()?;
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
    
    // Average reward (no penalty - penalty is handled in environment)
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
