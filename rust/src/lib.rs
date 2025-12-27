use pyo3::prelude::*;
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
fn extract_f64(_py: Python, obj: &Bound<PyAny>) -> PyResult<f64> {
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
    let step = random_steps.call_method1("__getitem__", ((step_idx, run_idx),))?;
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
    let prob = acceptance_probs.call_method1("__getitem__", ((step_idx, run_idx),))?;
    extract_f64(py, &prob)
}

/// Execute a single SA step
fn sa_step(
    curr_x: f64,
    curr_y: f64,
    curr_cost: f64,
    curr_temp: f64,
    dx_raw: f64,
    dy_raw: f64,
    step_size: f64,
    bounds: (f64, f64),
    acceptance_prob: f64,
) -> (f64, f64, f64, bool) {
    let dx = dx_raw * step_size;
    let dy = dy_raw * step_size;
    
    let cand_x = (curr_x + dx).clamp(bounds.0, bounds.1);
    let cand_y = (curr_y + dy).clamp(bounds.0, bounds.1);
    let cand_cost = rastrigin_2d(cand_x, cand_y);
    
    // Acceptance criterion
    let delta = cand_cost - curr_cost;
    if delta < 0.0 {
        return (cand_x, cand_y, cand_cost, true);
    }
    
    let prob = if curr_temp > 1e-9 {
        (-delta / curr_temp).exp()
    } else {
        0.0
    };
    
    if acceptance_prob < prob {
        (cand_x, cand_y, cand_cost, true)
    } else {
        (curr_x, curr_y, curr_cost, false)
    }
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
    let (mut curr_x, mut curr_y) = get_starting_point(py, starting_points, run_idx)?;
    let mut curr_cost = rastrigin_2d(curr_x, curr_y);
    let mut curr_temp = init_temp;
    
    let mut trajectory = Vec::with_capacity(num_steps + 1);
    trajectory.push((curr_x, curr_y, curr_cost));
    
    for step_idx in 0..num_steps {
        let (dx_raw, dy_raw) = get_random_step(py, random_steps, step_idx, run_idx)?;
        let acceptance_prob = get_acceptance_prob(py, acceptance_probs, step_idx, run_idx)?;
        
        let (new_x, new_y, new_cost, _accepted) = sa_step(
            curr_x, curr_y, curr_cost, curr_temp,
            dx_raw, dy_raw, step_size, bounds, acceptance_prob
        );
        
        curr_x = new_x;
        curr_y = new_y;
        curr_cost = new_cost;
        trajectory.push((curr_x, curr_y, curr_cost));
        curr_temp *= cooling_rate;
    }
    
    Ok((curr_cost, trajectory))
}

/// Compute final SA result from costs and trajectories
fn compute_result(
    costs: Vec<f64>,
    trajectories: Vec<Vec<(f64, f64, f64)>>,
    num_runs: usize,
) -> (f64, Vec<f64>, Vec<(f64, f64, f64)>, usize) {
    let total_reward: f64 = costs.iter().map(|&c| -c).sum();
    let avg_reward = total_reward / num_runs as f64;
    
    let mut sorted_indices: Vec<usize> = (0..costs.len()).collect();
    sorted_indices.sort_by(|&a, &b| costs[a].partial_cmp(&costs[b]).unwrap());
    let median_idx = sorted_indices[costs.len() / 2];
    let median_trajectory = trajectories[median_idx].clone();
    
    (avg_reward, costs, median_trajectory, median_idx)
}

/// Run Simulated Annealing algorithm (serial version)
#[pyfunction]
#[pyo3(signature = (init_temp, cooling_rate, step_size, num_steps, bounds, starting_points, random_steps, acceptance_probs, seed=None, num_runs=10))]
#[allow(unused_variables)]
fn run_sa(
    py: Python,
    init_temp: f64,
    cooling_rate: f64,
    step_size: f64,
    num_steps: usize,
    bounds: (f64, f64),
    starting_points: &Bound<PyAny>,
    random_steps: &Bound<PyAny>,
    acceptance_probs: &Bound<PyAny>,
    seed: Option<u64>,
    num_runs: usize,
) -> PyResult<(f64, Vec<f64>, Vec<(f64, f64, f64)>, usize)> {
    let mut costs = Vec::with_capacity(num_runs);
    let mut trajectories = Vec::with_capacity(num_runs);
    
    for run_idx in 0..num_runs {
        let (cost, trajectory) = run_single_sa(
            py, run_idx, init_temp, cooling_rate, step_size, num_steps, bounds,
            starting_points, random_steps, acceptance_probs,
        )?;
        costs.push(cost);
        trajectories.push(trajectory);
    }
    
    Ok(compute_result(costs, trajectories, num_runs))
}

/// Helper to run single SA with extracted data (for parallel execution)
fn run_single_sa_with_data(
    run_idx: usize,
    init_temp: f64,
    cooling_rate: f64,
    step_size: f64,
    num_steps: usize,
    bounds: (f64, f64),
    starting_pts: &[(f64, f64)],
    random_step_data: &[Vec<(f64, f64)>],
    acceptance_prob_data: &[Vec<f64>],
) -> (f64, Vec<(f64, f64, f64)>) {
    let (mut curr_x, mut curr_y) = starting_pts[run_idx];
    let mut curr_cost = rastrigin_2d(curr_x, curr_y);
    let mut curr_temp = init_temp;
    
    let mut trajectory = Vec::with_capacity(num_steps + 1);
    trajectory.push((curr_x, curr_y, curr_cost));
    
    for step_idx in 0..num_steps {
        let (dx_raw, dy_raw) = random_step_data[step_idx][run_idx];
        let acceptance_prob = acceptance_prob_data[step_idx][run_idx];
        
        let (new_x, new_y, new_cost, _accepted) = sa_step(
            curr_x, curr_y, curr_cost, curr_temp,
            dx_raw, dy_raw, step_size, bounds, acceptance_prob
        );
        
        curr_x = new_x;
        curr_y = new_y;
        curr_cost = new_cost;
        trajectory.push((curr_x, curr_y, curr_cost));
        curr_temp *= cooling_rate;
    }
    
    (curr_cost, trajectory)
}

/// Extract numpy arrays to Rust data structures for parallel processing
fn extract_samples(
    py: Python,
    num_runs: usize,
    num_steps: usize,
    starting_points: &Bound<PyAny>,
    random_steps: &Bound<PyAny>,
    acceptance_probs: &Bound<PyAny>,
) -> PyResult<(Vec<(f64, f64)>, Vec<Vec<(f64, f64)>>, Vec<Vec<f64>>)> {
    let mut starting_pts = Vec::with_capacity(num_runs);
    let mut random_step_data = vec![vec![(0.0, 0.0); num_runs]; num_steps];
    let mut acceptance_prob_data = vec![vec![0.0; num_runs]; num_steps];
    
    for run_idx in 0..num_runs {
        starting_pts.push(get_starting_point(py, starting_points, run_idx)?);
    }
    
    for step_idx in 0..num_steps {
        for run_idx in 0..num_runs {
            random_step_data[step_idx][run_idx] = 
                get_random_step(py, random_steps, step_idx, run_idx)?;
            acceptance_prob_data[step_idx][run_idx] = 
                get_acceptance_prob(py, acceptance_probs, step_idx, run_idx)?;
        }
    }
    
    Ok((starting_pts, random_step_data, acceptance_prob_data))
}

/// Run worker thread for parallel SA
fn run_worker_thread(
    thread_id: usize,
    runs_per_thread: usize,
    remainder: usize,
    init_temp: f64,
    cooling_rate: f64,
    step_size: f64,
    num_steps: usize,
    bounds: (f64, f64),
    starting_pts: Arc<Vec<(f64, f64)>>,
    random_step_data: Arc<Vec<Vec<(f64, f64)>>>,
    acceptance_prob_data: Arc<Vec<Vec<f64>>>,
    results: Arc<Mutex<Vec<(Vec<f64>, Vec<Vec<(f64, f64, f64)>>)>>>,
) {
    let runs_for_thread = runs_per_thread + if thread_id < remainder { 1 } else { 0 };
    let start_run_idx = thread_id * runs_per_thread + thread_id.min(remainder);
    
    let mut local_costs = Vec::with_capacity(runs_for_thread);
    let mut local_trajectories = Vec::with_capacity(runs_for_thread);
    
    for i in 0..runs_for_thread {
        let run_idx = start_run_idx + i;
        let (cost, trajectory) = run_single_sa_with_data(
            run_idx, init_temp, cooling_rate, step_size, num_steps, bounds,
            &starting_pts, &random_step_data, &acceptance_prob_data
        );
        local_costs.push(cost);
        local_trajectories.push(trajectory);
    }
    
    let mut results = results.lock().unwrap();
    results.push((local_costs, local_trajectories));
}

/// Collect results from worker threads
fn collect_parallel_results(
    results: Arc<Mutex<Vec<(Vec<f64>, Vec<Vec<(f64, f64, f64)>>)>>>,
    num_runs: usize,
) -> (Vec<f64>, Vec<Vec<(f64, f64, f64)>>) {
    let results = results.lock().unwrap();
    let mut costs = Vec::with_capacity(num_runs);
    let mut trajectories = Vec::with_capacity(num_runs);
    
    for (local_costs, local_trajectories) in results.iter() {
        costs.extend_from_slice(local_costs);
        trajectories.extend_from_slice(local_trajectories);
    }
    
    (costs, trajectories)
}

/// Run Simulated Annealing algorithm (parallel version)
#[pyfunction]
#[pyo3(signature = (init_temp, cooling_rate, step_size, num_steps, bounds, starting_points, random_steps, acceptance_probs, seed=None, num_runs=10, num_threads=None))]
#[allow(unused_variables)]
fn run_sa_parallel(
    py: Python,
    init_temp: f64,
    cooling_rate: f64,
    step_size: f64,
    num_steps: usize,
    bounds: (f64, f64),
    starting_points: &Bound<PyAny>,
    random_steps: &Bound<PyAny>,
    acceptance_probs: &Bound<PyAny>,
    seed: Option<u64>,
    num_runs: usize,
    num_threads: Option<usize>,
) -> PyResult<(f64, Vec<f64>, Vec<(f64, f64, f64)>, usize)> {
    let num_threads = num_threads.unwrap_or_else(|| 
        thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
    );
    
    let (starting_pts, random_step_data, acceptance_prob_data) = 
        extract_samples(py, num_runs, num_steps, starting_points, random_steps, acceptance_probs)?;
    
    let starting_pts = Arc::new(starting_pts);
    let random_step_data = Arc::new(random_step_data);
    let acceptance_prob_data = Arc::new(acceptance_prob_data);
    
    let runs_per_thread = num_runs / num_threads;
    let remainder = num_runs % num_threads;
    let results = Arc::new(Mutex::new(Vec::new()));
    let mut handles = vec![];
    
    for thread_id in 0..num_threads {
        let results_clone = Arc::clone(&results);
        let starting_pts_clone = Arc::clone(&starting_pts);
        let random_step_clone = Arc::clone(&random_step_data);
        let acceptance_clone = Arc::clone(&acceptance_prob_data);
        
        let handle = thread::spawn(move || {
            run_worker_thread(
                thread_id, runs_per_thread, remainder,
                init_temp, cooling_rate, step_size, num_steps, bounds,
                starting_pts_clone, random_step_clone, acceptance_clone, results_clone
            );
        });
        
        handles.push(handle);
    }
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    let (costs, trajectories) = collect_parallel_results(results, num_runs);
    Ok(compute_result(costs, trajectories, num_runs))
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
