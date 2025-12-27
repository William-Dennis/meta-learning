use pyo3::prelude::*;
use ndarray::Array2;
use ndarray_npy::ReadNpyExt;
use std::f64::consts::PI;
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;

// Global cache for random samples
static RANDOM_SAMPLES: OnceLock<RandomSamples> = OnceLock::new();

struct RandomSamples {
    starting_points: Array2<f64>,
    random_steps: Vec<Vec<(f64, f64)>>,  // [step_idx][run_idx] -> (dx, dy)
    acceptance_probs: Vec<Vec<f64>>,     // [step_idx][run_idx] -> prob
}

impl RandomSamples {
    fn load() -> PyResult<Self> {
        // Get the path to the data directory
        let data_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Cannot find parent directory"))?
            .join("src")
            .join("meta_learning")
            .join("data");
        
        // Load starting points: shape (NUM_STARTING_POINTS, 2)
        let starting_points_path = data_dir.join("starting_points.npy");
        let file = File::open(&starting_points_path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(
                format!("Cannot open starting_points.npy: {}. Please run: python src/meta_learning/random_sampling.py", e)
            ))?;
        let reader = BufReader::new(file);
        let starting_points: Array2<f64> = Array2::read_npy(reader)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Cannot read starting_points.npy: {}", e)
            ))?;
        
        // Load random steps: shape (NUM_STEP_ROWS, NUM_STEP_COLUMNS, 2)
        let random_steps_path = data_dir.join("random_steps.npy");
        let file = File::open(&random_steps_path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(
                format!("Cannot open random_steps.npy: {}. Please run: python src/meta_learning/random_sampling.py", e)
            ))?;
        let reader = BufReader::new(file);
        let random_steps_raw: ndarray::ArrayD<f64> = ndarray::ArrayD::read_npy(reader)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Cannot read random_steps.npy: {}", e)
            ))?;
        
        // Load acceptance probabilities: shape (NUM_STEP_ROWS, NUM_STEP_COLUMNS)
        let acceptance_probs_path = data_dir.join("acceptance_probs.npy");
        let file = File::open(&acceptance_probs_path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(
                format!("Cannot open acceptance_probs.npy: {}. Please run: python src/meta_learning/random_sampling.py", e)
            ))?;
        let reader = BufReader::new(file);
        let acceptance_probs_raw: Array2<f64> = Array2::read_npy(reader)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Cannot read acceptance_probs.npy: {}", e)
            ))?;
        
        // Convert random_steps to a more efficient structure: [step_idx][run_idx] -> (dx, dy)
        let shape = random_steps_raw.shape();
        let num_steps = shape[0];
        let num_runs = shape[1];
        
        let mut random_steps = vec![vec![(0.0, 0.0); num_runs]; num_steps];
        for step_idx in 0..num_steps {
            for run_idx in 0..num_runs {
                let dx = random_steps_raw[[step_idx, run_idx, 0]];
                let dy = random_steps_raw[[step_idx, run_idx, 1]];
                random_steps[step_idx][run_idx] = (dx, dy);
            }
        }
        
        // Convert acceptance_probs to efficient structure: [step_idx][run_idx] -> prob
        let mut acceptance_probs = vec![vec![0.0; num_runs]; num_steps];
        for step_idx in 0..num_steps {
            for run_idx in 0..num_runs {
                acceptance_probs[step_idx][run_idx] = acceptance_probs_raw[[step_idx, run_idx]];
            }
        }
        
        Ok(RandomSamples {
            starting_points,
            random_steps,
            acceptance_probs,
        })
    }
    
    fn get_starting_point(&self, run_idx: usize) -> PyResult<(f64, f64)> {
        if run_idx >= self.starting_points.nrows() {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                format!("Run index {} exceeds available starting points ({})", run_idx, self.starting_points.nrows())
            ));
        }
        Ok((self.starting_points[[run_idx, 0]], self.starting_points[[run_idx, 1]]))
    }
    
    fn get_random_step(&self, run_idx: usize, step_idx: usize) -> PyResult<(f64, f64)> {
        if step_idx >= self.random_steps.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                format!("Step index {} exceeds available steps ({})", step_idx, self.random_steps.len())
            ));
        }
        if run_idx >= self.random_steps[step_idx].len() {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                format!("Run index {} exceeds available columns ({})", run_idx, self.random_steps[step_idx].len())
            ));
        }
        Ok(self.random_steps[step_idx][run_idx])
    }
    
    fn get_acceptance_prob(&self, run_idx: usize, step_idx: usize) -> PyResult<f64> {
        if step_idx >= self.acceptance_probs.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                format!("Step index {} exceeds available steps ({})", step_idx, self.acceptance_probs.len())
            ));
        }
        if run_idx >= self.acceptance_probs[step_idx].len() {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                format!("Run index {} exceeds available columns ({})", run_idx, self.acceptance_probs[step_idx].len())
            ));
        }
        Ok(self.acceptance_probs[step_idx][run_idx])
    }
}

fn get_random_samples() -> PyResult<&'static RandomSamples> {
    RANDOM_SAMPLES.get_or_init(|| {
        RandomSamples::load().unwrap_or_else(|_| {
            panic!("Failed to load random samples");
        })
    });
    RANDOM_SAMPLES.get().ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Random samples not initialized")
    })
}

/// 2D Rastrigin function
#[inline]
fn rastrigin_2d(x: f64, y: f64) -> f64 {
    let scale = 1.5;
    let x_scaled = x / scale;
    let y_scaled = y / scale;
    20.0 + x_scaled.powi(2) - 10.0 * (2.0 * PI * x_scaled).cos()
        + y_scaled.powi(2) - 10.0 * (2.0 * PI * y_scaled).cos()
}

/// Run single SA optimization using pre-computed random samples
fn run_single_sa(
    run_idx: usize,
    init_temp: f64,
    cooling_rate: f64,
    step_size: f64,
    num_steps: usize,
    bounds: (f64, f64),
) -> PyResult<(f64, Vec<(f64, f64, f64)>)> {
    let samples = get_random_samples()?;
    
    // Get starting position from pre-computed samples
    let (mut curr_x, mut curr_y) = samples.get_starting_point(run_idx)?;
    let mut curr_cost = rastrigin_2d(curr_x, curr_y);
    let mut best_cost = curr_cost;
    
    let mut curr_temp = init_temp;
    
    let mut trajectory = Vec::with_capacity(num_steps + 1);
    trajectory.push((curr_x, curr_y, curr_cost));
    
    for step_idx in 0..num_steps {
        // Use pre-computed random steps
        let (dx_raw, dy_raw) = samples.get_random_step(run_idx, step_idx)?;
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
            let acceptance_prob = samples.get_acceptance_prob(run_idx, step_idx)?;
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
///     seed: Random seed (ignored - uses pre-computed samples)
///     num_runs: Number of SA runs to average over
/// 
/// Returns:
///     (avg_reward, costs, trajectory, median_idx)
#[pyfunction]
#[pyo3(signature = (init_temp, cooling_rate, step_size, num_steps, bounds, _seed=None, num_runs=10))]
fn run_sa(
    init_temp: f64,
    cooling_rate: f64,
    step_size: f64,
    num_steps: usize,
    bounds: (f64, f64),
    _seed: Option<u64>,
    num_runs: usize,
) -> PyResult<(f64, Vec<f64>, Vec<(f64, f64, f64)>, usize)> {
    let mut total_reward = 0.0;
    let mut costs = Vec::with_capacity(num_runs);
    let mut trajectories: Vec<Vec<(f64, f64, f64)>> = Vec::with_capacity(num_runs);
    
    for run_idx in 0..num_runs {
        let (curr_cost, trajectory) = run_single_sa(
            run_idx,
            init_temp,
            cooling_rate,
            step_size,
            num_steps,
            bounds,
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
///     seed: Random seed (ignored - uses pre-computed samples)
///     num_runs: Number of SA runs to average over
///     num_threads: Number of parallel threads (optional, defaults to CPU count)
/// 
/// Returns:
///     (avg_reward, costs, trajectory, median_idx)
#[pyfunction]
#[pyo3(signature = (init_temp, cooling_rate, step_size, num_steps, bounds, _seed=None, num_runs=10, num_threads=None))]
fn run_sa_parallel(
    init_temp: f64,
    cooling_rate: f64,
    step_size: f64,
    num_steps: usize,
    bounds: (f64, f64),
    _seed: Option<u64>,
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
        let start_run_idx = thread_id * runs_per_thread + thread_id.min(remainder);
        
        let handle = thread::spawn(move || -> PyResult<()> {
            let mut local_costs = Vec::with_capacity(runs_for_this_thread);
            let mut local_trajectories = Vec::with_capacity(runs_for_this_thread);
            
            for i in 0..runs_for_this_thread {
                let run_idx = start_run_idx + i;
                let (curr_cost, trajectory) = run_single_sa(
                    run_idx,
                    init_temp,
                    cooling_rate,
                    step_size,
                    num_steps,
                    bounds,
                )?;
                
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
