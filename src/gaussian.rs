use std::rand::{mod, Rng, task_rng, TaskRng};
use std::rand::distributions::{Sample, Normal,IndependentSample};
use super::{Distribution,ContinuousDistribution};
use std::num::Float;
use std::f64::consts::PI;

pub struct Gaussian<R> where R: Rng{
    σ: f64,
    μ: f64,
    distri: Normal,
    rng: R
}

impl Gaussian<TaskRng> {
    pub fn new(μ:f64, σ:f64) -> Gaussian<TaskRng> {
        Gaussian {
            μ: μ, 
            σ: σ,
            distri: Normal::new(μ,σ),
            rng: task_rng()
        }
    }
}

impl<R> Distribution<f64> for Gaussian<R> where R: Rng {
    fn draw(&mut self) -> f64 {
        self.distri.ind_sample(&mut self.rng)
    }
    fn mean(&self) -> f64 {
        self.μ
    }
    fn variance(&self) -> f64 {
        self.σ
    }
}

impl<R> ContinuousDistribution<f64> for Gaussian<R> where R: Rng {
    fn unnormalized_log_pdf(&self, x: f64) -> f64 {
        let d = (x - self.μ) / self.σ;
        -d * d / 2.0
    }
    fn log_normalizer(&self) -> f64 {
        (2.0 * PI).sqrt().ln() + self.σ.ln()
    }
    fn normalizer(&self) -> f64 {
        1.0 / (2.0 * PI).sqrt() / self.σ
    }
}

#[test]
fn test_gaussian_draw() {
    let mut gaussian = Gaussian::new(0.0,0.0);
    for _ in range(0,20i) {
        println!("{}", gaussian.draw());
    }
}
#[test]
fn test_gaussian() {
    let gaussian = Gaussian::new(3.0, 5.0);
    let pdf9 = gaussian.pdf(9f64);
    println!("{}", pdf9);
    assert!(pdf9 <= 0.03885);
    assert!(pdf9 >= 0.03883);
}

