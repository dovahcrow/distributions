use std::rand::{mod, Rng, task_rng, TaskRng};
use std::rand::distributions::{Sample, Normal,IndependentSample};
use super::{Distribution,ContinuousDistribution,HasCdf};
use std::num::Float;
use std::f64::consts::PI;
use rgsl::error;

pub struct Gaussian {
    sigma: f64,
    mu: f64,
    distri: Normal,
}

impl Gaussian {
    pub fn new(mu: f64, sigma: f64) -> Gaussian {
        Gaussian {
            mu: mu, 
            sigma: sigma,
            distri: Normal::new(mu,sigma),
        }
    }
}
impl Sample<f64> for Gaussian {
    fn sample<R>(&mut self, rng: &mut R) -> f64 where R: Rng {
        self.distri.sample(rng)
    }
}
impl IndependentSample<f64> for Gaussian {
    fn ind_sample<R>(&self, rng: &mut R) -> f64 where R: Rng {
        self.distri.ind_sample(rng)
    }
}

impl Distribution<f64> for Gaussian {
    fn mean(&self) -> f64 {
        self.mu
    }
    fn variance(&self) -> f64 {
        self.sigma
    }
}

impl ContinuousDistribution<f64> for Gaussian {
    fn unnormalized_log_pdf(&self, x: f64) -> f64 {
        let d = (x - self.mu) / self.sigma;
        -d * d / 2.0
    }
    fn log_normalizer(&self) -> f64 {
        (2.0 * PI).sqrt().ln() + self.sigma.ln()
    }
    fn normalizer(&self) -> f64 {
        1.0 / (2.0 * PI).sqrt() / self.sigma
    }
}

impl HasCdf<f64> for Gaussian {
    fn probability(&self, a: f64, b: f64) -> f64 {
        self.cdf(b) - self.cdf(a)
    }
    fn cdf(&self, x: f64) -> f64 {
        0.5 * (1. + error::erf((x - self.mu) / (2.0.sqrt() * self.sigma)))
    }
}

#[test]
fn test_gaussian_draw() {
    let mut gaussian = Gaussian::new(0.0,0.0);
    for _ in range(0,20i) {
        println!("{}", gaussian.sample(&mut task_rng()));
    }
}
#[test]
fn test_gaussian_pdf() {
    let gaussian = Gaussian::new(3.0, 5.0);
    let pdf9 = gaussian.pdf(9f64);
    assert!(pdf9 <= 0.03885);
    assert!(pdf9 >= 0.03883);
}

#[test]
fn test_gaussian_cdf() {
    let gaussian = Gaussian::new(9.0, 4.0);
    let cdf = gaussian.cdf(2.0);
    assert!(cdf >= 0.040058f64);
    assert!(cdf <= 0.040060f64);
}
