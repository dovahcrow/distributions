use std::rand::{mod, Rng, task_rng, TaskRng};
use std::rand::distributions::{Sample, Normal,IndependentSample};
use super::{Distribution,ContinuousDistribution,HasCdf};
use std::num::Float;
use std::f64::consts::PI;

pub struct Uniform {
    a: f64,
    b: f64,
    pub entropy: f64,
}

impl Uniform {
    pub fn new(a: f64, b: f64) -> Uniform {
        Uniform {
            a: a,
            b: b,
            entropy: (b - a).ln(),
        }
    }
}

impl Sample<f64> for Uniform {
    fn sample<R>(&mut self, rng: &mut R) -> f64 where R: Rng {
        self.ind_sample(rng)
    }
}

impl IndependentSample<f64> for Uniform {
    fn ind_sample<R>(&self, rng: &mut R) -> f64 where R: Rng {
        rng.gen_range(self.a, self.b)
    }
}

impl Distribution<f64> for Uniform {
    fn mean(&self) -> f64 {
        (self.a + self.b) / 2.0
    }
    fn variance(&self) -> f64 {
        (self.b - self.a).powi(2) / 12.0
    }
}

impl ContinuousDistribution<f64> for Uniform {
    fn unnormalized_log_pdf(&self, x: f64) -> f64 {
        if x <= self.b && x >= self.a {
            let f = 1. / (self.b - self.a);
            f.ln()
        } else {
            0f64
        }
    }
    fn log_normalizer(&self) -> f64 {
        self.entropy
    }
}

impl HasCdf<f64> for Uniform {
    fn probability(&self, a: f64, b: f64) -> f64 {
        (b - a) / (self.b - self.a)
    }
    fn cdf(&self, x: f64) -> f64 {
        match x {
            x if x <= self.a => 0f64,
            x if x >= self.b => 1f64,
            _ => (x - self.a) / (self.b - self.a)
        }
    }
}

#[test]
fn test_uniform_distr() {
    let mut d = Uniform::new(1.0, 3.);
    for _ in range(0u, 99u) {
        let rand = d.sample(&mut task_rng());
        assert!(rand <= 3.0 && rand >= 1.0);
    }
}
