#![experimental]
#![feature(non_ascii_idents)]
#![allow(unused_imports)]
#![warn(dead_code)]
extern crate rgsl;
use std::num::Float;

pub mod gaussian;
pub mod uniform;

pub trait Distribution<T> {
    fn mean(&self) -> T;
    fn variance(&self) -> T;
}

pub trait DiscreteDistribution<T>: Distribution<T> {

    fn probability_of(&self, x: T) -> f64;

    
    fn log_probability_of(&self, x: T) -> f64 {
        self.probability_of(x).ln()
    }
    fn unnormalized_probability_of(&self, x: T) -> f64 {
        self.probability_of(x)
    }
    fn unnormalized_log_probability_of(&self, x: T) -> f64 {
        self.unnormalized_probability_of(x).ln()
    }
}

pub trait ContinuousDistribution<T>: Distribution<T> {

    fn unnormalized_log_pdf(&self, x: T) -> f64;
    fn log_normalizer(&self) -> f64;

    
    fn pdf(&self, x: T) -> f64 {
        self.log_pdf(x).exp()
    }
    fn log_pdf(&self, x: T) -> f64 {
        self.unnormalized_log_pdf(x) - self.log_normalizer()
    }
    fn normalizer(&self) -> f64 {
        (-self.log_normalizer()).exp()
    }
}

pub trait HasCdf<T> where T: PartialOrd {
    fn probability(&self, a: T, b: T) -> f64;
    fn cdf(&self, a: T) -> f64;
}
