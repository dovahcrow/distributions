#![experimental]
#![feature(non_ascii_idents)]
#![allow(unused_imports)]
#![warn(dead_code)]
use std::num::Float;
pub mod gaussian;
pub trait Distribution<T> {
    fn draw(&mut self) -> T;
    fn get(&mut self) -> T {
        self.draw()
    }
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
    fn pdf(&self, x: T) -> f64 {
        self.log_pdf(x).exp()
    }

    fn log_pdf(&self, x: T) -> f64 {
        self.unnormalized_log_pdf(x) - self.log_normalizer()
    }

    fn unnormalized_log_pdf(&self, x: T) -> f64;
    fn log_normalizer(&self) -> f64;
    fn normalizer(&self) -> f64 {
        (-self.log_normalizer()).exp()
    }
}


#[test]
fn it_works() {

}
