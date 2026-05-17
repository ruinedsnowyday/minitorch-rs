use crate::autodiff::ScalarGraph;

pub trait Learnable {
    fn data(&self) -> &[f64];
    fn data_mut(&mut self) -> &mut [f64];
}

impl Learnable for f64 {
    fn data(&self) -> &[f64] {
        std::slice::from_ref(self)
    }
    fn data_mut(&mut self) -> &mut [f64] {
        std::slice::from_mut(self)
    }
}

impl Learnable for Vec<f64> {
    fn data(&self) -> &[f64] {
        self
    }
    fn data_mut(&mut self) -> &mut [f64] {
        self
    }
}

pub struct Parameter {
    pub parameter: Box<dyn Learnable>,
}

impl Parameter {
    pub fn new(parameter: Box<dyn Learnable>) -> Self {
        Parameter { parameter }
    }
}

pub trait Module {
    fn training(&self) -> bool;

    fn children(&self) -> Vec<(&str, &dyn Module)>;
    fn children_mut(&mut self) -> Vec<(&str, &mut dyn Module)>;
    fn parameters(&self) -> Vec<(String, &f64)>;
    fn set_train(&mut self);
    fn set_eval(&mut self);

    fn named_parameters(&self) -> Vec<(String, &f64)> {
        let mut out: Vec<(String, &f64)> = self
            .parameters()
            .into_iter()
            .map(|(name, param)| (name.to_string(), param))
            .collect();
        for (child_name, child) in self.children() {
            for (named_parameter, param) in child.named_parameters() {
                out.push((format!("{child_name}.{named_parameter}"), param));
            }
        }
        out
    }

    fn train(&mut self) {
        self.set_train();
        for (_, child) in self.children_mut() {
            child.train();
        }
    }

    fn eval(&mut self) {
        self.set_eval();
        for (_, child) in self.children_mut() {
            child.eval();
        }
    }

    fn step(&mut self, graph: &ScalarGraph, lr: f64) {
        for (_, child_mut) in self.children_mut() {
            child_mut.step(graph, lr);
        }
    }
}
