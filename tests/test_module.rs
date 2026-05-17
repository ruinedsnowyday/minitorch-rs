use minitorch_rs::module::*;

struct SimpleModule {
    trainable: bool,
    a: f64,
    b: f64,
}

impl Module for SimpleModule {
    fn training(&self) -> bool {
        self.trainable
    }

    fn children(&self) -> Vec<(&str, &dyn Module)> {
        vec![]
    }

    fn children_mut(&mut self) -> Vec<(&str, &mut dyn Module)> {
        vec![]
    }

    fn parameters(&self) -> Vec<(String, &f64)> {
        vec![("a".to_string(), &self.a), ("b".to_string(), &self.b)]
    }

    fn set_train(&mut self) {
        self.trainable = true;
    }

    fn set_eval(&mut self) {
        self.trainable = false;
    }
}

struct SimpleNet {
    trainable: bool,
    simple1: SimpleModule,
    simple2: SimpleModule,
}

impl Module for SimpleNet {
    fn training(&self) -> bool {
        self.trainable
    }

    fn children(&self) -> Vec<(&str, &dyn Module)> {
        vec![("simple1", &self.simple1), ("simple2", &self.simple2)]
    }

    fn children_mut(&mut self) -> Vec<(&str, &mut dyn Module)> {
        vec![
            ("simple1", &mut self.simple1),
            ("simple2", &mut self.simple2),
        ]
    }

    fn parameters(&self) -> Vec<(String, &f64)> {
        vec![]
    }

    fn set_train(&mut self) {
        self.trainable = true
    }

    fn set_eval(&mut self) {
        self.trainable = false
    }
}

fn make_simple_module(a: f64, b: f64) -> SimpleModule {
    SimpleModule {
        trainable: true,
        a,
        b,
    }
}

fn make_simple_net() -> SimpleNet {
    SimpleNet {
        trainable: true,
        simple1: make_simple_module(1.0, 2.0),
        simple2: make_simple_module(3.0, 4.0),
    }
}

// ===================================================================
// Parameter data access
// ===================================================================

#[test]
fn test_parameter_data() {
    let p = Parameter::new(Box::new(5.0f64));
    assert_eq!(p.parameter.data(), &[5.0]);
}

#[test]
fn test_parameter_data_mut() {
    let mut p = Parameter::new(Box::new(5.0f64));
    p.parameter.data_mut()[0] = 10.0;
    assert_eq!(p.parameter.data(), &[10.0]);
}

// ===================================================================
// Simple module — parameters, no children
// ===================================================================

#[test]
fn test_simple_module_parameters() {
    let m = make_simple_module(1.0, 2.0);
    let params = m.parameters();
    assert_eq!(params.len(), 2);
    assert_eq!(params[0].0, "a");
    assert_eq!(params[1].0, "b");
    assert_eq!(*params[0].1, 1.0);
    assert_eq!(*params[1].1, 2.0);
}

#[test]
fn test_simple_module_named_parameters() {
    let m = make_simple_module(1.0, 2.0);
    let named = m.named_parameters();
    assert_eq!(named.len(), 2);
    assert_eq!(named[0].0, "a");
    assert_eq!(named[1].0, "b");
}

#[test]
fn test_simple_module_no_children() {
    let m = make_simple_module(1.0, 2.0);
    assert!(m.children().is_empty());
}

// ===================================================================
// Train / eval
// ===================================================================

#[test]
fn test_train_eval_simple() {
    let mut m = make_simple_module(1.0, 2.0);
    assert!(m.training());
    m.eval();
    assert!(!m.training());
    m.train();
    assert!(m.training());
}

#[test]
fn test_train_eval_recursive() {
    let mut net = make_simple_net();
    assert!(net.training());
    assert!(net.simple1.training());
    assert!(net.simple2.training());

    net.eval();
    assert!(!net.training());
    assert!(!net.simple1.training());
    assert!(!net.simple2.training());

    net.train();
    assert!(net.training());
    assert!(net.simple1.training());
    assert!(net.simple2.training());
}

// ===================================================================
// Nested named_parameters — dotted path names
// ===================================================================

#[test]
fn test_net_named_parameters() {
    let net = make_simple_net();
    let named = net.named_parameters();
    assert_eq!(named.len(), 4);

    let names: Vec<&str> = named.iter().map(|(n, _)| n.as_str()).collect();
    assert!(names.contains(&"simple1.a"));
    assert!(names.contains(&"simple1.b"));
    assert!(names.contains(&"simple2.a"));
    assert!(names.contains(&"simple2.b"));
}

#[test]
fn test_net_named_parameters_values() {
    let net = make_simple_net();
    let named = net.named_parameters();

    for (name, param) in named {
        let val = *param;
        match name.as_str() {
            "simple1.a" => assert_eq!(val, 1.0),
            "simple1.b" => assert_eq!(val, 2.0),
            "simple2.a" => assert_eq!(val, 3.0),
            "simple2.b" => assert_eq!(val, 4.0),
            other => panic!("unexpected parameter name: {other}"),
        }
    }
}

#[test]
fn test_net_no_own_parameters() {
    let net = make_simple_net();
    assert!(net.parameters().is_empty());
}
