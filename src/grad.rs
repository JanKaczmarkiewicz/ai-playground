use std::cell::RefCell;
use std::fmt;

use std::ops::{Add, Mul};
use std::rc::Rc;

#[derive(fmt::Debug)]
pub enum Source {
    Addition(Rc<RefCell<Value>>, Rc<RefCell<Value>>),
    Multiplication(Rc<RefCell<Value>>, Rc<RefCell<Value>>),
}

#[derive(fmt::Debug)]
pub struct Value {
    pub data: f64,
    pub source: Option<Source>,
    pub gradient: f64,
}

impl Value {
    pub fn new(data: f64) -> Self {
        Self {
            data,
            source: None,
            gradient: 1.0,
        }
    }

    fn from_source(data: f64, source: Source) -> Self {
        Self {
            data,
            source: Some(source),
            gradient: 1.0,
        }
    }

    pub fn backprop(&self) {
        if let Some(source) = &self.source {
            match source {
                Source::Addition(left, right) => {
                    left.borrow_mut().gradient = self.gradient * 1.0;
                    right.borrow_mut().gradient = self.gradient * 1.0;

                    left.borrow().backprop();
                    right.borrow().backprop();
                }
                Source::Multiplication(left, right) => {
                    left.borrow_mut().gradient = self.gradient * right.borrow().data;
                    right.borrow_mut().gradient = self.gradient * left.borrow().data;

                    left.borrow().backprop();
                    right.borrow().backprop();
                }
            }
        }
    }
}

impl Add<Rc<RefCell<Value>>> for Value {
    type Output = Value;

    fn add(self, rhs: Rc<RefCell<Value>>) -> Self::Output {
        let rhs_data = rhs.borrow().data;
        Value::from_source(
            self.data + rhs_data,
            Source::Addition(Rc::new(RefCell::new(self)), rhs),
        )
    }
}

impl Add<f64> for Value {
    type Output = Value;

    fn add(self, rhs: f64) -> Self::Output {
        Value::from_source(
            self.data + rhs,
            Source::Addition(
                Rc::new(RefCell::new(self)),
                Rc::new(RefCell::new(Value::new(rhs))),
            ),
        )
    }
}

impl Add<Value> for f64 {
    type Output = Value;

    fn add(self, rhs: Value) -> Self::Output {
        Value::from_source(
            self + rhs.data,
            Source::Addition(
                Rc::new(RefCell::new(Value::new(self))),
                Rc::new(RefCell::new(rhs)),
            ),
        )
    }
}

impl Mul<Rc<RefCell<Value>>> for Value {
    type Output = Value;

    fn mul(self, rhs: Rc<RefCell<Self>>) -> Self::Output {
        let rhs_data = rhs.borrow().data;
        Value::from_source(
            self.data + rhs_data,
            Source::Multiplication(Rc::new(RefCell::new(self)), rhs),
        )
    }
}

impl Mul<f64> for Value {
    type Output = Value;

    fn mul(self, rhs: f64) -> Self::Output {
        Value::from_source(
            self.data + rhs,
            Source::Multiplication(
                Rc::new(RefCell::new(self)),
                Rc::new(RefCell::new(Value::new(rhs))),
            ),
        )
    }
}

impl Mul<Value> for f64 {
    type Output = Value;

    fn mul(self, rhs: Value) -> Self::Output {
        Value::from_source(
            self + rhs.data,
            Source::Multiplication(
                Rc::new(RefCell::new(Value::new(self))),
                Rc::new(RefCell::new(rhs)),
            ),
        )
    }
}

use layout::adt::dag::NodeHandle;

pub fn simple_graph(value: Value) {
    use layout::backends::svg::SVGWriter;
    use layout::core::base::Orientation;
    use layout::core::geometry::Point;
    use layout::core::style::*;
    use layout::core::utils::save_to_file;
    use layout::std_shapes::shapes::*;
    use layout::topo::layout::VisualGraph;

    let mut vg = VisualGraph::new(Orientation::LeftToRight);

    fn traverse(parent: &Rc<RefCell<Value>>, parent_handle: NodeHandle, vg: &mut VisualGraph) {
        let val = parent.borrow();

        if let Some(ref source) = val.source {
            match source {
                Source::Addition(left, right) => {
                    let handle_sign = vg.add_node(Element::create(
                        ShapeKind::new_box("+"),
                        StyleAttr::simple(),
                        Orientation::LeftToRight,
                        Point::new(100., 100.),
                    ));
                    vg.add_edge(Arrow::default(), handle_sign, parent_handle);

                    let handle0 = vg.add_node(Element::create(
                        ShapeKind::new_box(&format!(
                            "val: {}, grad: {}",
                            left.borrow().data,
                            left.borrow().gradient
                        )),
                        StyleAttr::simple(),
                        Orientation::LeftToRight,
                        Point::new(100., 100.),
                    ));
                    vg.add_edge(Arrow::default(), handle0, handle_sign);

                    traverse(left, handle0, vg);

                    let handle0 = vg.add_node(Element::create(
                        ShapeKind::new_box(&format!(
                            "val: {}, grad: {}",
                            right.borrow().data,
                            right.borrow().gradient
                        )),
                        StyleAttr::simple(),
                        Orientation::LeftToRight,
                        Point::new(100., 100.),
                    ));
                    vg.add_edge(Arrow::default(), handle0, handle_sign);

                    traverse(right, handle0, vg);
                }
                Source::Multiplication(left, right) => {
                    let handle_sign = vg.add_node(Element::create(
                        ShapeKind::new_box("*"),
                        StyleAttr::simple(),
                        Orientation::LeftToRight,
                        Point::new(100., 100.),
                    ));
                    vg.add_edge(Arrow::default(), handle_sign, parent_handle);

                    let handle0 = vg.add_node(Element::create(
                        ShapeKind::new_box(&format!(
                            "val: {}, grad: {}",
                            left.borrow().data,
                            left.borrow().gradient
                        )),
                        StyleAttr::simple(),
                        Orientation::LeftToRight,
                        Point::new(100., 100.),
                    ));
                    vg.add_edge(Arrow::default(), handle0, handle_sign);

                    traverse(left, handle0, vg);

                    let handle0 = vg.add_node(Element::create(
                        ShapeKind::new_box(&format!(
                            "val: {}, grad: {}",
                            right.borrow().data,
                            right.borrow().gradient
                        )),
                        StyleAttr::simple(),
                        Orientation::LeftToRight,
                        Point::new(100., 100.),
                    ));
                    vg.add_edge(Arrow::default(), handle0, handle_sign);

                    traverse(right, handle0, vg);
                }
            }
        }
    }

    let value_label = format!("val: {}, grad: {}", value.data, value.gradient);

    traverse(
        &Rc::new(RefCell::new(value)),
        vg.add_node(Element::create(
            ShapeKind::new_box(&value_label),
            StyleAttr::simple(),
            Orientation::LeftToRight,
            Point::new(100., 100.),
        )),
        &mut vg,
    );

    // Render the nodes to some rendering backend.
    let mut svg = SVGWriter::new();
    vg.do_it(false, false, false, &mut svg);

    // Save the output.
    let _ = save_to_file("./graph.svg", &svg.finalize());
}
