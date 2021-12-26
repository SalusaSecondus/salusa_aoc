use std::{collections::HashMap, hash::Hash, borrow::Borrow, fmt::Display};

pub trait MatrixTranspose {
    fn transpose(&self) -> Self;
}

impl<T> MatrixTranspose for Vec<Vec<T>>
where
    T: Clone,
{
    #[allow(clippy::needless_range_loop)]
    fn transpose(&self) -> Self {
        let old_x = self.len();
        let old_y = self[0].len();
        let mut result = vec![vec![]; old_y];

        for y in 0..old_y {
            for x in 0..old_x {
                result[y].push(self[x][y].clone());
            }
        }
        result
    }
}

impl<K, V> MatrixTranspose for HashMap<(K, K), V>
where
    K: Copy + Eq + std::hash::Hash,
    V: Clone,
{
    fn transpose(&self) -> Self {
        let mut result = HashMap::new();
        for (k, v) in self {
            result.insert((k.1, k.0), v.clone());
        }
        result
    }
}

#[derive(Debug, Clone)]
pub struct Graph<T>
where
    T: Clone + Eq + Hash,
{
    digraph: bool,
    map: HashMap<T, Vec<T>>,
}

impl<T> Graph<T>
where
    T: Clone + Eq + Hash,
{
    pub fn new(digraph: bool) -> Self {
        Self {
            digraph,
            map: HashMap::new(),
        }
    }

    pub fn add_edge(&mut self, from: T, to: T) {
        if self.digraph {
            self.map.entry(to.clone()).or_default().push(from.clone());
        }
        self.map.entry(from).or_default().push(to);
    }

    pub fn nodes(&self) -> std::collections::hash_map::Keys<T, Vec<T>> {
        self.map.keys()
    }

    pub fn edges<Q: ?Sized>(&self, node: &Q) -> Option<&Vec<T>>
    where
        T: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.map.get(node)
    }
}

impl<T> MatrixTranspose for Graph<T>
where
    T: Clone + Eq + Hash,
{
    fn transpose(&self) -> Self {
        if self.digraph {
            self.clone()
        } else {
            let mut result = Graph::new(self.digraph);
            for (node, edges) in &self.map {
                for dest in edges {
                    result.add_edge(dest.clone(), node.clone());
                }
            }
            result
        }
    }
}

impl<T> Graph<T>
where
    T: Ord + Clone + Eq + Hash,
{
    pub fn sort(&mut self) {
        self.map.values_mut().for_each(|v| v.sort_unstable());
    }
}

impl<T> Display for Graph<T>
where
    T: Clone + Eq + Hash + Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for e in &self.map {
            write!(f, "{} ->", e.0)?;
            for v in e.1.iter().enumerate() {
                if v.0 == 0 {
                    write!(f, " {}", v.1)?;
                } else {
                    write!(f, ", {}", v.1)?;
                }
            }
            writeln!(f)?;
        }
        Ok(())
    }
}


pub struct ReplacementEngine<T>
where
    T: Hash + Eq + Clone + 'static,
{
    pub elements: HashMap<T, u64>,
    rule: Box<dyn Fn(&T) -> Option<Vec<(T, u64)>>>
}

impl<T> ReplacementEngine<T>
where
    T: Hash + Eq + Clone + 'static
{
    pub fn from_fn<F>(elements: HashMap<T, u64>, rule: F) -> Self 
    where
    F: Fn(&T) -> Option<Vec<(T, u64)>> + 'static
    {
        let rule: Box<dyn Fn(&T) -> Option<Vec<(T, u64)>>> = Box::new(rule);

        
        Self { elements, rule }
    }

    pub fn from_hash(elements: HashMap<T, u64>, rules: HashMap<T, Vec<(T, u64)>>) -> Self {
        let rule: Box<dyn Fn(&T) -> Option<Vec<(T, u64)>>> = Box::new(move |src| rules.get(src).map(|o| o.to_owned()));

        
        Self { elements, rule }
    }

    pub fn step(&mut self) {
        let mut result = HashMap::new();
        for (node, count) in &self.elements {
            if let Some(replacements) = (self.rule)(node) {
                for r in replacements {
                    *result.entry(r.0.to_owned()).or_default() += *count * r.1;
                }
            }
        }
        self.elements = result;
    }
}
