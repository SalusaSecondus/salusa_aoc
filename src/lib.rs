use std::{
    borrow::Borrow,
    collections::{BinaryHeap, HashMap},
    fmt::Display,
    hash::Hash,
};

use min_max_heap::MinMaxHeap;
use num_traits::{One, Zero};
pub mod bitset;

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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EdgeWeight<T, W>
where
    T: Clone + Eq + Hash,
    W: Copy + Eq + Hash + Ord + One + Zero,
{
    dest: T,
    weight: W,
}

impl<T, W> PartialOrd for EdgeWeight<T, W>
where
    T: Clone + Eq + Hash,
    W: Copy + Eq + Hash + Ord + One + Zero,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T, W> Ord for EdgeWeight<T, W>
where
    T: Clone + Eq + Hash,
    W: Copy + Eq + Hash + Ord + One + Zero,
{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.weight.cmp(&self.weight)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct WeightedPath<T, W> where
T: Clone + Eq + Hash,
W: Copy + Eq + Hash + Ord {
    dest: T,
    weight: W,
    path: Vec<T>
}

impl <T, W> Ord for WeightedPath<T, W>  where
T: Clone + Eq + Hash,
W: Copy + Eq + Hash + Ord {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.weight.cmp(&self.weight)
    }
}

impl<T, W> PartialOrd for WeightedPath<T, W>  where
T: Clone + Eq + Hash,
W: Copy + Eq + Hash + Ord {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug, Clone)]
pub struct Graph<T, W>
where
    T: Clone + Eq + Hash,
    W: Copy + Eq + Hash + Ord + One + Zero,
{
    digraph: bool,
    map: HashMap<T, Vec<EdgeWeight<T, W>>>,
}

impl<T, W> Graph<T, W>
where
    T: Clone + Eq + Hash,
    W: Copy + Eq + Ord + Hash + One + Zero,
{
    pub fn new(digraph: bool) -> Self {
        Self {
            digraph,
            map: HashMap::new(),
        }
    }

    pub fn add_edge(&mut self, from: T, to: T) {
        if self.digraph {
            self.map.entry(to.clone()).or_default().push(EdgeWeight {
                dest: from.clone(),
                weight: W::one(),
            });
        }
        self.map.entry(from).or_default().push(EdgeWeight {
            dest: to,
            weight: W::one(),
        });
    }

    pub fn add_weighted_edge(&mut self, from: T, to: T, weight: W) {
        if self.digraph {
            self.map.entry(to.clone()).or_default().push(EdgeWeight {
                dest: from.clone(),
                weight,
            });
        }
        self.map
            .entry(from)
            .or_default()
            .push(EdgeWeight { dest: to, weight });
    }

    pub fn nodes(&self) -> std::collections::hash_map::Keys<T, Vec<EdgeWeight<T, W>>> {
        self.map.keys()
    }

    pub fn edges<Q: ?Sized>(&self, node: &Q) -> Vec<T>
    where
        T: Borrow<Q>,
        Q: Hash + Eq,
    {
        if let Some(edges) = self.map.get(node) {
            edges.iter().map(|e| e.dest.to_owned()).collect::<Vec<T>>()
        } else {
            vec![]
        }
    }

    pub fn distance_map(&self, start: &T) -> HashMap<T, W> {
        let mut result = HashMap::new();
        let mut queue: BinaryHeap<EdgeWeight<T, W>> = BinaryHeap::new();
        queue.push(EdgeWeight {
            dest: start.clone(),
            weight: W::zero(),
        });

        while let Some(current_node) = queue.pop() {
            // let current_node = current_node.dest;
            if result.contains_key(&current_node.dest) {
                continue;
            }
            result.insert(current_node.dest.clone(), current_node.weight);
            if let Some(edges) = self.map.get(&current_node.dest) {
                for e in edges {
                    if result.contains_key(&e.dest) {
                        continue;
                    }
                    queue.push(EdgeWeight {
                        dest: e.dest.clone(),
                        weight: e.weight + current_node.weight,
                    });
                }
            }
        }

        result
    }

    pub fn path_map(&self, start: &T) -> HashMap<T, (Vec<T>, W)> {
        let mut result = HashMap::new();



        let mut queue: BinaryHeap<WeightedPath<T, W>> = BinaryHeap::new();
        queue.push(WeightedPath {
            dest: start.clone(),
            weight: W::zero(),
            path: vec![]
        });

        while let Some(current_node) = queue.pop() {
            // let current_node = current_node.dest;
            if result.contains_key(&current_node.dest) {
                continue;
            }
            result.insert(current_node.dest.clone(), (current_node.path.clone(), current_node.weight));
            if let Some(edges) = self.map.get(&current_node.dest) {
                for e in edges {
                    if result.contains_key(&e.dest) {
                        continue;
                    }
                    let mut path = current_node.path.clone();
                    path.push(e.dest.clone());
                    queue.push(WeightedPath {
                        dest: e.dest.clone(),
                        weight: e.weight + current_node.weight,
                        path
                    });
                }
            }
        }

        result
    }
}

impl<T, W> MatrixTranspose for Graph<T, W>
where
    T: Clone + Eq + Hash,
    W: Copy + Eq + Ord + Hash + One + Zero,
{
    fn transpose(&self) -> Self {
        if self.digraph {
            self.clone()
        } else {
            let mut result = Graph::new(self.digraph);
            for (node, edges) in &self.map {
                for e in edges {
                    result.add_weighted_edge(e.dest.clone(), node.clone(), e.weight);
                }
            }
            result
        }
    }
}

impl<T, W> Graph<T, W>
where
    T: Ord + Clone + Eq + Hash,
    W: Copy + Eq + Ord + Hash + One + Zero,
{
    pub fn sort(&mut self) {
        self.map.values_mut().for_each(|v| v.sort_unstable());
    }
}

impl<T, W> Display for Graph<T, W>
where
    T: Clone + Eq + Hash + Display,
    W: Copy + Eq + Ord + Hash + One + Display + Zero,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for e in &self.map {
            write!(f, "{} ->", e.0)?;
            for v in e.1.iter().enumerate() {
                if v.0 == 0 {
                    write!(f, " {}({})", v.1.dest, v.1.weight)?;
                } else {
                    write!(f, ", {}({})", v.1.dest, v.1.weight)?;
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
    #[allow(clippy::type_complexity)]
    rule: Box<dyn Fn(&T) -> Option<Vec<(T, u64)>>>,
}

impl<T> ReplacementEngine<T>
where
    T: Hash + Eq + Clone + 'static,
{
    pub fn from_fn<F>(elements: HashMap<T, u64>, rule: F) -> Self
    where
        F: Fn(&T) -> Option<Vec<(T, u64)>> + 'static,
    {
        #[allow(clippy::type_complexity)]
        let rule: Box<dyn Fn(&T) -> Option<Vec<(T, u64)>>> = Box::new(rule);

        Self { elements, rule }
    }

    pub fn from_hash(elements: HashMap<T, u64>, rules: HashMap<T, Vec<(T, u64)>>) -> Self {
        #[allow(clippy::type_complexity)]
        let rule: Box<dyn Fn(&T) -> Option<Vec<(T, u64)>>> =
            Box::new(move |src| rules.get(src).map(|o| o.to_owned()));

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

pub trait SalusaAocIter: Iterator {
    fn max_n(self, size: usize) -> MinMaxHeap<Self::Item>;
    fn min_n(self, size: usize) -> MinMaxHeap<Self::Item>;
}

impl<T: Iterator> SalusaAocIter for T
where
    T::Item: Ord + PartialOrd + Copy,
{
    fn max_n(self, size: usize) -> MinMaxHeap<Self::Item> {
        let mut heap = MinMaxHeap::with_capacity(size + 1);

        for item in self {
            if heap.len() < size {
                heap.push(item);
            } else if let Some(curr_min) = heap.peek_min() {
                if *curr_min < item {
                    heap.push(item);
                    if heap.len() > size {
                        heap.pop_min();
                    }
                }
            }
        }
        heap
    }

    fn min_n(self, size: usize) -> MinMaxHeap<Self::Item> {
        let mut heap = MinMaxHeap::with_capacity(size + 1);

        for item in self {
            if heap.len() < size {
                heap.push(item);
            } else if let Some(curr_max) = heap.peek_max() {
                if *curr_max > item {
                    heap.push(item);
                    if heap.len() > size {
                        heap.pop_max();
                    }
                }
            }
        }
        heap
    }
}

#[cfg(test)]
mod tests {
    use crate::SalusaAocIter;

    #[test]
    fn test_min_max() {
        let samples = vec![97, 67, 18, 2, 42, 91, 75, 77, 87, 35, 81, 89, 39, 79, 5, 0];

        let mut sorted = samples.clone();
        sorted.sort_unstable();
        let sorted = sorted;

        for len in 0..=samples.len() {
            let min: Vec<i32> = samples.iter().min_n(len).drain_asc().copied().collect();
            let max: Vec<i32> = samples.iter().max_n(len).drain_asc().copied().collect();

            assert_eq!(min, sorted[0..len].to_vec());
            assert_eq!(max, sorted[samples.len() - len..samples.len()].to_vec());
        }
    }
}
