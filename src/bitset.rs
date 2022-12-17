use std::{
    fmt::Display,
    mem::size_of,
    ops::{BitAnd, BitOr, Not, Shl},
};

use num_traits::{One, Unsigned, Zero};

#[derive(Debug, Clone, Copy)]
pub struct BitSet<T> {
    val: T,
}

#[derive(Debug, Clone, Copy)]
pub struct Iter<T> {
    set: BitSet<T>,
    curr: u8,
}

impl<T> BitSet<T>
where
    T: Zero
        + One
        + Unsigned
        + Shl<u8, Output = T>
        + BitAnd<Output = T>
        + Ord
        + Copy
        + BitOr<Output = T>
        + Not<Output = T>,
{
    pub fn new() -> Self {
        Self { val: T::zero() }
    }

    pub fn contains(&self, value: &u8) -> bool {
        if *value > Self::max_val() {
            false
        } else {
            !(self.val & Self::mask(*value)).is_zero()
        }
    }

    pub fn insert(&mut self, value: u8) -> bool {
        let result = self.contains(&value);
        self.val = self.val | Self::mask(value);
        result
    }

    pub fn remove(&mut self, value: &u8) -> bool {
        let result = self.contains(&value);
        self.val = self.val & !Self::mask(*value);
        result
    }

    pub const fn max_val() -> u8 {
        (size_of::<T>() * 8) as u8 - 1
    }

    fn mask(value: u8) -> T {
        T::one() << value
    }

    pub fn iter(&self) -> Iter<T> {
        Iter::of(self)
    }
}

impl<T> Default for BitSet<T>
where
    T: Zero
        + One
        + Unsigned
        + Shl<u8, Output = T>
        + BitAnd<Output = T>
        + Ord
        + Copy
        + BitOr<Output = T>
        + Not<Output = T>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> IntoIterator for BitSet<T>
where
    T: Zero
        + One
        + Unsigned
        + Shl<u8, Output = T>
        + BitAnd<Output = T>
        + Ord
        + Copy
        + BitOr<Output = T>
        + Not<Output = T>,
{
    type Item = u8;

    type IntoIter = Iter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<T> IntoIterator for &BitSet<T>
where
    T: Zero
        + One
        + Unsigned
        + Shl<u8, Output = T>
        + BitAnd<Output = T>
        + Ord
        + Copy
        + BitOr<Output = T>
        + Not<Output = T>,
{
    type Item = u8;

    type IntoIter = Iter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl BitSet<u32> {
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn len(&self) -> usize {
        self.val.count_ones() as usize
    }
}

impl BitSet<u64> {
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn len(&self) -> usize {
        self.val.count_ones() as usize
    }
}

impl BitSet<u128> {
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn len(&self) -> usize {
        self.val.count_ones() as usize
    }
}

impl<T> Iter<T> {
    fn of(set: &BitSet<T>) -> Self
    where
        T: Zero
            + One
            + Unsigned
            + Shl<u8, Output = T>
            + BitAnd<Output = T>
            + Ord
            + Copy
            + BitOr<Output = T>
            + Not<Output = T>,
    {
        let mut curr = 0;
        while curr <= BitSet::<T>::max_val() && !set.contains(&curr) {
            curr += 1;
        }
        Self { set: *set, curr }
    }
}

impl<T> Iterator for Iter<T>
where
    T: Zero
        + One
        + Unsigned
        + Shl<u8, Output = T>
        + BitAnd<Output = T>
        + Ord
        + Copy
        + BitOr<Output = T>
        + Not<Output = T>,
{
    type Item = u8;

    fn next(&mut self) -> Option<Self::Item> {
        let max = BitSet::<T>::max_val();
        if self.curr > max {
            None
        } else {
            let result = Some(self.curr);
            self.curr += 1;
            while self.curr <= max && !self.set.contains(&self.curr) {
                self.curr += 1;
            }
            result
        }
    }
}

impl<T> Display for BitSet<T>
where
    T: Zero
        + One
        + Unsigned
        + Shl<u8, Output = T>
        + BitAnd<Output = T>
        + Ord
        + Copy
        + BitOr<Output = T>
        + Not<Output = T>
        + Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        let mut first = true;
        for v in self {
            if !first {
                write!(f, ", ")?;
            }
            write!(f, "{}", v)?;
            first = false;
        }
        write!(f, "]")
    }
}
