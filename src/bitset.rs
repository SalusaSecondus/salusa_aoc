use std::{
    fmt::Display,
    mem::size_of,
    ops::{Add, BitAnd, BitOr, Mul, Not, Shl},
};

use num_traits::{One, Zero};

#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, PartialOrd)]
pub struct BitSet<T> {
    val: T,
}

#[derive(Debug, Clone, Copy)]
pub struct Iter<T> {
    set: BitSet<T>,
    curr: u8,
    done: bool,
}

impl<T> BitSet<T>
where
    T: Zero
        + One
        // + Unsigned
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
        let result = self.contains(value);
        self.val = self.val & !Self::mask(*value);
        result
    }

    pub const fn max_val() -> u8 {
        ((size_of::<T>() * 8) - 1) as u8
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
        // + Unsigned
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
        // + Unsigned
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
        // + Unsigned
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

impl BitSet<Bs256> {
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn len(&self) -> usize {
        (self.val.high.count_ones() + self.val.low.count_ones()) as usize
    }
}

impl<T> Iter<T> {
    fn of(set: &BitSet<T>) -> Self
    where
        T: Zero
            + One
            // + Unsigned
            + Shl<u8, Output = T>
            + BitAnd<Output = T>
            + Ord
            + Copy
            + BitOr<Output = T>
            + Not<Output = T>,
    {
        let mut curr = 0;
        while curr <= BitSet::<T>::max_val() && !set.contains(&curr) {
            if curr == BitSet::<T>::max_val() {
                return Self {
                    set: *set,
                    curr: 0,
                    done: true,
                };
            }
            curr += 1;
        }
        Self {
            set: *set,
            curr,
            done: false,
        }
    }
}

impl<T> Iterator for Iter<T>
where
    T: Zero
        + One
        // + Unsigned
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
        if self.curr > max || self.done {
            None
        } else {
            let result = Some(self.curr);
            self.curr += 1;
            while self.curr <= max && !self.set.contains(&self.curr) {
                if self.curr == max {
                    self.done = true;
                    break;
                } else {
                    self.curr += 1;
                }
            }
            result
        }
    }
}

impl<T> Display for BitSet<T>
where
    T: Zero
        + One
        // + Unsigned
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Bs256 {
    high: u128,
    low: u128,
}

impl One for Bs256 {
    fn one() -> Self {
        Bs256 { high: 0, low: 1 }
    }

    fn is_one(&self) -> bool {
        self.high.is_zero() && self.low.is_one()
    }
}

impl Zero for Bs256 {
    fn zero() -> Self {
        Bs256 { high: 0, low: 0 }
    }

    fn is_zero(&self) -> bool {
        self.high.is_zero() && self.low.is_zero()
    }
}

impl Mul for Bs256 {
    type Output = Bs256;

    fn mul(self, _: Self) -> Self::Output {
        todo!("Not implemented, not needed")
    }
}

impl Add for Bs256 {
    type Output = Bs256;

    fn add(self, _: Self) -> Self::Output {
        todo!("Not implemented, not needed")
    }
}

impl Shl<u8> for Bs256 {
    type Output = Bs256;

    fn shl(self, rhs: u8) -> Self::Output {
        assert!(self.is_one());
        if rhs >= 128 {
            Bs256 {
                high: 1 << (rhs - 128),
                low: 0,
            }
        } else {
            Bs256 {
                high: 0,
                low: 1 << rhs,
            }
        }
    }
}

impl BitAnd for Bs256 {
    type Output = Bs256;

    fn bitand(self, rhs: Self) -> Self::Output {
        Bs256 {
            high: self.high & rhs.high,
            low: self.low & rhs.low,
        }
    }
}

impl BitOr for Bs256 {
    type Output = Bs256;

    fn bitor(self, rhs: Self) -> Self::Output {
        Bs256 {
            high: self.high | rhs.high,
            low: self.low | rhs.low,
        }
    }
}

impl Not for Bs256 {
    type Output = Bs256;

    fn not(self) -> Self::Output {
        Bs256 {
            high: !self.high,
            low: !self.low,
        }
    }
}

impl Ord for Bs256 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.high.cmp(&other.high) {
            std::cmp::Ordering::Less => std::cmp::Ordering::Less,
            std::cmp::Ordering::Greater => std::cmp::Ordering::Greater,
            std::cmp::Ordering::Equal => self.low.cmp(&other.low),
        }
    }
}

impl PartialOrd for Bs256 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Display for Bs256 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {})", self.high, self.low)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn foo256() {
        let mut tmp: BitSet<Bs256> = BitSet::new();
        println!("{:?}", tmp);
        tmp.insert(133);
        println!("{:?}", tmp);

        assert!(tmp.contains(&133));
    }
}
