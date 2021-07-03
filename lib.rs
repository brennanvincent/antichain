/// A collection of updates of the form `(T, i64)`.
///
/// A `ChangeBatch` accumulates updates of the form `(T, i64)`, where it is capable of consolidating
/// the representation and removing elements whose `i64` field accumulates to zero.
///
/// The implementation is designed to be as lazy as possible, simply appending to a list of updates
/// until they are required. This means that several seemingly simple operations may be expensive, in
/// that they may provoke a compaction. I've tried to prevent exposing methods that allow surprisingly
/// expensive operations; all operations should take an amortized constant or logarithmic time.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChangeBatch<T> {
    // A list of updates to which we append.
    updates: Vec<(T, i64)>,
    // The length of the prefix of `self.updates` known to be compact.
    clean: usize,
}

impl<T: Ord> ChangeBatch<T> {
    /// Allocates a new empty `ChangeBatch`.
    ///
    /// # Examples
    ///
    ///```
    /// use timely::progress::ChangeBatch;
    ///
    /// let mut batch = ChangeBatch::<usize>::new();
    /// assert!(batch.is_empty());
    ///```
    pub fn new() -> ChangeBatch<T> {
        ChangeBatch {
            updates: Vec::new(),
            clean: 0,
        }
    }

    /// Allocates a new `ChangeBatch` with a single entry.
    ///
    /// # Examples
    ///
    ///```
    /// use timely::progress::ChangeBatch;
    ///
    /// let mut batch = ChangeBatch::<usize>::new_from(17, 1);
    /// assert!(!batch.is_empty());
    ///```
    pub fn new_from(key: T, val: i64) -> ChangeBatch<T> {
        let mut result = ChangeBatch::new();
        result.update(key, val);
        result
    }

    /// Returns true if the change batch is not guaranteed compact.
    pub fn is_dirty(&self) -> bool {
        self.updates.len() > self.clean
    }

    /// Adds a new update, for `item` with `value`.
    ///
    /// This could be optimized to perform compaction when the number of "dirty" elements exceeds
    /// half the length of the list, which would keep the total footprint within reasonable bounds
    /// even under an arbitrary number of updates. This has a cost, and it isn't clear whether it
    /// is worth paying without some experimentation.
    ///
    /// # Examples
    ///
    ///```
    /// use timely::progress::ChangeBatch;
    ///
    /// let mut batch = ChangeBatch::<usize>::new();
    /// batch.update(17, 1);
    /// assert!(!batch.is_empty());
    ///```
    #[inline]
    pub fn update(&mut self, item: T, value: i64) {
        self.updates.push((item, value));
        self.maintain_bounds();
    }

    /// Performs a sequence of updates described by `iterator`.
    ///
    /// # Examples
    ///
    ///```
    /// use timely::progress::ChangeBatch;
    ///
    /// let mut batch = ChangeBatch::<usize>::new_from(17, 1);
    /// batch.extend(vec![(17, -1)].into_iter());
    /// assert!(batch.is_empty());
    ///```
    #[inline]
    pub fn extend<I: Iterator<Item = (T, i64)>>(&mut self, iterator: I) {
        self.updates.extend(iterator);
        self.maintain_bounds();
    }

    /// Extracts the `Vec<(T, i64)>` from the map, consuming it.
    ///
    /// # Examples
    ///
    ///```
    /// use timely::progress::ChangeBatch;
    ///
    /// let batch = ChangeBatch::<usize>::new_from(17, 1);
    /// assert_eq!(batch.into_inner(), vec![(17, 1)]);
    ///```
    pub fn into_inner(mut self) -> Vec<(T, i64)> {
        self.compact();
        self.updates
    }

    /// Iterates over the contents of the map.
    ///
    /// # Examples
    ///
    ///```
    /// use timely::progress::ChangeBatch;
    ///
    /// let mut batch = ChangeBatch::<usize>::new_from(17, 1);
    /// {   // scope allows borrow of `batch` to drop.
    ///     let mut iter = batch.iter();
    ///     assert_eq!(iter.next(), Some(&(17, 1)));
    ///     assert_eq!(iter.next(), None);
    /// }
    /// assert!(!batch.is_empty());
    ///```
    #[inline]
    pub fn iter(&mut self) -> ::std::slice::Iter<(T, i64)> {
        self.compact();
        self.updates.iter()
    }

    /// Drains the set of updates.
    ///
    /// This operation first compacts the set of updates so that the drained results
    /// have at most one occurence of each item.
    ///
    /// # Examples
    ///
    ///```
    /// use timely::progress::ChangeBatch;
    ///
    /// let mut batch = ChangeBatch::<usize>::new_from(17, 1);
    /// {   // scope allows borrow of `batch` to drop.
    ///     let mut iter = batch.drain();
    ///     assert_eq!(iter.next(), Some((17, 1)));
    ///     assert_eq!(iter.next(), None);
    /// }
    /// assert!(batch.is_empty());
    ///```
    #[inline]
    pub fn drain(&mut self) -> ::std::vec::Drain<(T, i64)> {
        self.compact();
        self.clean = 0;
        self.updates.drain(..)
    }

    /// Clears the map.
    ///
    /// # Examples
    ///
    ///```
    /// use timely::progress::ChangeBatch;
    ///
    /// let mut batch = ChangeBatch::<usize>::new_from(17, 1);
    /// batch.clear();
    /// assert!(batch.is_empty());
    ///```
    #[inline]
    pub fn clear(&mut self) {
        self.updates.clear();
        self.clean = 0;
    }

    /// True iff all keys have value zero.
    ///
    /// This method requires mutable access to `self` because it may need to compact the representation
    /// to determine if the batch of updates is indeed empty. We could also implement a weaker form of
    /// `is_empty` which just checks the length of `self.updates`, and which could confirm the absence of
    /// any updates, but could report false negatives if there are updates which would cancel.
    ///
    /// # Examples
    ///
    ///```
    /// use timely::progress::ChangeBatch;
    ///
    /// let mut batch = ChangeBatch::<usize>::new_from(17, 1);
    /// batch.update(17, -1);
    /// assert!(batch.is_empty());
    ///```
    #[inline]
    pub fn is_empty(&mut self) -> bool {
        if self.clean > self.updates.len() / 2 {
            false
        } else {
            self.compact();
            self.updates.is_empty()
        }
    }

    /// Compact and sort data, so that two instances can be compared without false negatives.
    #[deprecated(since = "0.9.0", note = "please use `compact` instead")]
    pub fn canonicalize(&mut self) {
        self.compact();
        self.updates.sort_by(|x, y| x.0.cmp(&y.0));
    }

    /// Drains `self` into `other`.
    ///
    /// This method has similar a effect to calling `other.extend(self.drain())`, but has the
    /// opportunity to optimize this to a `::std::mem::swap(self, other)` when `other` is empty.
    /// As many uses of this method are to propagate updates, this optimization can be quite
    /// handy.
    ///
    /// # Examples
    ///
    ///```
    /// use timely::progress::ChangeBatch;
    ///
    /// let mut batch1 = ChangeBatch::<usize>::new_from(17, 1);
    /// let mut batch2 = ChangeBatch::new();
    /// batch1.drain_into(&mut batch2);
    /// assert!(batch1.is_empty());
    /// assert!(!batch2.is_empty());
    ///```
    #[inline]
    pub fn drain_into(&mut self, other: &mut ChangeBatch<T>)
    where
        T: Clone,
    {
        if other.updates.is_empty() {
            ::std::mem::swap(self, other);
        } else {
            other.extend(self.updates.drain(..));
            self.clean = 0;
        }
    }

    /// Compact the internal representation.
    ///
    /// This method sort `self.updates` and consolidates elements with equal item, discarding
    /// any whose accumulation is zero. It is optimized to only do this if the number of dirty
    /// elements is non-zero.
    #[inline]
    pub fn compact(&mut self) {
        if self.clean < self.updates.len() && self.updates.len() > 1 {
            self.updates.sort_by(|x, y| x.0.cmp(&y.0));
            for i in 0..self.updates.len() - 1 {
                if self.updates[i].0 == self.updates[i + 1].0 {
                    self.updates[i + 1].1 += self.updates[i].1;
                    self.updates[i].1 = 0;
                }
            }
            self.updates.retain(|x| x.1 != 0);
        }
        self.clean = self.updates.len();
    }

    /// Expose the internal vector of updates.
    pub fn unstable_internal_updates(&self) -> &Vec<(T, i64)> {
        &self.updates
    }

    /// Expose the internal value of `clean`.
    pub fn unstable_internal_clean(&self) -> usize {
        self.clean
    }

    /// Maintain the bounds of pending (non-compacted) updates versus clean (compacted) data.
    /// This function tries to minimize work by only compacting if enough work has accumulated.
    fn maintain_bounds(&mut self) {
        // if we have more than 32 elements and at least half of them are not clean, compact
        if self.updates.len() > 32 && self.updates.len() >> 1 >= self.clean {
            self.compact()
        }
    }
}
/// A type that is partially ordered.
///
/// This trait is distinct from Rust's `PartialOrd` trait, because the implementation
/// of that trait precludes a distinct `Ord` implementation. We need an independent
/// trait if we want to have a partially ordered type that can also be sorted.
pub trait PartialOrder: Eq {
    /// Returns true iff one element is strictly less than the other.
    fn less_than(&self, other: &Self) -> bool {
        self.less_equal(other) && self != other
    }
    /// Returns true iff one element is less than or equal to the other.
    fn less_equal(&self, other: &Self) -> bool;
}

/// A set of mutually incomparable elements.
///
/// An antichain is a set of partially ordered elements, each of which is incomparable to the others.
/// This antichain implementation allows you to repeatedly introduce elements to the antichain, and
/// which will evict larger elements to maintain the *minimal* antichain, those incomparable elements
/// no greater than any other element.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Antichain<T> {
    elements: Vec<T>,
}

impl<T: PartialOrder> Antichain<T> {
    /// Updates the `Antichain` if the element is not greater than or equal to some present element.
    ///
    /// Returns true if element is added to the set
    ///
    /// # Examples
    ///
    ///```
    /// use timely::progress::frontier::Antichain;
    ///
    /// let mut frontier = Antichain::new();
    /// assert!(frontier.insert(2));
    /// assert!(!frontier.insert(3));
    ///```
    pub fn insert(&mut self, element: T) -> bool {
        if !self.elements.iter().any(|x| x.less_equal(&element)) {
            self.elements.retain(|x| !element.less_equal(x));
            self.elements.push(element);
            true
        } else {
            false
        }
    }

    /// Performs a sequence of insertion and return true iff any insertion does.
    ///
    /// # Examples
    ///
    ///```
    /// use timely::progress::frontier::Antichain;
    ///
    /// let mut frontier = Antichain::new();
    /// assert!(frontier.extend(Some(3)));
    /// assert!(frontier.extend(vec![2, 5]));
    /// assert!(!frontier.extend(vec![3, 4]));
    ///```
    pub fn extend<I: IntoIterator<Item = T>>(&mut self, iterator: I) -> bool {
        let mut added = false;
        for element in iterator {
            added = self.insert(element) || added;
        }
        added
    }

    /// Creates a new empty `Antichain`.
    ///
    /// # Examples
    ///
    ///```
    /// use timely::progress::frontier::Antichain;
    ///
    /// let mut frontier = Antichain::<u32>::new();
    ///```
    pub fn new() -> Antichain<T> {
        Antichain {
            elements: Vec::new(),
        }
    }

    /// Creates a new singleton `Antichain`.
    ///
    /// # Examples
    ///
    ///```
    /// use timely::progress::frontier::Antichain;
    ///
    /// let mut frontier = Antichain::from_elem(2);
    ///```
    pub fn from_elem(element: T) -> Antichain<T> {
        Antichain {
            elements: vec![element],
        }
    }

    /// Clears the contents of the antichain.
    ///
    /// # Examples
    ///
    ///```
    /// use timely::progress::frontier::Antichain;
    ///
    /// let mut frontier = Antichain::from_elem(2);
    /// frontier.clear();
    /// assert!(frontier.elements().is_empty());
    ///```
    pub fn clear(&mut self) {
        self.elements.clear()
    }

    /// Sorts the elements so that comparisons between antichains can be made.
    pub fn sort(&mut self)
    where
        T: Ord,
    {
        self.elements.sort()
    }

    /// Returns true if any item in the antichain is strictly less than the argument.
    ///
    /// # Examples
    ///
    ///```
    /// use timely::progress::frontier::Antichain;
    ///
    /// let mut frontier = Antichain::from_elem(2);
    /// assert!(frontier.less_than(&3));
    /// assert!(!frontier.less_than(&2));
    /// assert!(!frontier.less_than(&1));
    ///
    /// frontier.clear();
    /// assert!(!frontier.less_than(&3));
    ///```
    #[inline]
    pub fn less_than(&self, time: &T) -> bool {
        self.elements.iter().any(|x| x.less_than(time))
    }

    /// Returns true if any item in the antichain is less than or equal to the argument.
    ///
    /// # Examples
    ///
    ///```
    /// use timely::progress::frontier::Antichain;
    ///
    /// let mut frontier = Antichain::from_elem(2);
    /// assert!(frontier.less_equal(&3));
    /// assert!(frontier.less_equal(&2));
    /// assert!(!frontier.less_equal(&1));
    ///
    /// frontier.clear();
    /// assert!(!frontier.less_equal(&3));
    ///```
    #[inline]
    pub fn less_equal(&self, time: &T) -> bool {
        self.elements.iter().any(|x| x.less_equal(time))
    }

    /// Returns true if every element of `other` is greater or equal to some element of `self`.
    #[inline]
    pub fn dominates(&self, other: &Antichain<T>) -> bool {
        other
            .elements()
            .iter()
            .all(|t2| self.elements().iter().any(|t1| t1.less_equal(t2)))
    }

    /// Reveals the elements in the antichain.
    ///
    /// # Examples
    ///
    ///```
    /// use timely::progress::frontier::Antichain;
    ///
    /// let mut frontier = Antichain::from_elem(2);
    /// assert_eq!(frontier.elements(), &[2]);
    ///```
    #[inline]
    pub fn elements(&self) -> &[T] {
        &self.elements[..]
    }
}

/// An antichain based on a multiset whose elements frequencies can be updated.
///
/// The `MutableAntichain` maintains frequencies for many elements of type `T`, and exposes the set
/// of elements with positive count not greater than any other elements with positive count. The
/// antichain may both advance and retreat; the changes do not all need to be to elements greater or
/// equal to some elements of the frontier.
///
/// The type `T` must implement `PartialOrder` as well as `Ord`. The implementation of the `Ord` trait
/// is used to efficiently organize the updates for cancellation, and to efficiently determine the lower
/// bounds, and only needs to not contradict the `PartialOrder` implementation (that is, if `PartialOrder`
/// orders two elements, the so does the `Ord` implementation).
///
/// The `MutableAntichain` implementation is done with the intent that updates to it are done in batches,
/// and it is acceptable to rebuild the frontier from scratch when a batch of updates change it. This means
/// that it can be expensive to maintain a large number of counts and change few elements near the frontier.
///
/// There is an `update_dirty` method for single updates that leave the `MutableAntichain` in a dirty state,
/// but I strongly recommend against using them unless you must (on part of timely progress tracking seems
/// to be greatly simplified by access to this)
#[derive(Clone, Debug)]
pub struct MutableAntichain<T: PartialOrder + Ord> {
    dirty: usize,
    updates: Vec<(T, i64)>,
    frontier: Vec<T>,
    changes: ChangeBatch<T>,
}

impl<T: PartialOrder + Ord + Clone> MutableAntichain<T> {
    /// Creates a new empty `MutableAntichain`.
    ///
    /// # Examples
    ///
    ///```
    /// use timely::progress::frontier::MutableAntichain;
    ///
    /// let frontier = MutableAntichain::<usize>::new();
    /// assert!(frontier.is_empty());
    ///```
    #[inline]
    pub fn new() -> MutableAntichain<T> {
        MutableAntichain {
            dirty: 0,
            updates: Vec::new(),
            frontier: Vec::new(),
            changes: ChangeBatch::new(),
        }
    }

    /// Removes all elements.
    ///
    /// # Examples
    ///
    ///```
    /// use timely::progress::frontier::MutableAntichain;
    ///
    /// let mut frontier = MutableAntichain::<usize>::new();
    /// frontier.clear();
    /// assert!(frontier.is_empty());
    ///```
    #[inline]
    pub fn clear(&mut self) {
        self.dirty = 0;
        self.updates.clear();
        self.frontier.clear();
        self.changes.clear();
    }

    /// This method deletes the contents. Unlike `clear` it records doing so.
    pub fn empty(&mut self) {
        for index in 0..self.updates.len() {
            self.updates[index].1 = 0;
        }
        self.dirty = self.updates.len();
    }

    /// Reveals the minimal elements with positive count.
    ///
    /// # Examples
    ///
    ///```
    /// use timely::progress::frontier::MutableAntichain;
    ///
    /// let mut frontier = MutableAntichain::<usize>::new();
    /// assert!(frontier.frontier().len() == 0);
    ///```
    #[inline]
    pub fn frontier(&self) -> AntichainRef<T> {
        debug_assert_eq!(self.dirty, 0);
        AntichainRef::new(&self.frontier)
    }

    /// Creates a new singleton `MutableAntichain`.
    ///
    /// # Examples
    ///
    ///```
    /// use timely::progress::frontier::{AntichainRef, MutableAntichain};
    ///
    /// let mut frontier = MutableAntichain::new_bottom(0u64);
    /// assert!(frontier.frontier() == AntichainRef::new(&[0u64]));
    ///```
    #[inline]
    pub fn new_bottom(bottom: T) -> MutableAntichain<T> {
        MutableAntichain {
            dirty: 0,
            updates: vec![(bottom.clone(), 1)],
            frontier: vec![bottom],
            changes: ChangeBatch::new(),
        }
    }

    /// Returns true if there are no elements in the `MutableAntichain`.
    ///
    /// # Examples
    ///
    ///```
    /// use timely::progress::frontier::MutableAntichain;
    ///
    /// let mut frontier = MutableAntichain::<usize>::new();
    /// assert!(frontier.is_empty());
    ///```
    #[inline]
    pub fn is_empty(&self) -> bool {
        debug_assert_eq!(self.dirty, 0);
        self.frontier.is_empty()
    }

    /// Returns true if any item in the `MutableAntichain` is strictly less than the argument.
    ///
    /// # Examples
    ///
    ///```
    /// use timely::progress::frontier::MutableAntichain;
    ///
    /// let mut frontier = MutableAntichain::new_bottom(1u64);
    /// assert!(!frontier.less_than(&0));
    /// assert!(!frontier.less_than(&1));
    /// assert!(frontier.less_than(&2));
    ///```
    #[inline]
    pub fn less_than(&self, time: &T) -> bool {
        debug_assert_eq!(self.dirty, 0);
        self.frontier().less_than(time)
    }

    /// Returns true if any item in the `MutableAntichain` is less than or equal to the argument.
    ///
    /// # Examples
    ///
    ///```
    /// use timely::progress::frontier::MutableAntichain;
    ///
    /// let mut frontier = MutableAntichain::new_bottom(1u64);
    /// assert!(!frontier.less_equal(&0));
    /// assert!(frontier.less_equal(&1));
    /// assert!(frontier.less_equal(&2));
    ///```
    #[inline]
    pub fn less_equal(&self, time: &T) -> bool {
        debug_assert_eq!(self.dirty, 0);
        self.frontier().less_equal(time)
    }

    /// Allows a single-element push, but dirties the antichain and prevents inspection until cleaned.
    ///
    /// At the moment inspection is prevented via panic, so best be careful (this should probably be fixed).
    /// It is *very* important if you want to use this method that very soon afterwards you call something
    /// akin to `update_iter`, perhaps with a `None` argument if you have no more data, as this method will
    /// tidy up the internal representation.
    #[inline]
    pub fn update_dirty(&mut self, time: T, delta: i64) {
        self.updates.push((time, delta));
        self.dirty += 1;
    }

    /// Applies updates to the antichain and enumerates any changes.
    ///
    /// # Examples
    ///
    ///```
    /// use timely::progress::frontier::{AntichainRef, MutableAntichain};
    ///
    /// let mut frontier = MutableAntichain::new_bottom(1u64);
    /// let changes =
    /// frontier
    ///     .update_iter(vec![(1, -1), (2, 7)])
    ///     .collect::<Vec<_>>();
    ///
    /// assert!(frontier.frontier() == AntichainRef::new(&[2]));
    /// assert!(changes == vec![(1, -1), (2, 1)]);
    ///```
    #[inline]
    pub fn update_iter<'a, I>(&'a mut self, updates: I) -> ::std::vec::Drain<'a, (T, i64)>
    where
        I: IntoIterator<Item = (T, i64)>,
    {
        for (time, delta) in updates {
            self.updates.push((time, delta));
            self.dirty += 1;
        }

        // track whether a rebuild is needed.
        let mut rebuild_required = false;

        // determine if recently pushed data requires rebuilding the frontier.
        // note: this may be required even with an empty iterator, due to dirty data in self.updates.
        while self.dirty > 0 && !rebuild_required {
            let time = &self.updates[self.updates.len() - self.dirty].0;
            let delta = self.updates[self.updates.len() - self.dirty].1;

            let beyond_frontier = self.frontier.iter().any(|f| f.less_than(time));
            let before_frontier = !self.frontier.iter().any(|f| f.less_equal(time));
            rebuild_required =
                rebuild_required || !(beyond_frontier || (delta < 0 && before_frontier));

            self.dirty -= 1;
        }
        self.dirty = 0;

        if rebuild_required {
            self.rebuild()
        }
        self.changes.drain()
    }

    /// Applies updates to the antichain and applies `action` to each frontier change.
    ///
    /// This method applies a batch of updates and if any affects the frontier it is rebuilt.
    /// Once rebuilt, `action` is called with the corresponding changes to the frontier, which
    /// should be various times and `{ +1, -1 }` differences.
    ///
    /// # Examples
    ///
    ///```
    /// use timely::progress::frontier::{AntichainRef, MutableAntichain};
    ///
    /// let mut frontier = MutableAntichain::new_bottom(1u64);
    /// let mut changes = Vec::new();
    /// frontier.update_iter_and(vec![(1, -1), (2, 1)], |time, diff| {
    ///     changes.push((time.clone(), diff));
    /// });
    /// assert!(frontier.frontier() == AntichainRef::new(&[2]));
    /// changes.sort();
    /// assert_eq!(&changes[..], &[(1, -1), (2, 1)]);
    ///```
    #[inline]
    #[deprecated(
        since = "0.8.0",
        note = "`update_iter` now provides an iterator as output"
    )]
    pub fn update_iter_and<I, A>(&mut self, updates: I, mut action: A)
    where
        I: IntoIterator<Item = (T, i64)>,
        A: FnMut(&T, i64),
    {
        self.update_iter(updates)
            .for_each(|(time, diff)| action(&time, diff));
    }

    /// Sorts and consolidates `self.updates` and applies `action` to any frontier changes.
    ///
    /// This method is meant to be used for bulk updates to the frontier, and does more work than one might do
    /// for single updates, but is meant to be an efficient way to process multiple updates together. This is
    /// especially true when we want to apply very large numbers of updates.
    fn rebuild(&mut self) {
        // sort and consolidate updates; retain non-zero accumulations.
        if !self.updates.is_empty() {
            self.updates.sort_by(|x, y| x.0.cmp(&y.0));
            for i in 0..self.updates.len() - 1 {
                if self.updates[i].0 == self.updates[i + 1].0 {
                    self.updates[i + 1].1 += self.updates[i].1;
                    self.updates[i].1 = 0;
                }
            }
            self.updates.retain(|x| x.1 != 0);
        }

        for time in self.frontier.drain(..) {
            self.changes.update(time, -1);
        }

        // build new frontier using strictly positive times.
        // as the times are sorted, we don't need to worry that we might displace frontier elements.
        for time in self.updates.iter().filter(|x| x.1 > 0) {
            if !self.frontier.iter().any(|f| f.less_equal(&time.0)) {
                self.frontier.push(time.0.clone());
            }
        }

        for time in self.frontier.iter() {
            self.changes.update(time.clone(), 1);
        }
    }

    /// Reports the count for a queried time.
    pub fn count_for(&self, query_time: &T) -> i64 {
        self.updates
            .iter()
            .filter(|td| td.0.eq(query_time))
            .map(|td| td.1)
            .sum()
    }
}

/// A wrapper for elements of an antichain.
#[derive(PartialEq, Eq)]
pub struct AntichainRef<'a, T: 'a + PartialOrder> {
    /// Elements contained in the antichain.
    frontier: &'a [T],
}

impl<'a, T: 'a + PartialOrder> AntichainRef<'a, T> {
    /// Create a new `AntichainRef` from a reference to a slice of elements forming the frontier.
    pub fn new(frontier: &'a [T]) -> Self {
        Self { frontier }
    }

    /// Returns true if there are no elements in the `AntichainRef`.
    ///
    /// # Examples
    ///
    ///```
    /// use timely::progress::frontier::AntichainRef;
    ///
    /// let frontier = AntichainRef::<usize>::new(&[]);
    /// assert!(frontier.is_empty());
    ///```
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.frontier.is_empty()
    }

    /// Create an iterator over the elements in this `AntichainRef`.
    ///
    /// # Examples
    ///
    ///```
    /// use timely::progress::frontier::AntichainRef;
    ///
    /// let frontier = AntichainRef::new(&[1u64]);
    /// let mut iter = frontier.iter();
    /// assert_eq!(iter.next(), Some(&1u64));
    /// assert_eq!(iter.next(), None);
    ///```
    pub fn iter(&self) -> ::std::slice::Iter<T> {
        self.frontier.iter()
    }

    /// Returns true if any item in the `AntichainRef` is strictly less than the argument.
    ///
    /// # Examples
    ///
    ///```
    /// use timely::progress::frontier::AntichainRef;
    ///
    /// let frontier = AntichainRef::new(&[1u64]);
    /// assert!(!frontier.less_than(&0));
    /// assert!(!frontier.less_than(&1));
    /// assert!(frontier.less_than(&2));
    ///```
    #[inline]
    pub fn less_than(&self, time: &T) -> bool {
        self.iter().any(|x| x.less_than(time))
    }

    /// Returns true if any item in the `AntichainRef` is less than or equal to the argument.
    #[inline]
    ///
    /// # Examples
    ///
    ///```
    /// use timely::progress::frontier::AntichainRef;
    ///
    /// let frontier = AntichainRef::new(&[1u64]);
    /// assert!(!frontier.less_equal(&0));
    /// assert!(frontier.less_equal(&1));
    /// assert!(frontier.less_equal(&2));
    ///```
    pub fn less_equal(&self, time: &T) -> bool {
        self.iter().any(|x| x.less_equal(time))
    }

    /// Returns the number of elements in this `AntichainRef`.
    pub fn len(&self) -> usize {
        self.frontier.len()
    }

    /// Copies `self` into a new `Vec`.
    pub fn to_vec(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.frontier.to_vec()
    }
}

impl<'a, T: PartialOrder> ::std::ops::Deref for AntichainRef<'a, T> {
    type Target = [T];
    fn deref(&self) -> &Self::Target {
        self.frontier
    }
}

impl<'a, T: 'a + PartialOrder> ::std::iter::IntoIterator for &'a AntichainRef<'a, T> {
    type Item = &'a T;
    type IntoIter = ::std::slice::Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
