use crate::range::RangeAllocator;

#[test]
fn allocations_are_aligned_first_fit_and_disjoint() {
    let mut ranges = RangeAllocator::new(0, 1024);
    assert_eq!(ranges.allocate(10, 1), Some(0));
    assert_eq!(ranges.allocate(16, 64), Some(64));
    // The gap [10, 64) holds an aligned 32 at 32.
    assert_eq!(ranges.allocate(32, 32), Some(32));
    assert_eq!(ranges.allocate(8, 8), Some(16));
    assert_eq!(ranges.used(), 10 + 16 + 32 + 8);
    assert_eq!(ranges.allocate(0, 1), None, "a size of 0 allocates nothing");
    assert_eq!(ranges.allocate(2048, 1), None, "larger than the space");
}

#[test]
fn frees_coalesce_in_any_order() {
    for order in [[0, 1, 2, 3], [3, 2, 1, 0], [1, 3, 0, 2], [2, 0, 3, 1]] {
        let mut ranges = RangeAllocator::new(4096, 4096 + 400);
        let starts: Vec<u64> = (0..4).map(|_| ranges.allocate(100, 4).unwrap()).collect();
        assert_eq!(starts, [4096, 4196, 4296, 4396]);
        assert_eq!(ranges.allocate(1, 1), None, "full");
        for index in order {
            assert!(!ranges.is_empty());
            ranges.free(starts[index], 100);
        }
        assert!(ranges.is_empty(), "order {order:?} left fragments");
        assert_eq!(ranges.allocate(400, 1), Some(4096), "one range again");
    }
}

#[test]
fn a_freed_range_is_reused() {
    let mut ranges = RangeAllocator::new(0, 300);
    let a = ranges.allocate(100, 1).unwrap();
    let b = ranges.allocate(100, 1).unwrap();
    ranges.free(a, 100);
    assert_eq!(ranges.allocate(50, 1), Some(a));
    assert_eq!(ranges.allocate(50, 1), Some(a + 50));
    assert_eq!(ranges.allocate(100, 1), Some(200));
    assert_eq!(ranges.allocate(1, 1), None);
    ranges.free(b, 100);
    assert_eq!(ranges.used(), 200);
}
