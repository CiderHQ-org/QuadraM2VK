//! The spare scratch block of `generate_mipmaps`' filter chains (`mipmap_scratch.rs`): repeated
//! calls make no new block once the device holds one large enough, a leased block goes to no
//! other call, and the device keeps one spare, the largest.

use super::super::mipmap_scratch::{Spare, Take};

/// Block `id` of `size` bytes, as a test's stand-in.
type Block = (u32, u64);

#[test]
fn repeated_calls_make_no_new_block() {
    let mut spare: Spare<Block> = Spare::default();
    let mut made = 0;
    let mut lease = |spare: &mut Spare<Block>, need: u64| match spare.take(need) {
        Take::Reuse(block, size) => {
            assert!(size >= need);
            block
        }
        Take::Make(stale) => {
            assert_eq!(
                stale, None,
                "a spare too small is released as a larger block is made"
            );
            made += 1;
            (made, need)
        }
    };
    // The first call makes a block; every later call of that size or less, one at a time,
    // reuses it.
    for need in [4096, 4096, 64, 1024, 4096, 1] {
        let block = lease(&mut spare, need);
        assert_eq!(block, (1, 4096), "need {need}");
        assert_eq!(spare.give(block, block.1), None, "nothing to release");
    }
    assert_eq!(made, 1);
    assert_eq!(spare.take(4096), Take::Reuse((1, 4096), 4096));
}

#[test]
fn a_leased_block_goes_to_no_other_call() {
    let mut spare: Spare<Block> = Spare::default();
    assert_eq!(spare.take(100), Take::Make(None));
    assert_eq!(spare.give((1, 100), 100), None);
    // Two calls in flight at once: the first leases the spare, the second makes its own.
    assert_eq!(spare.take(100), Take::Reuse((1, 100), 100));
    assert_eq!(spare.take(100), Take::Make(None));
    // Both given back: the device keeps one, and the other is released.
    assert_eq!(spare.give((2, 100), 100), None);
    assert_eq!(spare.give((1, 100), 100), Some((1, 100)));
    assert_eq!(spare.take(100), Take::Reuse((2, 100), 100));
}

#[test]
fn the_spare_grows_to_the_largest_chain() {
    let mut spare: Spare<Block> = Spare::default();
    assert_eq!(spare.give((1, 64), 64), None);
    // A larger chain releases the spare too small and makes its own.
    assert_eq!(spare.take(4096), Take::Make(Some((1, 64))));
    assert_eq!(spare.give((2, 4096), 4096), None);
    // A smaller block given back beside a larger spare is the one released, and the other way
    // round.
    assert_eq!(spare.give((3, 64), 64), Some((3, 64)));
    assert_eq!(spare.give((4, 8192), 8192), Some((2, 4096)));
    assert_eq!(spare.take(1), Take::Reuse((4, 8192), 8192));
}
