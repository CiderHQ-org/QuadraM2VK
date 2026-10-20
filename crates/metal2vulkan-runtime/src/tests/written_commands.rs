//! What an indirect command buffer's records may hold (`written_commands.rs`): each route adds,
//! a blit copy's destination gains what its source gains later, and nothing is taken away.

use std::sync::Arc;

use metal2vulkan::bind::indirect_commands::{Action, CommandRecord};
use metal2vulkan::pipeline::CommandWrites as W;

use crate::written_commands::{Named, Writers, WrittenCommands};

fn draws(primitive_types: u8) -> W {
    W {
        operations: W::DRAW,
        primitive_types,
    }
}

/// A record the CPU left adds its draw's operation and primitive type, its barrier, and the
/// state it names; a type past Metal's five adds every type; an empty record adds nothing.
#[test]
fn a_record_adds_what_it_holds() {
    let written = Arc::new(WrittenCommands::default());
    written.add_record(&CommandRecord::default(), None);
    assert!(written.writes().is_empty());
    let mut record = CommandRecord {
        action: Action::DrawPrimitives {
            primitive_type: 3,
            vertex_start: 0,
            vertex_count: 3,
            instance_count: 1,
            base_instance: 0,
        },
        ..CommandRecord::default()
    };
    written.add_record(&record, None);
    assert_eq!(written.writes(), draws(0b1000));
    record.barrier = true;
    let named = Named {
        encodes: true,
        writes: W::NONE,
    };
    written.add_record(&record, Some(named));
    assert_eq!(
        written.writes().operations,
        W::DRAW | W::BARRIER | W::PIPELINE
    );
    assert!(written.named().encodes);
    record.action = Action::DrawIndexedPrimitives {
        primitive_type: 9,
        index_count: 3,
        index_type: 0,
        index_buffer: 0,
        instance_count: 1,
        base_vertex: 0,
        base_instance: 0,
    };
    written.add_record(&record, None);
    assert_eq!(written.writes().primitive_types, W::ALL_PRIMITIVE_TYPES);
}

/// A writer that may set a pipeline or copy a command may name any state; one that only draws
/// names none.
#[test]
fn a_writer_that_sets_pipelines_names_any_state() {
    let drawing = Arc::new(WrittenCommands::default());
    drawing.add_writer(draws(0b1000));
    assert_eq!(drawing.named(), Named::default());
    for bit in [W::PIPELINE, W::COPY] {
        let setting = Arc::new(WrittenCommands::default());
        setting.add_writer(W {
            operations: bit,
            primitive_types: 0,
        });
        assert!(setting.named().encodes);
        assert_eq!(setting.named().writes, W::ANY);
    }
}

/// A copy's destination takes its source's writes at the copy and what the source gains after,
/// through a chain of copies and around a cycle; the source gains nothing from its destination.
#[test]
fn a_copy_gains_what_its_source_gains() {
    let [a, b, c] = [(); 3].map(|_| Arc::new(WrittenCommands::default()));
    a.add_writer(draws(0b1));
    b.copied_from(&a);
    assert_eq!(b.writes(), draws(0b1));
    c.copied_from(&b);
    a.add_writer(draws(0b1000));
    assert_eq!(c.writes(), draws(0b1001));
    b.add_writer(W {
        operations: W::DRAW_INDEXED,
        primitive_types: 0,
    });
    assert_eq!(a.writes(), draws(0b1001));
    assert_eq!(c.writes().operations, W::DRAW | W::DRAW_INDEXED);
    // A cycle ends.
    a.copied_from(&c);
    c.add_writer(draws(0b10));
    for buffer in [&a, &b, &c] {
        assert_eq!(buffer.writes().primitive_types, 0b1011);
    }
}

/// An encoder's writers: a buffer made resident takes the state already set and each set after,
/// and an execution's named states, however many times it is made resident; a buffer it did not
/// make resident takes none.
#[test]
fn an_encoder_adds_its_states() {
    let buffer = Arc::new(WrittenCommands::default());
    let other = Arc::new(WrittenCommands::default());
    let mut writers = Writers::default();
    writers.state_set(draws(0b1));
    assert!(buffer.writes().is_empty());
    writers.made_writable(&buffer);
    writers.made_writable(&buffer);
    assert_eq!(buffer.writes(), draws(0b1));
    writers.state_set(draws(0b100));
    writers.runs(W {
        operations: W::DISPATCH,
        primitive_types: 0,
    });
    assert_eq!(buffer.writes().operations, W::DRAW | W::DISPATCH);
    assert_eq!(buffer.writes().primitive_types, 0b101);
    assert!(other.writes().is_empty());
}
