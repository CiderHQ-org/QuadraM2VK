use super::*;

fn of(name: &str) -> Option<MatMul2dFn> {
    MatMul2dFn::of_declaration(name.as_bytes(), Some(SECTION))
}

fn tensor(space: TensorSpace, elem: DataType) -> OperandForm {
    OperandForm::Tensor { space, elem }
}

fn coop(elem: DataType) -> OperandForm {
    OperandForm::Cooperative { elem }
}

#[test]
fn a_declaration_is_the_abi_only_in_its_section_and_under_its_prefix() {
    let name = b"__tensorops_impl_matmul2d_op_cooperative_tensor_init";
    assert_eq!(
        MatMul2dFn::of_declaration(name, Some(SECTION)),
        Some(MatMul2dFn::TensorInit)
    );
    assert_eq!(MatMul2dFn::of_declaration(name, None), None);
    assert_eq!(
        MatMul2dFn::of_declaration(name, Some(b"air.static_init")),
        None
    );
    assert_eq!(
        MatMul2dFn::of_declaration(b"__tensorops_impl_conv2d_op_run", Some(SECTION)),
        None
    );
    // A name of the family the contract does not model is the family, unmodeled.
    assert_eq!(
        of("__tensorops_impl_matmul2d_op_cooperative_tensor_copy"),
        Some(MatMul2dFn::Unmodeled)
    );
    assert_eq!(
        of("__tensorops_impl_matmul2d_op_cooperative_destination_tensor_init"),
        Some(MatMul2dFn::Unmodeled)
    );
    assert_eq!(
        MatMul2dFn::of_declaration(b"__tensorops_impl_matmul2d_op_\xff", Some(SECTION)),
        Some(MatMul2dFn::Unmodeled)
    );
}

/// Every `run` spelling the corpus calls, with its operands and its declaration's arity (the
/// arities the corpus declarations carry).
#[test]
fn the_corpus_run_forms_parse_with_their_arity() {
    use DataType::{BFloat16 as B, Float16 as H, Float32 as F};
    use TensorSpace::{Device as D, Threadgroup as T};
    let cases = [
        (
            "run_cooperative_dv_f32_tg_f32_f32",
            tensor(D, F),
            tensor(T, F),
            7,
        ),
        (
            "run_cooperative_dv_f16_tg_f16_f32",
            tensor(D, H),
            tensor(T, H),
            7,
        ),
        (
            "run_cooperative_dv_b16_tg_b16_f32",
            tensor(D, B),
            tensor(T, B),
            7,
        ),
        (
            "run_cooperative_tg_f32_tg_f32_f32",
            tensor(T, F),
            tensor(T, F),
            7,
        ),
        (
            "run_cooperative_tg_f16_dv_f16_f32",
            tensor(T, H),
            tensor(D, H),
            7,
        ),
        ("run_cooperative_dv_f32_f32_f32", tensor(D, F), coop(F), 6),
        ("run_cooperative_dv_f16_f16_f32", tensor(D, H), coop(H), 6),
        ("run_cooperative_dv_b16_b16_f32", tensor(D, B), coop(B), 6),
    ];
    for (name, left, right, arity) in cases {
        let f = of(&format!("__tensorops_impl_matmul2d_op_{name}")).unwrap();
        let MatMul2dFn::Run(run) = f else {
            panic!("{name}: {f:?}");
        };
        assert_eq!(
            run,
            RunForm {
                single_thread: false,
                left,
                right,
                destination: coop(F),
                v2: false,
            },
            "{name}"
        );
        assert_eq!(f.arity(), Some(arity), "{name}");
        assert!(!f.descriptor_by_value());
    }
}

#[test]
fn run_forms_and_their_argument_positions() {
    use DataType::{Float32 as F, Int32, Int8};
    use TensorSpace::{Device as D, Thread};
    let run = |name: &str| match of(&format!("__tensorops_impl_matmul2d_op_run_{name}")) {
        Some(MatMul2dFn::Run(r)) => r,
        other => panic!("{name}: {other:?}"),
    };
    // Three tensors: descriptor, then (pointer, type) per operand, then the thread count.
    let r = run("dv_i8_dv_i8_dv_i32");
    assert_eq!(
        (r.left, r.right, r.destination),
        (tensor(D, Int8), tensor(D, Int8), tensor(D, Int32))
    );
    assert_eq!((r.arity(), r.operand_args()), (8, [1, 3, 5]));
    // `_v2`: each input tensor adds its scale type, two scale block sizes, and two static
    // extents; a destination tensor its two static extents.
    let r = run("dv_f32_dv_f32_dv_f32_v2");
    assert!(r.v2);
    assert_eq!((r.arity(), r.operand_args()), (20, [1, 8, 15]));
    // The single-thread scope passes no thread count.
    let r = run("single_thread_th_f32_dv_f32_dv_f32_v2");
    assert!(r.single_thread);
    assert_eq!(r.left, tensor(Thread, F));
    assert_eq!(r.arity(), 19);
    let r = run("single_thread_dv_f32_dv_f32_dv_f32");
    assert_eq!(r.arity(), 7);
    // `cooperative_` marks any cooperative operand, a destination tensor included.
    let r = run("cooperative_f32_f32_dv_f32");
    assert_eq!(
        (r.left, r.right, r.destination),
        (coop(F), coop(F), tensor(D, F))
    );
    assert_eq!((r.arity(), r.operand_args()), (6, [1, 2, 3]));
    let r = run("cooperative_dv_f32_f32_f32_v2");
    assert_eq!((r.arity(), r.operand_args()), (11, [1, 8, 9]));
    // Ill-formed spellings are the family, unmodeled.
    for bad in [
        "run_dv_f32_dv_f32_f32",
        "run_cooperative_dv_f32_dv_f32_dv_f32",
        "run_single_thread_f32_f32_f32",
        "run_dv_f32_dv_f32",
        "run_dv_f32_dv_f32_dv_f32_dv_f32",
        "run_xx_f32_dv_f32_dv_f32",
        "run_dv_f64_dv_f32_dv_f32",
    ] {
        assert_eq!(
            of(&format!("__tensorops_impl_matmul2d_op_{bad}")),
            Some(MatMul2dFn::Unmodeled),
            "{bad}"
        );
    }
}

#[test]
fn the_helpers_take_the_descriptor_by_value_with_the_header_arities() {
    for (name, f, arity) in [
        (
            "cooperative_tensor_data_size",
            MatMul2dFn::TensorDataSize,
            12,
        ),
        (
            "cooperative_tensor_num_elements",
            MatMul2dFn::TensorNumElements,
            12,
        ),
        (
            "cooperative_tensor_get_element_pointer",
            MatMul2dFn::TensorGetElementPointer,
            13,
        ),
        (
            "cooperative_tensor_get_element_index",
            MatMul2dFn::TensorGetElementIndex,
            13,
        ),
        (
            "cooperative_tensor_get_coordinate",
            MatMul2dFn::TensorGetCoordinate,
            16,
        ),
        ("cooperative_tensor_init", MatMul2dFn::TensorInit, 13),
        (
            "cooperative_tensor_is_valid_element",
            MatMul2dFn::TensorIsValidElement,
            14,
        ),
    ] {
        let got = of(&format!("__tensorops_impl_matmul2d_op_{name}"));
        assert_eq!(got, Some(f), "{name}");
        assert_eq!(f.arity(), Some(arity), "{name}");
        assert!(f.descriptor_by_value(), "{name}");
    }
    assert_eq!(MatMul2dFn::Unmodeled.arity(), None);
}

#[test]
fn data_types_round_trip_their_header_values() {
    for d in DataType::ALL {
        assert_eq!(DataType::from_raw(u64::from(d.bits())), Some(d));
    }
    // The values the corpus passes: float32, half, bfloat, and the int32 coordinate type.
    assert_eq!(DataType::from_raw(268_435_488), Some(DataType::Float32));
    assert_eq!(DataType::from_raw(0x1000_0010), Some(DataType::Float16));
    assert_eq!(DataType::from_raw(0x9000_0010), Some(DataType::BFloat16));
    assert_eq!(DataType::from_raw(536_870_944), Some(DataType::Int32));
    assert_eq!(DataType::from_raw(0x1000_0040), None);
    assert!(DataType::Int8.is_integer() && DataType::UInt32.is_integer());
    assert!(!DataType::BFloat16.is_integer() && !DataType::Fp8E5M2.is_integer());
    assert_eq!(DataType::from_token("b16"), Some(DataType::BFloat16));
    assert_eq!(DataType::from_token("bf16"), None);
}

#[test]
fn descriptors_decode_from_fields_and_bytes() {
    let d = Descriptor::from_fields([16, 16, 32, 0, 1, 0, 1]).unwrap();
    assert_eq!(
        d,
        Descriptor {
            m: 16,
            n: 16,
            k: Inner::Static(32),
            transpose_left: false,
            transpose_right: true,
            relaxed_precision: false,
            mode: Mode::MultiplyAccumulate,
        }
    );
    assert_eq!(
        d.to_string(),
        "16x16x32 transpose_right multiply_accumulate"
    );
    // `dynamic_extent` as an `int` and `dynamic_length_v<int>`.
    for k in [0xFFFF_FFFF, 0x7FFF_FFFF] {
        assert_eq!(
            Descriptor::from_fields([64, 32, k, 0, 0, 0, 0]).map(|d| d.k),
            Some(Inner::Dynamic)
        );
    }
    for bad in [
        [0, 16, 16, 0, 0, 0, 0],
        [16, 0, 16, 0, 0, 0, 0],
        [16, 16, 0, 0, 0, 0, 0],
        [16, 16, 16, 2, 0, 0, 0],
        [16, 16, 16, 0, 0, 0, 2],
        [0x8000_0000, 16, 16, 0, 0, 0, 0],
    ] {
        assert_eq!(Descriptor::from_fields(bad), None, "{bad:?}");
    }
    let mut bytes = [0u8; DESCRIPTOR_BYTES];
    bytes[0..4].copy_from_slice(&64u32.to_le_bytes());
    bytes[4..8].copy_from_slice(&32u32.to_le_bytes());
    bytes[8..12].copy_from_slice(&16u32.to_le_bytes());
    bytes[12] = 1;
    bytes[15] = 0xAB;
    bytes[16..20].copy_from_slice(&0u32.to_le_bytes());
    let d = Descriptor::from_bytes(&bytes).unwrap();
    assert_eq!((d.m, d.n, d.k), (64, 32, Inner::Static(16)));
    assert!(d.transpose_left && !d.transpose_right && d.mode == Mode::Multiply);
    assert_eq!(Descriptor::from_bytes(&bytes[..19]), None);
}

#[test]
fn indices_and_descriptor_types_decode() {
    assert_eq!(OperandIndex::from_raw(2), Some(OperandIndex::Destination));
    assert_eq!(OperandIndex::from_raw(3), None);
    assert_eq!(DescriptorType::from_raw(2), Some(DescriptorType::Inline));
    assert_eq!(DescriptorType::from_raw(4), None);
}
