//! Over random pipelines: random compute and vertex+fragment pipelines of builder modules (random
//! descriptor sets, descriptor arrays, specialization constants, and matched and unmatched
//! varyings, packed under tight limits) link; relocation writes only the reserved literals, so
//! every stage's words with the relocation slots zeroed equal its serialized words whatever the
//! pipeline; the relocated words pass `spirv-val`; and the table is consistent with the layout and
//! between stages.

use super::*;
use crate::pipeline::testing::zero_relocations;
use crate::spirv::testing::{spirv_val_with, TargetEnv, ValOptions};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// The varying types the generator picks from (32-bit `float`/`uint`, 1–4 lanes).
const VTS: [(IoNumeric, u8); 6] = [
    (IoNumeric::Float, 1),
    (IoNumeric::Float, 2),
    (IoNumeric::Float, 3),
    (IoNumeric::Float, 4),
    (IoNumeric::Uint, 1),
    (IoNumeric::Uint, 3),
];

/// Random resources of `stage`.
fn resources(b: &mut ModuleBuilder, rng: &mut Rng, stage: Stage) {
    let mut buffers = BTreeSet::new();
    for _ in 0..rng.below(6) {
        buffers.insert(rng.below(31) as u32);
    }
    for i in buffers {
        ssbo(b, res(stage, ResourceKind::Buffer, i));
    }
    if rng.below(2) == 0 {
        let arg = ResourceKind::Synthetic(SyntheticBinding::ArgumentTable);
        ubo(b, res(stage, arg, 0), 1 + rng.below(4));
    }
    if rng.below(2) == 0 {
        let status = ResourceKind::Synthetic(SyntheticBinding::ExecutionStatus);
        ssbo(b, res(stage, status, 0));
    }
    let sampled = image(b, Dim::D2, ImageUsage::Sampled);
    let mut textures = BTreeSet::new();
    for _ in 0..rng.below(4) {
        textures.insert(rng.below(128) as u32);
    }
    for i in textures {
        opaque(
            b,
            res(stage, ResourceKind::Texture, i),
            sampled,
            1 + rng.below(3),
        );
    }
    let sampler = b.ty(Type::Sampler).expect("sampler");
    for i in 0..rng.below(3) as u32 {
        opaque(b, res(stage, ResourceKind::Sampler, i), sampler, 1);
    }
    let u = u32_ty(b);
    for i in 0..rng.below(3) as u32 {
        b.spec(SpecKey::TgLength(i), u, ConstBits(u64::from(i) * 16))
            .expect("spec");
    }
}

fn kernel(rng: &mut Rng) -> FunctionArtifact {
    let mut b = ModuleBuilder::new();
    resources(&mut b, rng, Stage::Kernel);
    from_module(
        Stage::Kernel,
        finish(b, ExecModel::GLCompute),
        StageIo::default(),
    )
}

/// A vertex stage with `outputs`, and a fragment stage reading `inputs` (a subset).
fn graphics(
    rng: &mut Rng,
    outputs: &[(String, (IoNumeric, u8))],
    inputs: &[usize],
) -> [FunctionArtifact; 2] {
    let mut stages = Vec::new();
    for (stage, model) in [
        (Stage::Vertex, ExecModel::Vertex),
        (Stage::Fragment, ExecModel::Fragment),
    ] {
        let mut io = StageIo::default();
        for (i, (name, (numeric, n))) in outputs.iter().enumerate() {
            let ty = io_ty(*numeric, *n);
            if stage == Stage::Vertex {
                io.outputs.push(varying(name, ty, None));
            } else if inputs.contains(&i) {
                let interp = match numeric {
                    IoNumeric::Float => SMOOTH,
                    _ => Some(Interpolation::Flat),
                };
                io.inputs.push(varying(name, ty, interp));
            }
        }
        let mut b = ModuleBuilder::new();
        resources(&mut b, rng, stage);
        let u = u32_ty(&mut b);
        b.spec(SpecKey::SampleCount, u, ConstBits(1)).expect("spec");
        declare_io(&mut b, &io);
        stages.push(from_module(stage, finish(b, model), io));
    }
    let f = stages.pop().expect("fragment");
    let v = stages.pop().expect("vertex");
    [v, f]
}

fn check_linked(l: &Linked, stages: &[&FunctionArtifact], what: &str) {
    let set = &l.layout.sets[0];
    for (n, b) in set.bindings.iter().enumerate() {
        assert_eq!(b.binding, n as u32, "{what}: dense");
    }
    for a in stages {
        for d in &a.interface.descriptors {
            let b = &set.bindings[l.table.bindings[&d.slot].1 as usize];
            assert_eq!((b.ty, b.count), (d.ty, d.count), "{what}");
        }
        // No two variables of one direction of a stage overlap in a location's components.
        for vars in [&a.interface.io.inputs, &a.interface.io.outputs] {
            let mut used = BTreeSet::new();
            for v in vars {
                let (at, c) = l.table.locations[&v.key];
                for k in c..c + u32::from(v.ty.components) {
                    assert!(k < 4, "{what}: component overflow");
                    assert!(used.insert((at, k)), "{what}: overlap at {at}.{k}");
                }
            }
        }
        let words = crate::pipeline::relocate::relocate(a, l).expect("relocate");
        assert_eq!(
            zero_relocations(a, &words),
            a.module.words,
            "{what}: link changed a non-relocation word"
        );
        let opts = ValOptions::for_requirements(&a.requirements);
        if let Err(e) = spirv_val_with(&words, TargetEnv::Vulkan1_2, opts) {
            panic!("{what}: spirv-val: {e}");
        }
    }
}

#[test]
fn random_pipelines_link_and_relocate_only_their_literals() {
    let p = profile("moltenvk-m3");
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    for round in 0..48 {
        let what = format!("round {round}");
        if round % 3 == 0 {
            let k = kernel(&mut rng);
            let l = link(&[&k], &LinkInputs::compute(), &p).expect("link compute");
            check_linked(&l, &[&k], &what);
            continue;
        }
        let n = rng.below(7) as usize;
        let outputs: Vec<(String, (IoNumeric, u8))> = (0..n)
            .map(|i| (format!("v{i}"), VTS[rng.below(VTS.len() as u64) as usize]))
            .collect();
        let inputs: Vec<usize> = (0..n).filter(|_| rng.below(2) == 0).collect();
        let [v, f] = graphics(&mut rng, &outputs, &inputs);
        // Every other round, limits tight enough that some pipelines need packing.
        let mut q = p.clone();
        if round % 2 == 0 {
            q.fields.limits.max_vertex_output_components = 16;
            q.fields.limits.max_fragment_input_components = 8;
        }
        let l = match link(&[&v, &f], &render(), &q) {
            Ok(l) => l,
            Err(Error::Unsupported(u)) if round % 2 == 0 => {
                assert_eq!(u.feature, Feature::InterfaceComponents, "{what}");
                continue;
            }
            Err(e) => panic!("{what}: {e}"),
        };
        check_linked(&l, &[&v, &f], &what);
        // Each fragment input has its vertex output's location.
        for x in &f.interface.io.inputs {
            assert!(
                v.interface.io.outputs.iter().any(|o| o.key == x.key),
                "{what}: fragment input without a producer"
            );
        }
        // The same vertex stage in another pipeline (alone): same zeroed words.
        let alone = link(&[&v], &LinkInputs::standalone(), &p).expect("alone");
        check_linked(&alone, &[&v], &what);
    }
}
