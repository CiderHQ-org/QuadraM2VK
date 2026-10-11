//! Native resource IDs paired with the shared descriptor policy header: the table a guarded
//! producer looks its command buffer up in. Copied from the product with the guard.

use metal2vulkan::bind::indirect_commands::{
    CommandBufferError, CommandDescriptor, Inheritance, COMMAND_BUFFER_HEADER_BYTES,
};
use std::num::NonZeroU64;

/// One native resource registration. Descriptor policy is checked before encoding.
pub struct NativeCommandPolicy {
    resource_id: NonZeroU64,
    header: [u8; COMMAND_BUFFER_HEADER_BYTES],
}

impl NativeCommandPolicy {
    pub fn new(
        resource_id: NonZeroU64,
        descriptor: &CommandDescriptor,
        capacity: u32,
        inheritance: Inheritance,
    ) -> Result<Self, CommandBufferError> {
        Ok(Self {
            resource_id,
            header: descriptor.encode_header(capacity, inheritance)?,
        })
    }
}

/// Immutable initialization and expected readback for a native guard table.
///
/// The first eight bytes contain a sticky fault word and the registration count.
/// Sorted records contain an eight-byte resource ID followed by the shared command
/// policy header. This allocation must remain live and immutable except for fault
/// words until producer completion. Constructing it does not admit GPU execution.
pub struct NativeCommandPolicies {
    bytes: Vec<u8>,
}

impl NativeCommandPolicies {
    const HEADER_BYTES: usize = 8;
    const RECORD_BYTES: usize = 8 + COMMAND_BUFFER_HEADER_BYTES;

    pub fn new(mut policies: Vec<NativeCommandPolicy>) -> Result<Self, CommandBufferError> {
        let count = u32::try_from(policies.len()).map_err(|_| CommandBufferError::Length)?;
        policies.sort_unstable_by_key(|policy| policy.resource_id);
        if policies
            .windows(2)
            .any(|pair| pair[0].resource_id == pair[1].resource_id)
        {
            return Err(CommandBufferError::Descriptor(
                "duplicate native resource ID".into(),
            ));
        }
        let size = policies
            .len()
            .checked_mul(Self::RECORD_BYTES)
            .and_then(|size| size.checked_add(Self::HEADER_BYTES))
            .ok_or(CommandBufferError::Length)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(size)
            .map_err(|_| CommandBufferError::Length)?;
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&count.to_le_bytes());
        for policy in policies {
            bytes.extend_from_slice(&policy.resource_id.get().to_le_bytes());
            bytes.extend_from_slice(&policy.header);
        }
        Ok(Self { bytes })
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn registration_count(&self) -> u32 {
        ((self.bytes.len() - Self::HEADER_BYTES) / Self::RECORD_BYTES) as u32
    }

    pub(super) fn lookup_source(name: &str, count: u32) -> String {
        let header = Self::HEADER_BYTES;
        let stride = Self::RECORD_BYTES;
        format!(
            r#"define internal i32 addrspace(1)* @{name}(i32 addrspace(1)* %table, i64 %key) alwaysinline nounwind {{
entry:
  %count_ptr = getelementptr i32, i32 addrspace(1)* %table, i64 1
  %count = load i32, i32 addrspace(1)* %count_ptr, align 4
  %valid_count = icmp eq i32 %count, {count}
  br i1 %valid_count, label %start, label %fault
start:
  %bytes = bitcast i32 addrspace(1)* %table to i8 addrspace(1)*
  br label %search
search:
  %low = phi i32 [ 0, %start ], [ %next_low, %advance ]
  %high = phi i32 [ {count}, %start ], [ %next_high, %advance ]
  %nonempty = icmp ult i32 %low, %high
  br i1 %nonempty, label %probe, label %fault
probe:
  %distance = sub i32 %high, %low
  %half = lshr i32 %distance, 1
  %middle = add i32 %low, %half
  %wide = zext i32 %middle to i64
  %scaled = mul i64 %wide, {stride}
  %offset = add i64 %scaled, {header}
  %record = getelementptr i8, i8 addrspace(1)* %bytes, i64 %offset
  %id_ptr = bitcast i8 addrspace(1)* %record to i64 addrspace(1)*
  %id = load i64, i64 addrspace(1)* %id_ptr, align 8
  %equal = icmp eq i64 %id, %key
  br i1 %equal, label %found, label %advance
advance:
  %less = icmp ult i64 %id, %key
  %after = add i32 %middle, 1
  %next_low = select i1 %less, i32 %after, i32 %low
  %next_high = select i1 %less, i32 %high, i32 %middle
  br label %search
found:
  %policy_bytes = getelementptr i8, i8 addrspace(1)* %record, i64 8
  %policy = bitcast i8 addrspace(1)* %policy_bytes to i32 addrspace(1)*
  ret i32 addrspace(1)* %policy
fault:
  call void @air.atomic.global.store.i32(i32 addrspace(1)* %table, i32 1, i32 0, i32 2, i1 true)
  ret i32 addrspace(1)* null
}}
"#
        )
    }

    /// Reject lookup faults, per-resource encoding faults, and changes to immutable policy.
    pub fn validate_readback(&self, bytes: &[u8]) -> Result<(), CommandBufferError> {
        if bytes.len() != self.bytes.len() {
            return Err(CommandBufferError::Length);
        }
        for offset in std::iter::once(0)
            .chain((Self::HEADER_BYTES + 8 + 4..bytes.len()).step_by(Self::RECORD_BYTES))
        {
            let fault = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
            if fault != 0 {
                return Err(CommandBufferError::EncodingFault(fault));
            }
        }
        if bytes != self.bytes {
            return Err(CommandBufferError::HeaderPolicyMismatch);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use metal2vulkan::bind::indirect_commands::{ComputeDispatch, StateSource};

    fn policy(id: u64) -> NativeCommandPolicy {
        NativeCommandPolicy::new(
            NonZeroU64::new(id).unwrap(),
            &CommandDescriptor::Compute {
                dispatch: ComputeDispatch::Both,
                max_buffer_bind_count: 0,
                max_threadgroup_memory_bind_count: 0,
                support_dynamic_attribute_stride: false,
            },
            1,
            Inheritance {
                pipeline: StateSource::Command,
                buffers: StateSource::Command,
            },
        )
        .unwrap()
    }

    #[test]
    fn registrations_are_unique_and_lookup_is_order_independent() {
        assert!(NativeCommandPolicies::new(vec![policy(3), policy(3)]).is_err());
        let table = NativeCommandPolicies::new(vec![policy(9), policy(3)]).unwrap();
        assert_eq!(table.registration_count(), 2);
        assert_eq!(
            u64::from_le_bytes(table.bytes()[8..16].try_into().unwrap()),
            3
        );
        assert_eq!(
            u64::from_le_bytes(table.bytes()[32..40].try_into().unwrap()),
            9
        );
        table.validate_readback(table.bytes()).unwrap();
        let empty = NativeCommandPolicies::new(vec![]).unwrap();
        empty.validate_readback(empty.bytes()).unwrap();
        assert_eq!(empty.registration_count(), 0);
    }

    #[test]
    fn generated_lookup_executes_hits_misses_and_count_mismatch_on_cpu() {
        use std::{fmt::Write, process::Command};
        match Command::new("clang").arg("--version").output() {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                eprintln!("skipping native lookup execution: clang is unavailable");
                return;
            }
            result => assert!(result.unwrap().status.success()),
        }
        let scratch = crate::ScratchDir::new("native-command-lookup").unwrap();
        let mut llvm = String::from("define void @air.atomic.global.store.i32(i32* %p, i32 %v, i32 %order, i32 %scope, i1 %flag) {\n store atomic i32 %v, i32* %p monotonic, align 4\n ret void\n}\n");
        let mut declarations = String::from("#include <stdint.h>\n#include <string.h>\n");
        let mut main = String::from("int main(void) {\n");
        for count in [0, 1, 7, 32] {
            let table = NativeCommandPolicies::new(
                (1..=count)
                    .rev()
                    .map(|id| policy(u64::from(id) * 2))
                    .collect(),
            )
            .unwrap();
            let name = format!("lookup_{count}");
            llvm.push_str(
                &NativeCommandPolicies::lookup_source(&name, count)
                    .replace(" addrspace(1)", "")
                    .replace("define internal", "define"),
            );
            writeln!(
                declarations,
                "extern uint32_t* {name}(uint32_t*, uint64_t);"
            )
            .unwrap();
            let bytes = table
                .bytes()
                .iter()
                .map(u8::to_string)
                .collect::<Vec<_>>()
                .join(",");
            writeln!(main, "{{ uint64_t storage[{}]; uint8_t initial[] = {{{bytes}}}; uint32_t* words = (uint32_t*)storage;", table.bytes().len() / 8).unwrap();
            writeln!(main, "for (uint64_t key = 0; key <= {}; ++key) {{ memcpy(storage, initial, sizeof initial); uint32_t* result = {name}(words, key);", count * 2 + 2).unwrap();
            writeln!(main, "if (key && key <= {} && key % 2 == 0) {{ if ((uint8_t*)result != (uint8_t*)storage + 16 + (key / 2 - 1) * 24 || words[0]) return 1; }} else if (result || words[0] != 1) return 2; }}", count * 2).unwrap();
            writeln!(main, "memcpy(storage, initial, sizeof initial); words[1] = {}; if ({name}(words, 2) || words[0] != 1) return 3; }}", count + 1).unwrap();
        }
        main.push_str("return 0; }\n");
        declarations.push_str(&main);
        let ir = scratch.path().join("lookup.ll");
        let c = scratch.path().join("harness.c");
        let executable = scratch.path().join("lookup");
        std::fs::write(&ir, llvm).unwrap();
        std::fs::write(&c, declarations).unwrap();
        let output = Command::new("clang")
            .args(["-Wno-override-module", "-O1"])
            .arg(ir)
            .arg(c)
            .arg("-o")
            .arg(&executable)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(Command::new(executable).status().unwrap().success());
    }

    #[test]
    fn every_changed_readback_byte_is_rejected() {
        let table = NativeCommandPolicies::new(vec![policy(9), policy(3)]).unwrap();
        for offset in 0..table.bytes().len() {
            let mut bytes = table.bytes().to_vec();
            bytes[offset] ^= 1;
            assert!(table.validate_readback(&bytes).is_err(), "byte {offset}");
        }
        assert_eq!(
            table.validate_readback(&table.bytes()[1..]),
            Err(CommandBufferError::Length)
        );
    }
}
