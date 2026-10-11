//! The AIR identity sanitizer: the text every source and module row is stored as, and so the bytes
//! its `air_sha256` or `module_sha256` hashes. Copied from the product's tool runner unchanged,
//! less the datalayout it also returned: an edit here re-keys every stored row, so it is a corpus
//! migration, never a refactor.

/// The target triple the sanitizer writes in place of AIR's.
const SANITIZED_TRIPLE: &str = "spirv-unknown-vulkan1.2";

/// `ll_text` as validation stores it: the target triple rewritten to `spirv-unknown-vulkan1.2`,
/// the AIR datalayout kept, llvm-dis's input-path `ModuleID` comment dropped, and the LLVM metadata
/// and runtime root globals that are not part of the shader entry body (`@llvm.global_ctors`,
/// `@llvm.global_dtors`, `@llvm.used`, `@llvm.compiler.used`) removed. Every kept line ends in one
/// newline.
pub fn sanitize_ll_text(ll_text: &str) -> String {
    let mut out = String::with_capacity(ll_text.len());
    for line in ll_text.lines() {
        let t = line.trim_start();
        if t.starts_with("target triple") {
            out.push_str(&format!("target triple = \"{SANITIZED_TRIPLE}\"\n"));
            continue;
        }
        // llvm-dis derives this comment from its input filename. Validation disassembles through
        // a uniquely named scratch directory, so retaining it makes identical bitcode acquire a
        // different corpus identity on every harvest.
        if t.starts_with("; ModuleID =") {
            continue;
        }
        if t.starts_with("@llvm.global_ctors")
            || t.starts_with("@llvm.global_dtors")
            || t.starts_with("@llvm.used")
            || t.starts_with("@llvm.compiler.used")
        {
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }
    out
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    #[test]
    fn the_sanitizer_rewrites_the_triple_and_drops_the_path_comment_and_root_globals_by_prefix() {
        let ll = "; ModuleID = '/tmp/scratch-91/k.air'\r\nsource_filename = \"k.metal\"\n\
                  target datalayout = \"e-p:64:64:64\"\n\
                  \x20 target triple = \"air64_v28-apple-macosx26.0.0\"\n\
                  @llvm.used = appending global [1 x ptr] [ptr @k], section \"llvm.metadata\"\n\
                  @llvm.compiler.used = appending global [0 x ptr] zeroinitializer\n\
                  @llvm.global_ctors = appending global [0 x { i32, ptr, ptr }] zeroinitializer\n\
                  @llvm.global_dtors = appending global [0 x { i32, ptr, ptr }] zeroinitializer\n\
                  @llvm.used.by.prefix = global i32 0\n\
                  define void @k() {\n  ret void\n}";
        assert_eq!(
            sanitize_ll_text(ll),
            "source_filename = \"k.metal\"\ntarget datalayout = \"e-p:64:64:64\"\n\
             target triple = \"spirv-unknown-vulkan1.2\"\ndefine void @k() {\n  ret void\n}\n",
            "a root global is dropped by its name's prefix, as every stored identity was sanitized"
        );
    }

    #[test]
    fn sanitized_text_is_a_fixed_point() {
        let ll = "; ModuleID = 'a'\ntarget triple = \"air64\"\ndefine void @k() {\n  ret void\n}\n";
        let once = sanitize_ll_text(ll);
        assert_eq!(sanitize_ll_text(&once), once);
    }
}
