//! The validation files a set of classifier inputs depends on, resolved from their syntax.
//!
//! The analyzer key fingerprints the classifier's own files (`METAL2VULKAN_CLASSIFIER_INPUTS`,
//! `validation/build.rs`). This resolver holds that list closed: every module file an input's
//! non-test code can reach by a path (`crate::`, `super::`, `self::`, or a child module's name, in
//! `use` trees including groups and renames, in expressions and types, and in macro tokens) or
//! declares (`mod x;`) resolves to the longest existing file of the path, which must itself be an
//! input or be listed with why classification does not depend on it. Items under `#[cfg(test)]`
//! are skipped: tests do not change what classification computes. The product's runtime guard
//! holds its runtime inputs closed over the product's module files the same way
//! (`candidate::engine::artifact`), across the workspace's crates: an input below another
//! crate's directory (`metal2vulkan-air/src/...`) is that crate's module file, a path through that
//! crate's name (`metal2vulkan_air::...`) reaches its files, and so does one through a name a crate
//! root imports from it (`use metal2vulkan_air as air;`, then `crate::air::...`).

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use syn::visit::Visit;

/// A crate of the workspace: the name its dependents write (`metal2vulkan_air`) and its directory
/// below the root (`metal2vulkan-air`), whose `src/` holds its modules.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Member {
    pub(crate) name: &'static str,
    pub(crate) dir: &'static str,
}

impl Member {
    /// The directory of the member's modules: `<dir>/src/`.
    fn source(&self) -> String {
        format!("{}/src/", self.dir)
    }
}

/// Module files the inputs reach, relative to `root` (`src/...` for the crate at the root,
/// `<dir>/src/...` for a [`Member`]'s), with the inputs that reach each; or the spellings the
/// resolver refuses.
pub(crate) fn dependencies(
    root: &Path,
    members: &[Member],
    inputs: &[&str],
) -> Result<BTreeMap<String, BTreeSet<String>>, Vec<String>> {
    let mut found = BTreeMap::<String, BTreeSet<String>>::new();
    let mut errors = Vec::new();
    let mut crates = Crates {
        root,
        members,
        imports: BTreeMap::new(),
    };
    for input in inputs.iter().filter(|input| input.ends_with(".rs")) {
        let text = std::fs::read_to_string(root.join(input))
            .map_err(|error| vec![format!("{input}: {error}")])?;
        let file = syn::parse_file(&text).map_err(|error| vec![format!("{input}: {error}")])?;
        let Some((source, module)) = module_path(members, input) else {
            errors.push(format!("{input}: not a module file under a crate's src/"));
            continue;
        };
        let mut walker = Walker {
            crates: &mut crates,
            input,
            scopes: vec![Scope {
                source,
                module,
                children: child_modules(&file.items),
            }],
            found: BTreeSet::new(),
            errors: Vec::new(),
        };
        walker.visit_file(&file);
        for dependency in walker.found {
            found
                .entry(dependency)
                .or_default()
                .insert((*input).to_string());
        }
        errors.extend(walker.errors);
    }
    if errors.is_empty() {
        Ok(found)
    } else {
        Err(errors)
    }
}

/// Whether `relative` names a file below `root` with every component spelt as on disk: a
/// case-insensitive file system (APFS) finds `air/src/ir/Value.rs` for a type `Value` beside the
/// module file `value.rs`, which a module path never names.
fn is_module_file(root: &Path, relative: &str) -> bool {
    let mut directory = root.to_path_buf();
    for component in relative.split('/') {
        let listed = std::fs::read_dir(&directory).is_ok_and(|entries| {
            entries
                .flatten()
                .any(|entry| entry.file_name().to_str() == Some(component))
        });
        if !listed {
            return false;
        }
        directory.push(component);
    }
    directory.is_file()
}

/// `src/a/b.rs` and `src/a/b/mod.rs` are module `a::b` of the crate at the root, and
/// `<dir>/src/a/b.rs` a [`Member`]'s, whose source directory comes first; `src/lib.rs` is the
/// crate root.
fn module_path(members: &[Member], input: &str) -> Option<(String, Vec<String>)> {
    let (source, path) = members
        .iter()
        .find_map(|member| {
            let source = member.source();
            let path = input.strip_prefix(&source)?;
            Some((source, path))
        })
        .or_else(|| Some(("src/".to_string(), input.strip_prefix("src/")?)))?;
    let path = path.strip_suffix(".rs")?;
    let mut segments = path.split('/').map(str::to_string).collect::<Vec<_>>();
    if segments.last().is_some_and(|last| last == "mod") || path == "lib" {
        segments.pop();
    }
    Some((source, segments))
}

fn child_modules(items: &[syn::Item]) -> BTreeSet<String> {
    items
        .iter()
        .filter_map(|item| match item {
            syn::Item::Mod(module) if !is_test_only(&module.attrs) => {
                Some(module.ident.to_string())
            }
            _ => None,
        })
        .collect()
}

fn is_test_only(attrs: &[syn::Attribute]) -> bool {
    attrs.iter().any(|attr| {
        attr.path().is_ident("cfg")
            && attr
                .parse_args::<syn::Ident>()
                .is_ok_and(|ident| ident == "test")
    })
}

/// The crates the walk resolves into, and the names each crate root imports from a [`Member`].
struct Crates<'a> {
    root: &'a Path,
    members: &'a [Member],
    /// By a crate's source directory (`src/`, `metal2vulkan-air/src/`), its root's imports of a member: each
    /// imported name with the member's source directory and the module path it names there.
    imports: BTreeMap<String, Imports>,
}

/// A crate root's imports of members' modules and items: `use metal2vulkan_air as air;` names
/// `air` for the member's root, `use metal2vulkan_air::budget;` names `budget` for its module.
type Imports = BTreeMap<String, (String, Vec<String>)>;

impl Crates<'_> {
    /// The source directory of the member named `name`.
    fn member_source(&self, name: &str) -> Option<String> {
        self.members
            .iter()
            .find(|member| member.name == name)
            .map(Member::source)
    }

    /// The imports of the crate root below `source`, read once; a root that cannot be read or
    /// parsed imports nothing here, and its own walk (as an input or a dependency) reports it.
    fn imports(&mut self, source: &str) -> &Imports {
        if !self.imports.contains_key(source) {
            let mut imports = Imports::new();
            let root = std::fs::read_to_string(self.root.join(format!("{source}lib.rs")))
                .ok()
                .and_then(|text| syn::parse_file(&text).ok());
            for item in root.iter().flat_map(|file| &file.items) {
                if let syn::Item::Use(item) = item {
                    if !is_test_only(&item.attrs) && item.leading_colon.is_none() {
                        self.member_imports(&item.tree, &mut Vec::new(), &mut imports);
                    }
                }
            }
            self.imports.insert(source.to_string(), imports);
        }
        &self.imports[source]
    }

    /// Record `prefix` under `name` when it starts at a member's name.
    fn import(&self, prefix: &[String], name: &syn::Ident, out: &mut Imports) {
        if let Some((first, rest)) = prefix.split_first() {
            if let Some(source) = self.member_source(first) {
                out.insert(name.to_string(), (source, rest.to_vec()));
            }
        }
    }

    fn member_imports(&self, tree: &syn::UseTree, prefix: &mut Vec<String>, out: &mut Imports) {
        match tree {
            syn::UseTree::Path(path) => {
                prefix.push(path.ident.to_string());
                self.member_imports(&path.tree, prefix, out);
                prefix.pop();
            }
            syn::UseTree::Name(name) => {
                prefix.push(name.ident.to_string());
                self.import(prefix, &name.ident, out);
                prefix.pop();
            }
            syn::UseTree::Rename(rename) => {
                prefix.push(rename.ident.to_string());
                self.import(prefix, &rename.rename, out);
                prefix.pop();
            }
            syn::UseTree::Glob(_) => {}
            syn::UseTree::Group(group) => {
                for tree in &group.items {
                    self.member_imports(tree, prefix, out);
                }
            }
        }
    }
}

struct Scope {
    /// The source directory of the crate this scope is in (`src/`, `air/src/`).
    source: String,
    module: Vec<String>,
    children: BTreeSet<String>,
}

struct Walker<'a, 'c> {
    crates: &'a mut Crates<'c>,
    input: &'a str,
    scopes: Vec<Scope>,
    found: BTreeSet<String>,
    errors: Vec<String>,
}

impl Walker<'_, '_> {
    fn scope(&self) -> &Scope {
        self.scopes.last().expect("the file scope")
    }

    /// A path as written, from its first segment: absolute within the crate when it starts at
    /// `crate`, `super`, `self`, or a child module of the current scope, or (in a crate root's
    /// own scope) a name the root imports from a member; within a member when it starts at the
    /// member's name; otherwise another crate, a prelude or local item, or a generic, none of
    /// which is a file of the workspace.
    fn reference(&mut self, segments: &[String]) {
        let scope = self.scope();
        let mut source = scope.source.clone();
        let mut module = scope.module.clone();
        let child = segments
            .first()
            .is_some_and(|first| scope.children.contains(first));
        let imported = module.is_empty()
            && segments
                .first()
                .is_some_and(|first| self.crates.imports(&source).contains_key(first));
        let rest = match segments.first().map(String::as_str) {
            Some("crate") => {
                module.clear();
                &segments[1..]
            }
            Some("self") => &segments[1..],
            Some("super") => {
                let supers = segments.iter().take_while(|segment| *segment == "super");
                let count = supers.count();
                if count > module.len() {
                    self.errors
                        .push(format!("{}: `super` above the crate root", self.input));
                    return;
                }
                module.truncate(module.len() - count);
                &segments[count..]
            }
            Some(_) if child || imported => segments,
            Some(first) => match self.crates.member_source(first) {
                Some(member) => {
                    source = member;
                    module.clear();
                    &segments[1..]
                }
                None => return,
            },
            None => return,
        };
        module.extend(rest.iter().cloned());
        // A crate-absolute path through a root's import continues in the member it names. Each
        // step moves to another crate's root import, so the walk is bounded by the members.
        for _ in 0..=self.crates.members.len() {
            let Some(first) = module.first() else {
                break;
            };
            let Some((target, path)) = self.crates.imports(&source).get(first).cloned() else {
                break;
            };
            source = target;
            module.splice(..1, path);
        }
        let file = self.longest_file(&source, &module);
        self.found.insert(file);
    }

    /// The longest prefix of `module` that is a file below the crate's `source`: `src/a/b.rs` or
    /// `src/a/b/mod.rs`, then `src/a.rs`, down to `src/lib.rs`.
    fn longest_file(&self, source: &str, module: &[String]) -> String {
        (1..=module.len())
            .rev()
            .find_map(|length| {
                let stem = module[..length].join("/");
                [
                    format!("{source}{stem}.rs"),
                    format!("{source}{stem}/mod.rs"),
                ]
                .into_iter()
                .find(|candidate| is_module_file(self.crates.root, candidate))
            })
            .unwrap_or_else(|| format!("{source}lib.rs"))
    }

    fn use_tree(&mut self, tree: &syn::UseTree, prefix: &mut Vec<String>) {
        match tree {
            syn::UseTree::Path(path) => {
                prefix.push(path.ident.to_string());
                self.use_tree(&path.tree, prefix);
                prefix.pop();
            }
            syn::UseTree::Name(name) => {
                prefix.push(name.ident.to_string());
                self.reference(prefix);
                prefix.pop();
            }
            syn::UseTree::Rename(rename) => {
                prefix.push(rename.ident.to_string());
                self.reference(prefix);
                prefix.pop();
            }
            syn::UseTree::Glob(_) => self.reference(prefix),
            syn::UseTree::Group(group) => {
                for tree in &group.items {
                    self.use_tree(tree, prefix);
                }
            }
        }
    }

    /// Paths inside macro invocations are unparsed tokens: find `crate`/`super`/`self` or a child
    /// module followed by `::` segments, at every nesting level.
    fn macro_tokens(&mut self, tokens: proc_macro2::TokenStream) {
        use proc_macro2::TokenTree;
        let tokens = tokens.into_iter().collect::<Vec<_>>();
        let mut index = 0;
        while index < tokens.len() {
            if let TokenTree::Group(group) = &tokens[index] {
                self.macro_tokens(group.stream());
            }
            let mut segments = Vec::new();
            let mut cursor = index;
            while let Some(TokenTree::Ident(ident)) = tokens.get(cursor) {
                segments.push(ident.to_string());
                let separator = matches!(
                    (tokens.get(cursor + 1), tokens.get(cursor + 2)),
                    (Some(TokenTree::Punct(a)), Some(TokenTree::Punct(b)))
                        if a.as_char() == ':' && b.as_char() == ':'
                );
                if !separator {
                    break;
                }
                cursor += 3;
            }
            if segments.len() > 1 {
                self.reference(&segments);
                index = cursor.max(index + 1);
            } else {
                index += 1;
            }
        }
    }
}

impl<'ast> Visit<'ast> for Walker<'_, '_> {
    fn visit_item(&mut self, item: &'ast syn::Item) {
        let attrs = match item {
            syn::Item::Const(item) => &item.attrs,
            syn::Item::Enum(item) => &item.attrs,
            syn::Item::ExternCrate(item) => &item.attrs,
            syn::Item::Fn(item) => &item.attrs,
            syn::Item::ForeignMod(item) => &item.attrs,
            syn::Item::Impl(item) => &item.attrs,
            syn::Item::Macro(item) => &item.attrs,
            syn::Item::Mod(item) => &item.attrs,
            syn::Item::Static(item) => &item.attrs,
            syn::Item::Struct(item) => &item.attrs,
            syn::Item::Trait(item) => &item.attrs,
            syn::Item::TraitAlias(item) => &item.attrs,
            syn::Item::Type(item) => &item.attrs,
            syn::Item::Union(item) => &item.attrs,
            syn::Item::Use(item) => &item.attrs,
            _ => {
                self.errors.push(format!(
                    "{}: an item form the resolver does not read",
                    self.input
                ));
                return;
            }
        };
        if !is_test_only(attrs) {
            syn::visit::visit_item(self, item);
        }
    }

    fn visit_item_mod(&mut self, item: &'ast syn::ItemMod) {
        if item.attrs.iter().any(|attr| attr.path().is_ident("path")) {
            self.errors.push(format!(
                "{}: `#[path]` module {} is not resolved; declare it at its default path",
                self.input, item.ident
            ));
            return;
        }
        let source = self.scope().source.clone();
        let mut module = self.scope().module.clone();
        module.push(item.ident.to_string());
        match &item.content {
            None => {
                let stem = module.join("/");
                let file = [
                    format!("{source}{stem}.rs"),
                    format!("{source}{stem}/mod.rs"),
                ]
                .into_iter()
                .find(|candidate| is_module_file(self.crates.root, candidate));
                match file {
                    Some(file) => {
                        self.found.insert(file);
                    }
                    None => self
                        .errors
                        .push(format!("{}: `mod {};` has no file", self.input, item.ident)),
                }
            }
            Some((_, items)) => {
                self.scopes.push(Scope {
                    source,
                    module,
                    children: child_modules(items),
                });
                for item in items {
                    self.visit_item(item);
                }
                self.scopes.pop();
            }
        }
    }

    fn visit_item_use(&mut self, item: &'ast syn::ItemUse) {
        if item.leading_colon.is_none() {
            self.use_tree(&item.tree, &mut Vec::new());
        }
    }

    fn visit_path(&mut self, path: &'ast syn::Path) {
        if path.leading_colon.is_none() {
            let segments = path
                .segments
                .iter()
                .map(|segment| segment.ident.to_string())
                .collect::<Vec<_>>();
            if segments.len() > 1 || matches!(segments[0].as_str(), "crate" | "super" | "self") {
                self.reference(&segments);
            }
        }
        syn::visit::visit_path(self, path);
    }

    fn visit_macro(&mut self, mac: &'ast syn::Macro) {
        self.visit_path(&mac.path);
        self.macro_tokens(mac.tokens.clone());
    }

    /// `pub(crate)`, `pub(super)`, and `pub(in path)` name who may see an item, not what it
    /// depends on.
    fn visit_visibility(&mut self, _: &'ast syn::Visibility) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The spellings a text scan for `crate::<module>` missed: each reaches a file that is not an
    /// input, so the guard names it.
    #[test]
    fn every_spelling_of_a_dependency_is_resolved() {
        let scratch = crate::ScratchDir::new("classifier-inputs").unwrap();
        let root = scratch.path();
        let write = |path: &str, text: &str| {
            let path = root.join(path);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, text).unwrap();
        };
        for module in [
            "src/lib.rs",
            "src/budgets.rs",
            "src/review.rs",
            "src/library_module.rs",
            "src/check.rs",
            "src/check/cfg_fact.rs",
            "src/check/loop_free.rs",
            "src/check/admission.rs",
            "src/hash.rs",
            "src/index.rs",
            "src/store.rs",
            "src/engine.rs",
            "src/triage/child.rs",
            "src/triage/tested.rs",
        ] {
            write(module, "pub fn f() {}\n");
        }
        write(
            "src/triage.rs",
            r#"
use crate::{budgets as spend, review as look};
use super::library_module as library;
use crate::check::cfg_fact::has_cfg_cycle;
mod child;
#[cfg(test)]
mod tested;

pub(crate) struct Visible;
pub(super) fn visible() {}

pub fn classify() -> bool {
    let _ = crate::check::loop_free::Bound::new();
    let _: crate::check::admission::NotSupported = todo!();
    let _ = crate::check::has_cfg_cycle("");
    let _ = child::f();
    let digest = format!("{}", crate::hash::sha256_bytes(b""));
    matches!(crate::index::State::A, _);
    vec![crate::store::Store::default()];
    has_cfg_cycle("")
}

mod inner {
    pub fn f() {
        let _ = super::super::engine::load();
        let _ = self::g();
    }
    fn g() {}
}

#[cfg(test)]
mod tests {
    use crate::tested_only::Everything;
}
"#,
        );
        let found = dependencies(root, &[], &["src/triage.rs"]).unwrap();
        assert_eq!(
            found.keys().map(String::as_str).collect::<Vec<_>>(),
            [
                "src/budgets.rs",
                "src/check.rs",
                "src/check/admission.rs",
                "src/check/cfg_fact.rs",
                "src/check/loop_free.rs",
                "src/engine.rs",
                "src/hash.rs",
                "src/index.rs",
                "src/library_module.rs",
                "src/review.rs",
                "src/store.rs",
                "src/triage.rs",
                "src/triage/child.rs",
            ]
        );

        write("src/triage.rs", "#[path = \"elsewhere.rs\"]\nmod moved;\n");
        write("src/elsewhere.rs", "\n");
        assert!(dependencies(root, &[], &["src/triage.rs"]).is_err());
        write("src/triage.rs", "mod missing;\n");
        assert!(dependencies(root, &[], &["src/triage.rs"]).is_err());
        write("src/triage.rs", "pub fn f() { super::super::x(); }\n");
        assert!(dependencies(root, &[], &["src/triage.rs"]).is_err());
    }

    /// A path into another workspace crate reaches its files: through the crate's name, through a
    /// name the crate root imports from it, and, inside it, through its own `crate::`.
    #[test]
    fn paths_into_a_member_crate_are_resolved() {
        let scratch = crate::ScratchDir::new("classifier-inputs-members").unwrap();
        let root = scratch.path();
        let write = |path: &str, text: &str| {
            let path = root.join(path);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, text).unwrap();
        };
        write(
            "src/lib.rs",
            "pub(crate) use front as air;\nuse front::budget;\npub use front::budget::Budget;\n\
             #[cfg(test)]\nuse front::tested;\nmod user;\n",
        );
        write(
            "src/user.rs",
            "use crate::air::canon::Format;\npub fn f(_: crate::Budget) {\n    \
             crate::budget::charge();\n    front::graph::walk();\n    \
             other::thing::call();\n}\n",
        );
        for module in [
            "front/src/lib.rs",
            "front/src/canon/mod.rs",
            "front/src/budget/mod.rs",
            "front/src/graph.rs",
            "front/src/tested.rs",
            "front/src/error.rs",
        ] {
            write(module, "pub fn f() {}\n");
        }
        write(
            "front/src/inner.rs",
            "use crate::error::Error;\nuse super::graph;\n",
        );
        let members = [Member {
            name: "front",
            dir: "front",
        }];
        let found = dependencies(root, &members, &["src/user.rs", "front/src/inner.rs"]).unwrap();
        assert_eq!(
            found
                .iter()
                .map(|(file, inputs)| (file.as_str(), inputs.iter().map(String::as_str).collect()))
                .collect::<Vec<(&str, Vec<&str>)>>(),
            [
                ("front/src/budget/mod.rs", vec!["src/user.rs"]),
                ("front/src/canon/mod.rs", vec!["src/user.rs"]),
                ("front/src/error.rs", vec!["front/src/inner.rs"]),
                (
                    "front/src/graph.rs",
                    vec!["front/src/inner.rs", "src/user.rs"]
                ),
            ]
        );
        // Without the member, the same paths name no file of this workspace.
        let found = dependencies(root, &[], &["src/user.rs"]).unwrap();
        assert_eq!(
            found.keys().map(String::as_str).collect::<Vec<_>>(),
            ["src/lib.rs"]
        );
    }

    /// Each crate is a member in its own directory, as in the workspace's crates/: one member's
    /// paths through another's name reach that member's `src/`, and its own `crate::` stays in its
    /// directory (the runtime crate on the translator's).
    #[test]
    fn a_member_reaches_another_by_its_name() {
        let scratch = crate::ScratchDir::new("classifier-inputs-sibling-member").unwrap();
        let root = scratch.path();
        let write = |path: &str, text: &str| {
            let path = root.join(path);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, text).unwrap();
        };
        for module in [
            "tr/src/lib.rs",
            "tr/src/device/mod.rs",
            "tr/src/metal.rs",
            "rt/src/lib.rs",
            "rt/src/device.rs",
        ] {
            write(module, "pub fn f() {}\n");
        }
        write(
            "rt/src/queue.rs",
            "use crate::device::Device;\nuse translator::device::Profile;\n\
             pub fn f() {\n    translator::metal::convert();\n}\n",
        );
        let members = [
            Member {
                name: "translator",
                dir: "tr",
            },
            Member {
                name: "rt",
                dir: "rt",
            },
        ];
        let found = dependencies(root, &members, &["rt/src/queue.rs"]).unwrap();
        assert_eq!(
            found.keys().map(String::as_str).collect::<Vec<_>>(),
            [
                "rt/src/device.rs",
                "tr/src/device/mod.rs",
                "tr/src/metal.rs"
            ]
        );
    }
}
