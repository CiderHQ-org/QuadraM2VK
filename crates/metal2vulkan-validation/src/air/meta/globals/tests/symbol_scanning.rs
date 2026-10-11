use super::*;

#[test]
fn symbol_scan_preserves_boundaries_and_quoted_spellings() {
    let text = r#"%result = call i32 @callee(i32 %a, ptr @cell-extra, ptr @cell.1, ptr @"cell with spaces", ptr @"escaped\22name") !note !"@not_a_symbol""#;
    assert_eq!(
        symbol_references(text).collect::<Vec<_>>(),
        [
            "%result",
            "@callee",
            "%a",
            "@cell-extra",
            "@cell.1",
            r#"@"cell with spaces""#,
            r#"@"escaped\22name""#,
        ]
    );
    assert_eq!(
        parse_global_name(r#"ptr @"cell with spaces", i32 1"#).as_deref(),
        Some(r#"@"cell with spaces""#)
    );
}
