"""The Vulkan specification's "Required Format Support" tables as typed requirements.

The source is the specification's own AsciiDoc (`chapters/formats.txt` of a pinned Vulkan-Docs
tag): `vk.xml` describes formats but not which features they must support. The text is evaluated
for one core version with no extension attributes (`ifdef`/`ifndef`/`endif`, block and single-line
forms), then every table anchored `formats-mandatory-features-*` in the `Required Format Support`
section is laid out as a grid (column and row spans) and read cell by cell:

* the header rows name each feature column (`N+>| ename:VK_FORMAT_FEATURE_*` spans the columns to
  the left of its `{downarrow}` cell);
* `{sym1}` is an unconditional requirement;
* every other symbol takes its meaning from a footnote row of the same table, which must match
  one of three footnote forms exactly (a feature condition, a choice between two formats, a
  choice between whole tables); anything else is an error, never a guess;
* a footnote that states a prohibition rather than a requirement is kept as text in
  `not_represented`.

Which bitmask a feature belongs to (`optimalTilingFeatures` or `bufferFeatures`) comes from the
section's own "Feature bits in ..." tables, and the section's sentence that formats required to
support `SAMPLED_IMAGE` also support `TRANSFER_SRC` and `TRANSFER_DST` adds those two bits
wherever `SAMPLED_IMAGE` is required. The sentence is matched verbatim, so a source without it
fails.

The result is `{"tables", "unconditional", "conditional", "not_represented"}`. `unconditional`
maps each format to `{member: [bits]}` (`VkFormatProperties` member names); each `conditional`
requirement is `{"tables", "any_of": [table, ...]}` plus `"if_feature"` when it only binds a device
supporting that `VkPhysicalDeviceFeatures` member: at least one table of `any_of` must hold in
full.
"""
import re

# Vulkan-Docs v1.2.203 (commit 30061d7ad920d8bcd0f936531abe9ca49d5eb22f) is the last revision of
# the Vulkan 1.2 specification; v1.3.204 follows it.
SPEC = {
    "url": ("https://raw.githubusercontent.com/KhronosGroup/Vulkan-Docs/v1.2.203/"
            "chapters/formats.txt"),
    "sha256": "2031f11fe2b7269502eecab10f0c6f386e33c34fcc5be02f0c25fc8bb50badd5",
    "tag": "v1.2.203",
}
# The attributes the specification defines when it is built for core Vulkan 1.2 alone.
VERSION_1_2 = ["VK_VERSION_1_0", "VK_VERSION_1_1", "VK_VERSION_1_2"]
SECTION = "[[features-required-format-support]]"
TABLE_ANCHOR = re.compile(r"^\[\[(formats-mandatory-features-[a-z0-9-]+)\]\]$")
FIELDS = {
    "optimalTilingFeatures": "Feature bits in pname:optimalTilingFeatures",
    "bufferFeatures": "Feature bits in pname:bufferFeatures",
}
TRANSFER_RULE = ("Formats that are required to support ename:VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT "
                 "must: also support ename:VK_FORMAT_FEATURE_TRANSFER_SRC_BIT and "
                 "ename:VK_FORMAT_FEATURE_TRANSFER_DST_BIT.")
SAMPLED = "VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT"
TRANSFER = ["VK_FORMAT_FEATURE_TRANSFER_SRC_BIT", "VK_FORMAT_FEATURE_TRANSFER_DST_BIT"]
MARKS = {"", "{sym1}", "{sym2}", "{sym3}", "{sym4}"}

FEATURE_FOOTNOTE = re.compile(
    r"^Format features marked (?:with )?\{(sym\d)\} must: be supported for "
    r"pname:optimalTilingFeatures if the sname:VkPhysicalDevice supports the "
    r"<<features-(\w+), pname:(\w+)>> feature\.$")
FORMAT_CHOICE_FOOTNOTE = re.compile(
    r"^ename:(VK_FORMAT_FEATURE_\w+) feature must: be supported for at least one of "
    r"ename:(VK_FORMAT_\w+) and ename:(VK_FORMAT_\w+), and must: be supported for at least one "
    r"of ename:(VK_FORMAT_\w+) and ename:(VK_FORMAT_\w+)\.$")
TABLE_CHOICE_FOOTNOTE = re.compile(
    r"^The ename:(VK_FORMAT_FEATURE_\w+), ename:(VK_FORMAT_FEATURE_\w+) and "
    r"ename:(VK_FORMAT_FEATURE_\w+) features must: be supported in pname:optimalTilingFeatures "
    r"for all the formats in at least one of: this table, <<([\w-]+)>>, or <<([\w-]+)>>\.$")
PROHIBITION_FOOTNOTE = re.compile(
    r"^pname:bufferFeatures must: not support any features for these formats$")

# An AsciiDoc cell spec directly before its `|`: [colspan][.rowspan]+, [n*], [halign][.valign],
# [style]. It is preceded by whitespace or starts the text.
CELL_SPEC = re.compile(
    r"(?:^|(?<=\s))(?P<span>\d*(?:\.\d+)?\+)?(?P<dup>\d+\*)?(?P<halign>[<^>])?"
    r"(?P<valign>\.[<^>])?(?P<style>[adehlmsv])?$")


class SpecError(Exception):
    pass


def preprocess(text, attributes):
    """The lines of `text` that a build defining exactly `attributes` keeps, without comments."""
    defined = set(attributes)
    out = []
    stack = []  # (attribute expression, holds)

    def holds(kind, names):
        if "," in names and "+" in names:
            raise SpecError(f"mixed conditional expression {names}")
        if kind == "ifdef":
            if "+" in names:
                return all(n in defined for n in names.split("+"))
            return any(n in defined for n in names.split(","))
        if "," in names or "+" in names:
            # Not needed by the tables: refuse it rather than pick a reading.
            raise SpecError(f"multi-attribute ifndef::{names}[]")
        return names not in defined

    comment_block = False
    for line in text.splitlines():
        if line == "////":
            comment_block = not comment_block
            continue
        if comment_block or re.match(r"^//(?!/)", line):
            continue
        match = re.match(r"^(ifdef|ifndef|endif|ifeval)::([^\[]*)\[(.*)\]$", line)
        keep = all(h for _, h in stack)
        if match is None:
            if keep:
                out.append(line)
            continue
        kind, names, content = match.groups()
        if kind == "ifeval":
            raise SpecError(f"ifeval is not supported: {line}")
        if kind == "endif":
            if not stack or (names and stack[-1][0] != names):
                raise SpecError(f"unbalanced {line}")
            stack.pop()
            continue
        if content:
            if keep and holds(kind, names):
                out.append(content)
            continue
        stack.append((names, holds(kind, names)))
    if stack or comment_block:
        raise SpecError(f"unterminated conditionals {[n for n, _ in stack]} or comment block")
    return out


class Cell:
    def __init__(self, text, spec):
        if spec.group("dup"):
            raise SpecError(f"cell duplication {spec.group('dup')} is not supported")
        self.colspan, self.rowspan = 1, 1
        span = spec.group("span")
        if span:
            match = re.fullmatch(r"(\d*)(?:\.(\d+))?\+", span)
            self.colspan = int(match.group(1) or 1)
            self.rowspan = int(match.group(2) or 1)
        self.style = spec.group("style")
        self.text = " ".join(text.split())
        self.row = None
        self.column = None


def cells(body):
    """The cells of an AsciiDoc table body, each with the span and style its spec gives."""
    pieces = body.split("|")
    spec = CELL_SPEC.fullmatch(pieces[0].strip())
    if spec is None:
        raise SpecError(f"text before the first cell: {pieces[0]!r}")
    result = []
    for index, piece in enumerate(pieces[1:], start=1):
        if index + 1 < len(pieces):
            following = CELL_SPEC.search(piece)
            if following is None:
                # The next `|` directly follows text: the next cell has no spec.
                following = CELL_SPEC.search("")
                text = piece
            else:
                text = piece[:following.start()]
        else:
            following, text = None, piece
        result.append(Cell(text, spec))
        spec = following
    return result


def grid(body, columns):
    """The rows of a table body laid out over `columns`, honoring column and row spans."""
    rows = []
    occupied = set()
    row, column = 0, 0
    for cell in cells(body):
        while (row, column) in occupied:
            column += 1
            if column == columns:
                row, column = row + 1, 0
        if column + cell.colspan > columns:
            raise SpecError(f"cell {cell.text!r} overflows row {row}")
        for r in range(row, row + cell.rowspan):
            for c in range(column, column + cell.colspan):
                if (r, c) in occupied:
                    raise SpecError(f"cell {cell.text!r} overlaps a spanning cell")
                occupied.add((r, c))
        cell.row, cell.column = row, column
        while len(rows) <= row:
            rows.append([])
        rows[row].append(cell)
        column += cell.colspan
        if column == columns:
            row, column = row + 1, 0
    if column != 0 or any(r >= row for r, _ in occupied):
        raise SpecError("the table ends inside a row or a row span")
    return rows


def delimited_tables(lines):
    """Every `|====` table in `lines`: (anchor or None, title or None, attribute line, body)."""
    out = []
    index = 0
    while index < len(lines):
        if lines[index] != "|====":
            index += 1
            continue
        header = lines[max(0, index - 3):index]
        anchors = [TABLE_ANCHOR.match(line) for line in header]
        anchor = next((m.group(1) for m in anchors if m), None)
        title = next((line[1:] for line in header
                      if line.startswith(".") and not line.startswith("..")), None)
        attributes = next((line for line in header
                           if line.startswith("[") and not line.startswith("[[")), "")
        try:
            end = lines.index("|====", index + 1)
        except ValueError:
            raise SpecError("unterminated table") from None
        out.append((anchor, title, attributes, "\n".join(lines[index + 1:end])))
        index = end + 1
    return out


def column_count(attributes):
    match = re.search(r'cols="([^"]*)"', attributes)
    if match is None:
        return 1
    entries = match.group(1).split(",")
    if any("*" in entry for entry in entries):
        raise SpecError(f"column multipliers are not supported: {attributes}")
    return len(entries)


def ename(text, prefix):
    match = re.fullmatch(r"ename:(" + prefix + r"\w+)", text)
    if match is None:
        raise SpecError(f"expected a {prefix}* enumerant, found {text!r}")
    return match.group(1)


def feature_fields(tables):
    """Feature bit -> its `VkFormatProperties` member, in the section's order."""
    fields = {}
    for field, title in FIELDS.items():
        bodies = [body for _, t, _, body in tables if t and t.lower() == title.lower()]
        if len(bodies) != 1:
            raise SpecError(f"expected one table titled {title!r}, found {len(bodies)}")
        for row in grid(bodies[0], 1):
            fields[ename(row[0].text, "VK_FORMAT_FEATURE_")] = field
    missing = [bit for bit in [SAMPLED] + TRANSFER if bit not in fields]
    if missing:
        raise SpecError(f"the Feature bits tables lack {missing}")
    return fields


def read_table(anchor, attributes, body, fields):
    """One mandatory table: its rows as [(format, {feature: symbol})] and its footnotes."""
    columns = column_count(attributes)
    rows = grid(body, columns)
    format_row = next((i for i, row in enumerate(rows)
                       if len(row) == 1 and row[0].text == "Format" and row[0].style == "s"),
                      None)
    if format_row is None:
        raise SpecError(f"{anchor}: no Format header row")
    features = {}
    for index, row in enumerate(rows[:format_row]):
        if len(row) != 2 or row[1].text != "{downarrow}":
            raise SpecError(f"{anchor}: unexpected header row {[c.text for c in row]}")
        name, arrow = row
        if name.column != 0 or arrow.column != name.colspan:
            raise SpecError(f"{anchor}: {name.text} does not point at its column")
        if index + arrow.rowspan - 1 != format_row:
            raise SpecError(f"{anchor}: the {name.text} arrow does not reach the Format row")
        features[arrow.column] = ename(name.text, "VK_FORMAT_FEATURE_")
    if sorted(features) != list(range(1, columns)):
        raise SpecError(f"{anchor}: feature columns {sorted(features)} do not cover the table")
    unknown = sorted(set(features.values()) - set(fields))
    if unknown:
        raise SpecError(f"{anchor}: features outside the Feature bits tables: {unknown}")
    entries = []
    footnotes = []
    for row in rows[format_row + 1:]:
        if len(row) == 1 and row[0].colspan == columns:
            footnotes.append(row[0].text)
            continue
        if len(row) != columns or any(c.colspan != 1 or c.rowspan != 1 for c in row):
            raise SpecError(f"{anchor}: malformed row {[c.text for c in row]}")
        name = ename(row[0].text, "VK_FORMAT_")
        marks = {}
        for cell in row[1:]:
            if cell.text not in MARKS:
                raise SpecError(f"{anchor}: {name}: unknown mark {cell.text!r}")
            if cell.text:
                marks[features[cell.column]] = cell.text[1:-1]
        entries.append((name, marks))
    return entries, footnotes


def table(pairs, fields):
    """`{format: {member: [bits]}}` from (format, bit) pairs."""
    out = {}
    for name, bit in pairs:
        out.setdefault(name, {}).setdefault(fields[bit], []).append(bit)
    return out


def interpret(anchor, entries, footnotes, fields):
    """The typed requirements of one table: (unconditional pairs, conditional requirements,
    table choices, not-represented footnotes)."""
    meanings = {}
    not_represented = []
    for text in footnotes:
        feature = FEATURE_FOOTNOTE.match(text)
        formats = FORMAT_CHOICE_FOOTNOTE.match(text)
        tables = TABLE_CHOICE_FOOTNOTE.match(text)
        if feature:
            symbol, target, member = feature.groups()
            if target != member:
                raise SpecError(f"{anchor}: footnote names two features: {text}")
            meanings.setdefault(symbol, []).append(("feature", member))
        elif formats:
            bit, a, b, c, d = formats.groups()
            meanings.setdefault("sym2", []).append(("formats", bit, [[a, b], [c, d]]))
        elif tables:
            bits = list(tables.groups()[:3])
            meanings.setdefault("sym2", []).append(
                ("tables", bits, sorted([anchor, tables.group(4), tables.group(5)])))
        elif PROHIBITION_FOOTNOTE.match(text):
            not_represented.append({"table": anchor, "text": text})
        else:
            raise SpecError(f"{anchor}: unrecognized footnote: {text}")
    used = {symbol for _, marks in entries for symbol in marks.values()}
    for symbol, meaning in meanings.items():
        if len(meaning) != 1:
            raise SpecError(f"{anchor}: {symbol} has {len(meaning)} footnotes")
        if symbol not in used:
            raise SpecError(f"{anchor}: a footnote defines {symbol}, which no cell uses")
    for symbol in sorted(used - {"sym1"} - set(meanings)):
        raise SpecError(f"{anchor}: {symbol} is used but no footnote defines it")

    def marked(symbol):
        return sorted((name, bit) for name, marks in entries for bit, s in marks.items()
                      if s == symbol)

    conditional = []
    choices = []
    for symbol, [meaning] in sorted(meanings.items()):
        pairs = marked(symbol)
        if meaning[0] == "feature":
            wrong = sorted({bit for _, bit in pairs if fields[bit] != "optimalTilingFeatures"})
            if wrong:
                raise SpecError(f"{anchor}: {symbol} marks buffer features {wrong}")
            conditional.append({"tables": [anchor], "if_feature": meaning[1],
                                "any_of": [table(pairs, fields)]})
        elif meaning[0] == "formats":
            _, bit, groups = meaning
            if pairs != sorted((name, bit) for group in groups for name in group):
                raise SpecError(f"{anchor}: {symbol} marks {pairs}, not its footnote's formats")
            for group in groups:
                conditional.append({"tables": [anchor], "any_of": [
                    table([(name, bit)], fields) for name in group]})
        else:
            _, bits, anchors = meaning
            if pairs != sorted((name, bit) for name, _ in entries for bit in bits):
                raise SpecError(f"{anchor}: {symbol} does not mark exactly {bits} on every row")
            choices.append((tuple(anchors), tuple(bits), anchor))
    return marked("sym1"), conditional, choices, not_represented


def with_transfer(entries):
    """A table with `TRANSFER_SRC`/`TRANSFER_DST` added wherever `SAMPLED_IMAGE` is required."""
    for props in entries.values():
        bits = props.get("optimalTilingFeatures", [])
        if SAMPLED in bits:
            bits.extend(bit for bit in TRANSFER if bit not in bits)
    return entries


def canonical(entries, order):
    """Formats sorted by name, members by name, bits in the section's order."""
    return {name: {member: sorted(bits, key=order.index)
                   for member, bits in sorted(props.items())}
            for name, props in sorted(entries.items())}


def section_text(text):
    """The raw lines of the Required Format Support section, up to the next `==`/`===` heading.
    The section must be outside every conditional and close the ones it opens."""
    lines = text.splitlines()
    if SECTION not in lines:
        raise SpecError(f"no {SECTION} section")
    start = lines.index(SECTION)
    end = next((i for i in range(start + 2, len(lines)) if re.match(r"^={2,3} ", lines[i])),
               len(lines))
    for bounds in ((0, start), (start, end)):
        depth = 0
        for line in lines[slice(*bounds)]:
            if re.match(r"^(ifdef|ifndef)::[^\[]*\[\]$", line):
                depth += 1
            elif re.match(r"^endif::[^\[]*\[\]$", line):
                depth -= 1
        if depth != 0:
            raise SpecError(f"the {SECTION} section is not outside every conditional")
    return "\n".join(lines[start:end])


def required_formats(text, attributes):
    """The Required Format Support tables of `text` built with `attributes` (module doc)."""
    section = preprocess(section_text(text), attributes)
    if TRANSFER_RULE not in " ".join(" ".join(section).split()):
        raise SpecError("the SAMPLED_IMAGE -> TRANSFER_SRC/TRANSFER_DST sentence is missing")
    tables = delimited_tables(section)
    fields = feature_fields(tables)
    order = list(fields)
    anchors = []
    rows = {}
    unconditional = []
    conditional = []
    choices = {}
    not_represented = []
    for anchor, _, attributes_line, body in tables:
        if anchor is None:
            continue
        entries, footnotes = read_table(anchor, attributes_line, body, fields)
        anchors.append(anchor)
        rows[anchor] = entries
        pairs, extra, table_choices, prohibitions = interpret(anchor, entries, footnotes,
                                                              fields)
        unconditional.extend(pairs)
        conditional.extend(extra)
        not_represented.extend(prohibitions)
        for key, bits, stated_by in table_choices:
            choices.setdefault(key, []).append((bits, stated_by))
    if not anchors:
        raise SpecError("no formats-mandatory-features-* tables")
    names = [name for anchor in anchors for name, _ in rows[anchor]]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise SpecError(f"formats listed by two tables: {duplicates}")
    # A choice between tables is stated by each table it names, with the same features.
    for key, statements in sorted(choices.items()):
        if sorted(stated_by for _, stated_by in statements) != list(key):
            raise SpecError(f"the choice between {key} is not stated by each of its tables")
        if len({bits for bits, _ in statements}) != 1:
            raise SpecError(f"the choice between {key} names different features")
        bits = statements[0][0]
        conditional.append({"tables": list(key), "any_of": [
            table([(name, bit) for name, _ in rows[anchor] for bit in bits], fields)
            for anchor in key]})
    return {
        "tables": anchors,
        "unconditional": canonical(with_transfer(table(unconditional, fields)), order),
        "conditional": [dict(r, any_of=[canonical(with_transfer(t), order) for t in r["any_of"]])
                        for r in conditional],
        "not_represented": not_represented,
    }
