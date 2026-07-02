from __future__ import annotations

import builtins
import html
import io
import keyword
import re
import token as py_token
import tokenize
from bisect import bisect_right
from collections import Counter
from typing import Iterable

from token_level_eval.common import SpanTag


CONTENT_COARSE = {"Noun", "Verb", "Adjective", "Adverb", "Interjection", "Qualifier"}
FUNCTION_COARSE = {
    "Existential",
    "Pronoun",
    "Det/Article",
    "Preposition",
    "Conjunction",
    "Aux BE",
    "Aux HAVE",
    "Aux DO",
    "Modal",
    "TO",
    "Wh-word",
}

BROWN_TO_COARSE: dict[str, str] = {
    **dict.fromkeys(["NN", "NN$", "NNS", "NNS$", "NP", "NP$", "NPS", "NPS$", "NR", "NR$", "NRS"], "Noun"),
    **dict.fromkeys(["VB", "VBD", "VBG", "VBN", "VBZ", "VBP"], "Verb"),
    **dict.fromkeys(["JJ", "JJ$", "JJR", "JJS", "JJT"], "Adjective"),
    **dict.fromkeys(["RB", "RB$", "RBR", "RBT", "RBS", "RN"], "Adverb"),
    "UH": "Interjection",
    "QL": "Qualifier",
    "QLP": "Qualifier",
    **dict.fromkeys(["PP$", "PP$$", "PPL", "PPLS", "PPO", "PPS", "PPSS", "PN", "PN$", "PRP", "PRP$"], "Pronoun"),
    **dict.fromkeys(["AT", "DT", "DT$", "DTI", "DTS", "DTX", "ABL", "ABN", "ABX", "AP", "AP$"], "Det/Article"),
    "IN": "Preposition",
    **dict.fromkeys(["CC", "CS"], "Conjunction"),
    **dict.fromkeys(["BE", "BED", "BEDZ", "BEG", "BEM", "BEN", "BER", "BEZ"], "Aux BE"),
    **dict.fromkeys(["HV", "HVD", "HVG", "HVN", "HVZ"], "Aux HAVE"),
    **dict.fromkeys(["DO", "DOD", "DOZ"], "Aux DO"),
    "MD": "Modal",
    "TO": "TO",
    "EX": "Existential",
    **dict.fromkeys(["WDT", "WP$", "WPO", "WPS", "WQL", "WRB", "WP"], "Wh-word"),
    **dict.fromkeys(["CD", "CD$", "OD"], "Numeral"),
    "RP": "Particle",
    **dict.fromkeys([",", ".", ":", "'", "''", "``", "--", "*", "SYM"], "Punctuation"),
    "-LRB-": "Open Bracket",
    "-RRB-": "Close Bracket",
}

OPEN_BRACKETS = set("([{<")
CLOSE_BRACKETS = set(")]}>")
VOID_HTML_TAGS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}

WORD_RE = re.compile(r"\w+(?:['-]\w+)*|[^\w\s]", re.UNICODE)
HTML_CHUNK_RE = re.compile(r"<!--.*?-->|<![^>]*>|<[^>]+>|[^<]+", re.DOTALL)
ATTR_RE = re.compile(r"([:\w.-]+)(\s*=\s*)(\"[^\"]*\"|'[^']*'|[^\s\"'=<>`]+)?", re.DOTALL)
LATEX_TOKEN_RE = re.compile(
    r"(?P<comment>%[^\n]*)|"
    r"(?P<begin>\\begin\s*\{[^}]+\})|"
    r"(?P<end>\\end\s*\{[^}]+\})|"
    r"(?P<command>\\[A-Za-z@]+|\\.)|"
    r"(?P<display>\$\$|\\\[|\\\])|"
    r"(?P<inline>\$)|"
    r"(?P<brace>[{}])|"
    r"(?P<bracket>[\[\]])|"
    r"(?P<table>&|\\\\|\\hline|\\toprule|\\midrule|\\bottomrule)|"
    r"(?P<newline>\n+)|"
    r"(?P<space>[ \t\r]+)|"
    r"(?P<text>[^\\%${}\[\]&\n \t\r]+)",
    re.DOTALL,
)


def aggregate_class(coarse: str) -> str:
    if coarse in CONTENT_COARSE:
        return "Content"
    if coarse in FUNCTION_COARSE:
        return "Function"
    return "Other"


def _simple_pos(word: str) -> str:
    lower = word.lower()
    if len(word) == 1 and word in OPEN_BRACKETS:
        return "("
    if len(word) == 1 and word in CLOSE_BRACKETS:
        return ")"
    if re.fullmatch(r"[^\w\s]", word):
        return "."
    if re.fullmatch(r"\d+(?:[.,]\d+)*", word):
        return "CD"
    if lower in {"a", "an", "the", "this", "that", "these", "those"}:
        return "DT"
    if lower in {"and", "or", "but", "nor", "yet", "so"}:
        return "CC"
    if lower in {"in", "on", "at", "by", "with", "for", "from", "of", "to", "into", "over", "under"}:
        return "IN" if lower != "to" else "TO"
    if lower in {"he", "she", "it", "they", "we", "i", "you", "him", "her", "them", "us", "me"}:
        return "PRP"
    if lower in {"is", "am", "are", "was", "were", "be", "being", "been"}:
        return "BE"
    if lower in {"has", "have", "had", "having"}:
        return "HV"
    if lower in {"do", "does", "did", "doing"}:
        return "DO"
    if lower in {"can", "could", "may", "might", "must", "shall", "should", "will", "would"}:
        return "MD"
    if lower in {"there"}:
        return "EX"
    if lower in {"who", "what", "when", "where", "why", "how", "which", "whose", "whom"}:
        return "WDT"
    if lower.endswith("ly"):
        return "RB"
    if lower.endswith(("ing", "ed", "ize", "ise")):
        return "VB"
    if lower.endswith(("ous", "ive", "able", "ible", "al", "ic")):
        return "JJ"
    return "NN"


def _pos_tags(words: list[str]) -> list[str]:
    try:
        import nltk

        return [tag for _, tag in nltk.pos_tag(words)]
    except Exception:
        return [_simple_pos(word) for word in words]


def tag_prose(text: str) -> list[SpanTag]:
    matches = list(WORD_RE.finditer(text))
    words = [match.group(0) for match in matches]
    pos_tags = _pos_tags(words)
    spans: list[SpanTag] = []
    for match, pos in zip(matches, pos_tags):
        token_text = match.group(0)
        if len(token_text) == 1 and token_text in OPEN_BRACKETS:
            coarse = "Open Bracket"
        elif len(token_text) == 1 and token_text in CLOSE_BRACKETS:
            coarse = "Close Bracket"
        else:
            coarse = BROWN_TO_COARSE.get(pos, "Other")
        spans.append(SpanTag(match.start(), match.end(), coarse, pos, token_text))
    return spans


def _line_starts(text: str) -> list[int]:
    starts = [0]
    for match in re.finditer("\n", text):
        starts.append(match.end())
    return starts


def _abs_pos(starts: list[int], row_col: tuple[int, int]) -> int:
    row, col = row_col
    return starts[max(row - 1, 0)] + col


def _python_op_tag(value: str) -> tuple[str, str]:
    if value in "([{":
        return "Open Bracket", "enclosure_open"
    if value in ")]}":
        return "Close Bracket", "enclosure_close"
    if value in {"+", "-", "*", "/", "//", "%", "**", "@", "@"}:
        return "Operator", "arithmetic"
    if value in {"=", "+=", "-=", "*=", "/=", "%=", "//=", "**=", ":="}:
        return "Operator", "assignment"
    if value in {"==", "!=", "<", ">", "<=", ">=", "is", "in"}:
        return "Operator", "comparison"
    if value in {"&", "|", "^", "~", "<<", ">>"}:
        return "Operator", "bitwise"
    return "Delimiter", "delimiter"


def tag_python(text: str) -> list[SpanTag]:
    starts = _line_starts(text)
    spans: list[SpanTag] = []
    builtins_set = set(dir(builtins))
    soft_keywords = set(getattr(keyword, "softkwlist", ()))
    try:
        token_iter = tokenize.generate_tokens(io.StringIO(text).readline)
        for tok in token_iter:
            value = tok.string
            if tok.type in {tokenize.ENCODING, tokenize.ENDMARKER} or not value:
                continue
            start = _abs_pos(starts, tok.start)
            end = _abs_pos(starts, tok.end)
            if tok.type == tokenize.NAME:
                if keyword.iskeyword(value):
                    coarse, fine = "Keyword", "keyword"
                elif value in soft_keywords:
                    coarse, fine = "Keyword", "soft_keyword"
                elif value in builtins_set:
                    coarse, fine = "Builtin", "builtin"
                elif value.startswith("__") and value.endswith("__"):
                    coarse, fine = "Identifier", "dunder"
                elif value.startswith("_"):
                    coarse, fine = "Identifier", "private"
                elif re.match(r"[A-Z][A-Za-z0-9]+$", value):
                    coarse, fine = "Identifier", "upper_camel"
                else:
                    coarse, fine = "Identifier", "identifier"
            elif tok.type == tokenize.STRING:
                prefix = value[: max(value.find(value.lstrip("rRuUbBfF")), 0)].lower()
                if value.startswith(('"""', "'''")) or value.lower().lstrip("rubf").startswith(('"""', "'''")):
                    fine = "triple_quoted"
                elif "f" in prefix:
                    fine = "fstring"
                elif "r" in prefix:
                    fine = "raw"
                elif "b" in prefix:
                    fine = "bytes"
                else:
                    fine = "string"
                coarse = "String"
            elif tok.type == tokenize.NUMBER:
                lower = value.lower()
                if lower.startswith("0x"):
                    fine = "hex"
                elif lower.startswith("0o"):
                    fine = "octal"
                elif "j" in lower:
                    fine = "complex"
                elif "." in value or "e" in lower:
                    fine = "float"
                else:
                    fine = "integer"
                coarse = "Number"
            elif tok.type == tokenize.COMMENT:
                coarse, fine = "Comment", "COMMENT"
            elif tok.type in {tokenize.NEWLINE, tokenize.NL}:
                coarse, fine = "Structure", "NEWLINE"
            elif tok.type == tokenize.INDENT:
                coarse, fine = "Structure", "INDENT"
            elif tok.type == tokenize.DEDENT:
                coarse, fine = "Structure", "DEDENT"
            elif tok.type == tokenize.OP:
                coarse, fine = _python_op_tag(value)
            else:
                coarse, fine = "Other", py_token.tok_name.get(tok.type, str(tok.type))
            spans.append(SpanTag(start, end, coarse, fine, value))
    except tokenize.TokenError:
        # Partial files are common in corpora; keep whatever tokenize produced.
        pass
    return spans


def _add_html_text_spans(spans: list[SpanTag], text: str, start: int, end: int) -> None:
    segment = text[start:end]
    for match in re.finditer(r"&[A-Za-z0-9#]+;|\S+|\s+", segment):
        value = match.group(0)
        if value.isspace():
            coarse, fine = "Whitespace", "ws"
        elif value.startswith("&") and value.endswith(";"):
            coarse, fine = "Text", "entity"
        else:
            coarse, fine = "Text", "content"
        spans.append(SpanTag(start + match.start(), start + match.end(), coarse, fine, value))


def _tag_html_tag(spans: list[SpanTag], source: str, start: int, end: int) -> None:
    value = source[start:end]
    if value.startswith("<!--"):
        spans.append(SpanTag(start, end, "Comment", "comment", value))
        return
    if value.lower().startswith("<!doctype") or value.startswith("<!"):
        spans.append(SpanTag(start, end, "Tag", "doctype", value))
        return

    close = value.startswith("</")
    self_close = value.rstrip().endswith("/>")
    name_match = re.match(r"</?\s*([A-Za-z][\w:-]*)", value)
    tag_name = name_match.group(1).lower() if name_match else ""
    if name_match:
        name_start = start + name_match.start(1)
        name_end = start + name_match.end(1)
        if close:
            fine = "close"
        elif self_close:
            fine = "self_close"
        elif tag_name in VOID_HTML_TAGS:
            fine = "void"
        else:
            fine = "open"
        spans.append(SpanTag(name_start, name_end, "Tag", fine, source[name_start:name_end]))

    for offset, char in enumerate(value):
        abs_i = start + offset
        if char == "<":
            spans.append(SpanTag(abs_i, abs_i + 1, "Punctuation", "angle_open", char))
        elif char == ">":
            spans.append(SpanTag(abs_i, abs_i + 1, "Punctuation", "angle_close", char))
        elif char == "/":
            spans.append(SpanTag(abs_i, abs_i + 1, "Punctuation", "slash", char))
        elif char in {"'", '"'}:
            spans.append(SpanTag(abs_i, abs_i + 1, "Punctuation", "quote", char))

    attr_region_start = name_match.end(0) if name_match else 1
    attr_region_end = len(value) - (2 if self_close else 1)
    for match in ATTR_RE.finditer(value, attr_region_start, max(attr_region_end, attr_region_start)):
        name = match.group(1)
        if not name or name.startswith("/"):
            continue
        name_start = start + match.start(1)
        name_end = start + match.end(1)
        spans.append(SpanTag(name_start, name_end, "Attribute", "name", name))
        if match.group(2):
            eq_index = value.find("=", match.start(2), match.end(2))
            if eq_index >= 0:
                spans.append(SpanTag(start + eq_index, start + eq_index + 1, "Attribute", "equals", "="))
        if match.group(3):
            val_start = start + match.start(3)
            val_end = start + match.end(3)
            fine = "value_quoted" if match.group(3).startswith(("'", '"')) else "value_unquoted"
            spans.append(SpanTag(val_start, val_end, "Attribute", fine, source[val_start:val_end]))


def tag_html(text: str) -> list[SpanTag]:
    spans: list[SpanTag] = []
    for match in HTML_CHUNK_RE.finditer(text):
        value = match.group(0)
        if value.startswith("<"):
            _tag_html_tag(spans, text, match.start(), match.end())
        else:
            _add_html_text_spans(spans, text, match.start(), match.end())
    return spans


def _latex_command_fine(value: str) -> str:
    command = value.lstrip("\\")
    if command in {"section", "subsection", "subsubsection", "paragraph", "chapter"}:
        return "section"
    if command in {"ref", "cref", "Cref", "autoref", "label"}:
        return "ref"
    if command in {"cite", "citet", "citep", "citealt", "parencite"}:
        return "cite"
    if command in {"textbf", "emph", "textit", "underline", "mathbf", "mathrm"}:
        return "formatting"
    return "control"


def tag_latex(text: str) -> list[SpanTag]:
    spans: list[SpanTag] = []
    for match in LATEX_TOKEN_RE.finditer(text):
        value = match.group(0)
        kind = match.lastgroup or "text"
        if kind == "comment":
            coarse, fine = "Comment", "comment"
        elif kind in {"begin", "end"}:
            coarse, fine = "Environment", "begin/end"
        elif kind == "command":
            coarse, fine = "Command", _latex_command_fine(value)
        elif kind == "display":
            coarse = "Math"
            fine = "display_open/close"
        elif kind == "inline":
            coarse, fine = "Math", "inline_open/close"
        elif kind == "brace":
            coarse = "Group"
            fine = "brace_open" if value == "{" else "brace_close"
        elif kind == "bracket":
            coarse = "Group"
            fine = "bracket_open" if value == "[" else "bracket_close"
        elif kind == "table":
            coarse = "Table"
            fine = "ampersand" if value == "&" else "linebreak"
        elif kind == "newline":
            coarse = "Newline"
            fine = "blank_line" if "\n\n" in value else "newline"
        elif kind == "space":
            coarse, fine = "Text", "space"
        else:
            coarse, fine = "Text", "content"
        spans.append(SpanTag(match.start(), match.end(), coarse, fine, value))
    return spans


def tag_source(text: str, domain: str) -> list[SpanTag]:
    normalized = domain.lower()
    if normalized in {"prose", "text", "news", "wikipedia", "pg19", "arxiv"}:
        return tag_prose(text)
    if normalized in {"python", "py", "code"}:
        return tag_python(text)
    if normalized in {"html", "htm", "xml", "markup"}:
        return tag_html(text)
    if normalized in {"latex", "tex"}:
        return tag_latex(text)
    return tag_prose(text)


def tags_for_span(source_tags: Iterable[SpanTag], start: int, end: int) -> list[SpanTag]:
    if end <= start:
        return []
    return [tag for tag in source_tags if tag.overlaps(start, end)]


def word_position(source_tags: Iterable[SpanTag], start: int, end: int) -> str:
    candidates = [
        tag for tag in source_tags
        if tag.overlaps(start, end) and re.search(r"\w", tag.text) and tag.coarse not in {"Punctuation", "Whitespace"}
    ]
    if not candidates:
        return "none"
    tag = max(candidates, key=lambda item: min(item.end, end) - max(item.start, start))
    if start <= tag.start and end >= tag.end:
        return "whole"
    if start <= tag.start:
        return "prefix"
    if end >= tag.end:
        return "suffix"
    return "middle"


def copy_features(token_ids: list[int], max_ngram: int) -> tuple[list[dict[str, bool]], list[int | None], Counter[int]]:
    seen_by_n = {n: set() for n in range(1, max_ngram + 1)}
    last_seen: dict[int, int] = {}
    features: list[dict[str, bool]] = []
    prev_distances: list[int | None] = []
    counts = Counter(token_ids)

    for pos, token_id in enumerate(token_ids):
        row: dict[str, bool] = {}
        for n in range(1, max_ngram + 1):
            if pos - n + 1 < 0:
                row[f"copy_{n}"] = False
                continue
            ngram = tuple(token_ids[pos - n + 1 : pos + 1])
            row[f"copy_{n}"] = ngram in seen_by_n[n]
        prev_distances.append(pos - last_seen[token_id] if token_id in last_seen else None)
        features.append(row)

        for n in range(1, max_ngram + 1):
            if pos - n + 1 >= 0:
                seen_by_n[n].add(tuple(token_ids[pos - n + 1 : pos + 1]))
        last_seen[token_id] = pos

    return features, prev_distances, counts


def approximate_offsets(text: str, token_texts: list[str]) -> list[tuple[int, int]]:
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for token in token_texts:
        clean = token.replace("Ġ", " ").replace("▁", " ")
        if not clean:
            offsets.append((cursor, cursor))
            continue
        found = text.find(clean, cursor)
        if found < 0 and clean.startswith(" "):
            found = text.find(clean.lstrip(), cursor)
            clean = clean.lstrip()
        if found < 0:
            offsets.append((cursor, cursor))
            continue
        offsets.append((found, found + len(clean)))
        cursor = found + len(clean)
    return offsets


def token_type_frequency(token_ids: list[int]) -> dict[int, int]:
    return dict(Counter(token_ids))


def sorted_tag_names(spans: Iterable[SpanTag]) -> tuple[list[str], list[str], list[str]]:
    coarse = sorted({span.coarse for span in spans})
    fine = sorted({f"{span.coarse}/{span.fine}" for span in spans})
    aggregate = sorted({aggregate_class(span.coarse) for span in spans})
    return coarse, fine, aggregate

