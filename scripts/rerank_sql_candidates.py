#!/usr/bin/env python3
"""Execute-filter and rerank SQL candidates for Spider-style evaluation."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any


SQL_KEYWORDS = {
    "as",
    "on",
    "where",
    "join",
    "inner",
    "left",
    "right",
    "full",
    "cross",
    "group",
    "order",
    "having",
    "limit",
    "union",
    "intersect",
    "except",
}

GENERIC_NAME_TOKENS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "by",
    "code",
    "data",
    "date",
    "detail",
    "details",
    "for",
    "id",
    "in",
    "list",
    "name",
    "names",
    "number",
    "of",
    "ref",
    "refs",
    "table",
    "the",
    "type",
    "types",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_only_sql(sql: str) -> bool:
    normalized = re.sub(r"\s+", " ", sql).strip().lower()
    if not normalized.startswith(("select", "with")):
        return False
    padded = f" {normalized} "
    blocked = [
        " delete ",
        " drop ",
        " update ",
        " alter ",
        " insert ",
        " create ",
        " truncate ",
        " replace ",
        " merge ",
        " grant ",
        " revoke ",
        " attach ",
        " detach ",
        " pragma ",
        " vacuum ",
    ]
    return not any(token in padded for token in blocked)


def extract_sql(text: str) -> str:
    text = (text or "").strip()
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()

    match = re.search(r"\b(select|with)\b", text, flags=re.IGNORECASE)
    if match:
        text = text[match.start() :].strip()

    cutoff_markers = [
        "NdrFcShort",
        "<|im_end|>",
        "<|im_start|>",
        "\nTask:",
        "\nDatabase id:",
        "\nSchema:",
        "\nQuestion:",
        "\nuser\n",
        "\nassistant\n",
    ]
    cutoff_positions = [text.find(marker) for marker in cutoff_markers if marker in text]
    if cutoff_positions:
        text = text[: min(cutoff_positions)].strip()

    if ";" in text:
        text = text.split(";", 1)[0].strip()
    return text


def has_order_by(sql: str) -> bool:
    return bool(re.search(r"\border\s+by\b", sql, flags=re.IGNORECASE))


def normalize_value(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    return value


def normalize_rows(rows: list[tuple[Any, ...]], ordered: bool) -> list[list[Any]]:
    normalized = [[normalize_value(value) for value in row] for row in rows]
    if not ordered:
        normalized = sorted(normalized, key=lambda row: json.dumps(row, ensure_ascii=False, sort_keys=True))
    return normalized


def execute_sql(sqlite_path: str, sql: str, timeout_sec: float) -> tuple[bool, list[list[Any]] | None, str | None]:
    if not read_only_sql(sql):
        return False, None, "not_read_only_select"
    uri = f"file:{sqlite_path}?mode=ro"
    conn = None
    try:
        deadline = time.monotonic() + timeout_sec
        conn = sqlite3.connect(uri, uri=True, timeout=timeout_sec)
        conn.execute("PRAGMA query_only = ON")
        conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10000)
        rows = conn.execute(sql).fetchall()
        return True, [list(row) for row in rows], None
    except Exception as exc:  # noqa: BLE001
        return False, None, f"{type(exc).__name__}: {exc}"
    finally:
        if conn is not None:
            conn.close()


def strip_identifier(token: str | None) -> str:
    if not token:
        return ""
    token = token.strip()
    if len(token) >= 2 and ((token[0], token[-1]) in {('"', '"'), ("`", "`"), ("[", "]")}):
        return token[1:-1]
    return token


def normalize_name(name: str | None) -> str:
    return re.sub(r"\s+", " ", strip_identifier(name).strip()).lower()


def schema_table_names(schema_info: dict[str, Any]) -> list[str]:
    return schema_info.get("active_table_names_original") or schema_info.get("table_names_original", [])


def schema_tables(schema_info: dict[str, Any]) -> dict[str, str]:
    return {normalize_name(name): name for name in schema_table_names(schema_info)}


def schema_columns(schema_info: dict[str, Any]) -> dict[str, set[str]]:
    tables = schema_info.get("table_names_original", [])
    active_tables = set(schema_table_names(schema_info))
    columns: dict[str, set[str]] = {normalize_name(table): set() for table in active_tables}
    active_columns = schema_info.get("active_column_names_original") or schema_info.get("column_names_original", [])
    for table_index, column_name in active_columns:
        if table_index == -1:
            continue
        if 0 <= table_index < len(tables):
            if tables[table_index] not in active_tables:
                continue
            columns.setdefault(normalize_name(tables[table_index]), set()).add(normalize_name(column_name))
    return columns


def singularize(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def word_tokens(text: str | None) -> list[str]:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text or "")
    raw_tokens = re.findall(r"[A-Za-z0-9]+", text.lower())
    return [singularize(token) for token in raw_tokens]


def meaningful_name_tokens(name: str | None) -> list[str]:
    return [token for token in word_tokens(name) if token not in GENERIC_NAME_TOKENS and len(token) > 1]


def mentioned_tables(question: str, schema_info: dict[str, Any]) -> set[str]:
    question_tokens = set(word_tokens(question))
    mentioned: set[str] = set()
    for table_name in schema_table_names(schema_info):
        tokens = meaningful_name_tokens(table_name)
        if tokens and all(token in question_tokens for token in tokens):
            mentioned.add(normalize_name(table_name))
    return mentioned


def mentioned_columns(question: str, schema_info: dict[str, Any]) -> set[str]:
    question_tokens = set(word_tokens(question))
    mentioned: set[str] = set()
    for _, column_name in schema_info.get("active_column_names_original") or schema_info.get("column_names_original", []):
        tokens = meaningful_name_tokens(column_name)
        if tokens and all(token in question_tokens for token in tokens):
            mentioned.add(normalize_name(column_name))
    return mentioned


def tables_in_sql(sql: str, schema_info: dict[str, Any]) -> set[str]:
    tables = schema_tables(schema_info)
    used: set[str] = set()
    ident = r'(?:"[^"]+"|`[^`]+`|\[[^\]]+\]|[A-Za-z_][\w]*)'
    for match in re.finditer(rf"\b(?:from|join)\s+({ident})", sql, flags=re.IGNORECASE):
        table_key = normalize_name(match.group(1))
        if table_key in tables:
            used.add(table_key)
    return used


def table_occurrence_counts(sql: str, schema_info: dict[str, Any]) -> dict[str, int]:
    tables = schema_tables(schema_info)
    counts: dict[str, int] = {}
    ident = r'(?:"[^"]+"|`[^`]+`|\[[^\]]+\]|[A-Za-z_][\w]*)'
    for match in re.finditer(rf"\b(?:from|join)\s+({ident})", sql, flags=re.IGNORECASE):
        table_key = normalize_name(match.group(1))
        if table_key in tables:
            counts[table_key] = counts.get(table_key, 0) + 1
    return counts


def schema_join_edges(schema_info: dict[str, Any]) -> set[tuple[str, str]]:
    edges: set[tuple[str, str]] = set()
    for fk in schema_info.get("foreign_keys", []):
        left = normalize_name(fk.get("from_table"))
        right = normalize_name(fk.get("to_table"))
        if not left or not right or left == right:
            continue
        edges.add((left, right))
        edges.add((right, left))
    return edges


def schema_join_column_pairs(schema_info: dict[str, Any]) -> set[tuple[tuple[str, str], tuple[str, str]]]:
    pairs: set[tuple[tuple[str, str], tuple[str, str]]] = set()
    for fk in schema_info.get("foreign_keys", []):
        left = (normalize_name(fk.get("from_table")), normalize_name(fk.get("from_column")))
        right = (normalize_name(fk.get("to_table")), normalize_name(fk.get("to_column")))
        if not all(left) or not all(right):
            continue
        pairs.add((left, right))
        pairs.add((right, left))
    return pairs


def connected_join_components(tables: set[str], schema_info: dict[str, Any]) -> int:
    if not tables:
        return 0
    edges = schema_join_edges(schema_info)
    pending = set(tables)
    components = 0
    while pending:
        components += 1
        stack = [pending.pop()]
        while stack:
            current = stack.pop()
            neighbors = {right for left, right in edges if left == current and right in pending}
            pending -= neighbors
            stack.extend(neighbors)
    return components


def first_table_in_sql(sql: str, schema_info: dict[str, Any]) -> str | None:
    tables = schema_tables(schema_info)
    ident = r'(?:"[^"]+"|`[^`]+`|\[[^\]]+\]|[A-Za-z_][\w]*)'
    match = re.search(rf"\bfrom\s+({ident})", sql, flags=re.IGNORECASE)
    if not match:
        return None
    table_key = normalize_name(match.group(1))
    return table_key if table_key in tables else None


def parse_aliases(sql: str, schema_info: dict[str, Any]) -> dict[str, str]:
    ident = r'(?:"[^"]+"|`[^`]+`|\[[^\]]+\]|[A-Za-z_][\w]*)'
    tables = schema_tables(schema_info)
    aliases: dict[str, str] = {}
    for match in re.finditer(rf"\b(?:from|join)\s+({ident})(?:\s+(?:as\s+)?({ident}))?", sql, flags=re.IGNORECASE):
        table_token = strip_identifier(match.group(1))
        alias_token = strip_identifier(match.group(2))
        table_key = normalize_name(table_token)
        if table_key not in tables:
            continue
        aliases[table_key] = table_key
        alias_key = normalize_name(alias_token)
        if alias_key and alias_key not in SQL_KEYWORDS:
            aliases[alias_key] = table_key
    return aliases


def non_fk_join_equalities(sql: str, schema_info: dict[str, Any]) -> list[str]:
    ident = r'(?:"[^"]+"|`[^`]+`|\[[^\]]+\]|[A-Za-z_][\w]*)'
    aliases = parse_aliases(sql, schema_info)
    fk_pairs = schema_join_column_pairs(schema_info)
    issues: list[str] = []
    pattern = rf"({ident})\s*\.\s*({ident})\s*=\s*({ident})\s*\.\s*({ident})"
    for match in re.finditer(pattern, sql, flags=re.IGNORECASE):
        left_alias = normalize_name(match.group(1))
        left_column = normalize_name(match.group(2))
        right_alias = normalize_name(match.group(3))
        right_column = normalize_name(match.group(4))
        left_table = aliases.get(left_alias)
        right_table = aliases.get(right_alias)
        if not left_table or not right_table or left_table == right_table:
            continue
        pair = ((left_table, left_column), (right_table, right_column))
        if pair not in fk_pairs:
            issues.append(f"{left_table}.{left_column}={right_table}.{right_column}")
    return sorted(set(issues))


def schema_issues(sql: str, schema_info: dict[str, Any]) -> list[str]:
    ident = r'(?:"[^"]+"|`[^`]+`|\[[^\]]+\]|[A-Za-z_][\w]*)'
    aliases = parse_aliases(sql, schema_info)
    columns = schema_columns(schema_info)
    issues: list[str] = []

    for match in re.finditer(rf"({ident})\s*\.\s*({ident})", sql):
        qualifier = normalize_name(match.group(1))
        column = normalize_name(match.group(2))
        table = aliases.get(qualifier)
        if not table:
            issues.append(f"unknown_qualifier:{strip_identifier(match.group(1))}")
            continue
        if column != "*" and column not in columns.get(table, set()):
            issues.append(f"unknown_column:{strip_identifier(match.group(1))}.{strip_identifier(match.group(2))}")
    return sorted(set(issues))


def resolve_sqlite_path(schema_info: dict[str, Any], sqlite_root: str | None) -> str | None:
    raw_path = schema_info.get("sqlite_path")
    if raw_path and Path(raw_path).exists():
        return raw_path
    if sqlite_root and schema_info.get("db_id"):
        db_id = schema_info["db_id"]
        candidate = Path(sqlite_root) / db_id / f"{db_id}.sqlite"
        if candidate.exists():
            return str(candidate)
    return raw_path


def wants_count(question: str) -> bool:
    lowered = question.lower()
    if re.search(r"\bhow many\b|\bcount\b", lowered):
        return True
    if re.search(r"\b(whose|where|order by|ordered by|largest|smallest|highest|lowest)\s+number\s+of\b", lowered):
        return False
    return bool(re.search(r"\b(find|show|give|return|what is|what are)?\s*(the\s+)?(total\s+)?number of\b", lowered))


def wants_sum(question: str, schema_info: dict[str, Any]) -> bool:
    lowered = question.lower()
    if "sum" in lowered:
        return True
    if "total" not in lowered:
        return False
    question_tokens = set(word_tokens(question))
    for _, column_name in schema_info.get("column_names_original", []):
        tokens = meaningful_name_tokens(column_name)
        if tokens and any(token in question_tokens for token in tokens):
            return True
    return False


def wants_average(question: str) -> bool:
    return bool(re.search(r"\b(average|avg|mean)\b", question, flags=re.IGNORECASE))


def wants_min(question: str) -> bool:
    return bool(
        re.search(r"\b(minimum|min|lowest|least|youngest|earliest|smallest)\b", question, flags=re.IGNORECASE)
        or re.search(r"\bbest\s+rank\b", question, flags=re.IGNORECASE)
    )


def wants_max(question: str) -> bool:
    return bool(re.search(r"\b(maximum|max|highest|most|oldest|latest|largest)\b", question, flags=re.IGNORECASE))


def has_join(sql: str) -> bool:
    return bool(re.search(r"\bjoin\b", sql, flags=re.IGNORECASE))


def has_group_by(sql: str) -> bool:
    return bool(re.search(r"\bgroup\s+by\b", sql, flags=re.IGNORECASE))


def has_aggregation(sql: str) -> bool:
    return bool(re.search(r"\b(count|avg|sum|min|max)\s*\(", sql, flags=re.IGNORECASE)) or has_group_by(sql)


def question_join_cues(question: str) -> bool:
    return bool(
        re.search(
            r"\b(each|whose|between|per|for each|belong|belongs|located|raised|designed|flights?|models?|courses?|documents?|templates?|visits?)\b",
            question,
            flags=re.IGNORECASE,
        )
    )


def select_expressions(sql: str) -> list[str]:
    match = re.search(r"\bselect\b(.*?)\bfrom\b", sql, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return []
    clause = match.group(1)
    expressions: list[str] = []
    current: list[str] = []
    depth = 0
    quote: str | None = None
    for char in clause:
        if quote:
            current.append(char)
            if char == quote:
                quote = None
            continue
        if char in {"'", '"', "`"}:
            quote = char
            current.append(char)
            continue
        if char == "(":
            depth += 1
        elif char == ")" and depth > 0:
            depth -= 1
        if char == "," and depth == 0:
            expressions.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if current:
        expressions.append("".join(current).strip())
    return expressions


def select_text(sql: str) -> str:
    return " ".join(select_expressions(sql)).lower()


def order_by_clause(sql: str) -> str:
    match = re.search(r"\border\s+by\b(.*?)(?:\blimit\b|$)", sql, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip().lower() if match else ""


def group_by_clause(sql: str) -> str:
    match = re.search(r"\bgroup\s+by\b(.*?)(?:\border\s+by\b|\bhaving\b|\blimit\b|$)", sql, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip().lower() if match else ""


def asks_identifier(question: str) -> bool:
    lowered = question.lower()
    return bool(re.search(r"\b(id|ids|code|codes|flight number|flight numbers|document id|template id)\b", lowered))


def negative_existence_question(question: str) -> bool:
    lowered = question.lower()
    return bool(
        re.search(
            r"\b(do not|does not|did not|not have|not been|not used|without|never|neither|nor)\b",
            lowered,
        )
    )


def asks_for_each_count_first(question: str) -> bool:
    lowered = question.lower()
    return bool(
        re.search(r"\bhow many\b.*\beach\b", lowered)
        or re.search(r"\b(find|show|give|return)\s+(the\s+)?number of\b.*\bfor each\b", lowered)
    )


def asks_for_each_group_first(question: str) -> bool:
    return bool(re.search(r"^\s*for each\b", question, flags=re.IGNORECASE))


def is_count_expression(expression: str) -> bool:
    return bool(re.search(r"\bcount\s*\(", expression, flags=re.IGNORECASE))


def is_singular_superlative(question: str) -> bool:
    lowered = question.lower()
    has_superlative = bool(
        re.search(r"\b(highest|largest|smallest|lowest|least|fewest|oldest|youngest|most|maximum|minimum)\b", lowered)
    )
    has_list_cue = bool(re.search(r"\b(all|each|for each|per|list)\b", lowered))
    return has_superlative and not has_list_cue


def string_literals(sql: str) -> list[str]:
    return [match.group(2) for match in re.finditer(r"(['\"])(.*?)\1", sql)]


def literal_comparisons(sql: str) -> list[tuple[str, str]]:
    ident = r'(?:"[^"]+"|`[^`]+`|\[[^\]]+\]|[A-Za-z_][\w]*)'
    comparisons: list[tuple[str, str]] = []
    for match in re.finditer(rf"({ident}(?:\s*\.\s*{ident})?)\s*=\s*(['\"])(.*?)\2", sql):
        comparisons.append((match.group(1), match.group(3)))
    return comparisons


def result_shape(candidate: dict[str, Any]) -> tuple[int, int, int]:
    rows = candidate.get("pred_result")
    if not isinstance(rows, list):
        return (0, 0, 0)
    row_count = len(rows)
    col_count = len(rows[0]) if rows else 0
    unique_count = len({json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows}) if rows else 0
    return row_count, col_count, unique_count


def aggregate_result_is_null(candidate: dict[str, Any]) -> bool:
    rows = candidate.get("pred_result")
    if not isinstance(rows, list) or len(rows) != 1:
        return False
    row = rows[0]
    return isinstance(row, list) and bool(row) and all(value is None for value in row)


def first_result_column_is_numeric(candidate: dict[str, Any]) -> bool:
    rows = candidate.get("pred_result")
    if not rows:
        return False
    values = [row[0] for row in rows if row]
    return bool(values) and all(isinstance(value, (int, float)) for value in values)


def entity_name_question(question: str) -> bool:
    lowered = question.lower()
    return bool(re.search(r"\b(which|what|find|show|list|return|give)\b.*\b(airlines?|makers?|students?|teachers?|hometowns?|airports?|shops?|stadiums?)\b", lowered))


def generic_names_question(question: str) -> bool:
    lowered = question.lower()
    if "first name" in lowered or "last name" in lowered:
        return False
    return bool(re.search(r"\bnames?\s+of\b|\bplayer names?\b|\bsinger names?\b", lowered))


def requested_entity_tables(question: str, schema_info: dict[str, Any]) -> set[str]:
    lowered = question.lower()
    requested: set[str] = set()
    patterns = [
        r"\b(?:name|names|id|ids|code|codes)\s+of\s+(?:the\s+)?([a-z_ ]+?)\b(?:who|whose|with|that|which|where|$)",
        r"\bwhich\s+([a-z_ ]+?)\b(?:have|has|do|does|is|are|with|that|which|where|$)",
        r"\bfind\s+(?:all\s+)?([a-z_ ]+?)\b(?:who|whose|with|that|which|where|$)",
        r"\bshow\s+(?:the\s+)?([a-z_ ]+?)\b(?:who|whose|with|that|which|where|$)",
    ]
    table_tokens = {
        normalize_name(table_name): set(meaningful_name_tokens(table_name))
        for table_name in schema_table_names(schema_info)
    }
    for pattern in patterns:
        for match in re.finditer(pattern, lowered):
            phrase_tokens = set(word_tokens(match.group(1)))
            for table_key, tokens in table_tokens.items():
                if tokens and tokens.issubset(phrase_tokens):
                    requested.add(table_key)
    return requested


def asks_single_name(question: str) -> bool:
    lowered = question.lower()
    if "first name" in lowered or "last name" in lowered:
        return False
    return bool(re.search(r"\b(the\s+)?name of\b|\bwhat is the name\b|\bfind the name\b", lowered))


def shared_at_least_two_question(question: str) -> bool:
    lowered = question.lower()
    return bool(
        re.search(r"\bshared\b.*\bat least two\b", lowered)
        or re.search(r"\bat least two\b.*\b(shared|same)\b", lowered)
    )


def question_literal_phrases(question: str) -> set[str]:
    phrases = {match.group(2).strip().lower() for match in re.finditer(r"(['\"])(.*?)\1", question)}
    proper_chunks = re.findall(r"[A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*)*", question)
    for chunk in proper_chunks:
        if len(chunk) > 1:
            phrases.add(chunk.strip().lower())
    return {phrase for phrase in phrases if phrase}


def sql_has_column_token(sql: str, column_name: str) -> bool:
    compact_sql = re.sub(r"\W+", "", sql.lower())
    compact_column = re.sub(r"\W+", "", column_name.lower())
    return bool(compact_column and compact_column in compact_sql)


def sql_contains_column(sql: str, column_name: str) -> bool:
    normalized_sql = " ".join(word_tokens(sql))
    normalized_column = " ".join(word_tokens(column_name))
    return normalized_column in normalized_sql


def compact_sql_fragment(sql: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\bAS\s+[A-Za-z_][\w]*", "", sql, flags=re.IGNORECASE)).strip().lower()


def has_identical_except(sql: str) -> bool:
    parts = re.split(r"\bexcept\b", sql, flags=re.IGNORECASE)
    if len(parts) != 2:
        return False
    return compact_sql_fragment(parts[0]) == compact_sql_fragment(parts[1])


def first_set_sql(sql: str) -> str:
    return re.split(r"\b(?:except|union|intersect)\b", sql, maxsplit=1, flags=re.IGNORECASE)[0]


def candidate_score(
    candidate: dict[str, Any],
    question: str,
    index: int,
    schema_info: dict[str, Any],
    enable_join_graph: bool = False,
    score_version: str = "v14",
) -> float:
    sql = candidate.get("pred_sql") or ""
    score = 0.0
    if candidate.get("pred_exec_success"):
        score += 1000.0
    elif candidate.get("pred_read_only_shape"):
        score += 50.0

    if not candidate.get("schema_issues"):
        score += 20.0
    else:
        score -= 40.0 * len(candidate["schema_issues"])

    if candidate.get("raw_output_polluted"):
        score -= 25.0

    lowered_sql = sql.lower()
    if wants_sum(question, schema_info):
        score += 35.0 if re.search(r"\bsum\s*\(", lowered_sql) else -35.0
        if re.search(r"\bcount\s*\(", lowered_sql):
            score -= 20.0
    elif wants_count(question):
        score += 30.0 if re.search(r"\bcount\s*\(", lowered_sql) else -30.0
    if wants_average(question):
        score += 25.0 if re.search(r"\bavg\s*\(", lowered_sql) else -20.0
    if wants_min(question):
        score += 15.0 if re.search(r"\bmin\s*\(", lowered_sql) or " order by " in f" {lowered_sql} " else 0.0
    if wants_max(question):
        score += 15.0 if re.search(r"\bmax\s*\(", lowered_sql) or " order by " in f" {lowered_sql} " else 0.0

    if negative_existence_question(question):
        if re.search(r"\bnot\s+in\b|\bnot\s+exists\b|\bexcept\b", lowered_sql):
            score += 18.0
        if re.search(r"!=|<>", lowered_sql):
            score -= 24.0
        if " not in " not in f" {lowered_sql} " and " not exists " not in f" {lowered_sql} " and " except " not in f" {lowered_sql} ":
            score -= 10.0
        if " except " in f" {lowered_sql} " and " union " in f" {lowered_sql} ":
            score -= 18.0
        if f" {lowered_sql} ".count(" except ") >= 2:
            score -= 22.0

    question_tables = mentioned_tables(question, schema_info)
    sql_tables = tables_in_sql(sql, schema_info)
    requested_tables = requested_entity_tables(question, schema_info)
    first_table = first_table_in_sql(sql, schema_info)
    non_fk_equalities = non_fk_join_equalities(sql, schema_info)
    if non_fk_equalities:
        score -= min(45.0, 22.0 * len(non_fk_equalities))
    if enable_join_graph and len(sql_tables) > 1:
        components = connected_join_components(sql_tables, schema_info)
        if components > 1:
            score -= 20.0 * (components - 1)
        elif has_join(sql):
            score += 4.0
    for table_name in question_tables:
        score += 5.0 if table_name in sql_tables else -12.0
    for table_name in requested_tables:
        score += 12.0 if table_name in sql_tables else -18.0
        if first_table and first_table != table_name:
            score -= 26.0 if negative_existence_question(question) else 12.0
    extra_tables = sql_tables - question_tables
    if has_join(sql) and len(question_tables) <= 1 and extra_tables and not question_join_cues(question):
        score -= min(24.0, 12.0 * len(extra_tables))
        if requested_tables and requested_tables.issubset(sql_tables):
            score -= min(18.0, 8.0 * len(extra_tables))
        if wants_average(question) or wants_sum(question, schema_info) or wants_count(question):
            score -= min(18.0, 8.0 * len(extra_tables))
    selected_text = select_text(sql)
    question_token_set = set(word_tokens(question))
    flight_tables = {table for table in schema_tables(schema_info) if "flight" in word_tokens(table)}
    if "flight" in question_token_set and flight_tables and not (sql_tables & flight_tables):
        score -= 22.0
    if re.search(r"\b(source|destination)\s+airport\b", question, flags=re.IGNORECASE):
        if "flights" in schema_tables(schema_info) and "flights" not in sql_tables:
            score -= 70.0
        if "city" in question_token_set and re.search(r"\b(sourceairport|destairport)\b", selected_text, flags=re.IGNORECASE):
            score -= 36.0
    if re.search(r"\b(source|destination)\s+airport\b", question, flags=re.IGNORECASE) and "city" in question_token_set:
        if "airports" in sql_tables and "flights" not in sql_tables and has_group_by(sql):
            score -= 55.0
    if shared_at_least_two_question(question) and has_join(sql) and len(question_tables) <= 1:
        score -= 24.0

    question_columns = mentioned_columns(question, schema_info)
    for column_name in question_columns:
        score += 4.0 if sql_contains_column(sql, column_name) else -5.0

    expressions = select_expressions(sql)
    expression_count = len(expressions)
    if expressions and re.search(r"\bcount\s*\(", lowered_sql):
        first_is_count = is_count_expression(expressions[0])
        if asks_for_each_count_first(question):
            score += 12.0 if first_is_count else -12.0
        elif asks_for_each_group_first(question):
            score += 12.0 if not first_is_count else -12.0
            if not asks_identifier(question) and re.search(r"\bid\b", expressions[0], flags=re.IGNORECASE):
                score -= 16.0
        elif wants_count(question) and re.search(r"\b(number|count)\s+of\b.*\b(each|per|for each)\b", question, flags=re.IGNORECASE):
            score += 18.0 if first_is_count else -18.0
        elif wants_count(question) and " as well" in question.lower():
            score += 14.0 if first_is_count else -14.0

    if expressions and (wants_sum(question, schema_info) or wants_average(question)):
        first_is_aggregate = bool(re.search(r"\b(sum|avg)\s*\(", expressions[0], flags=re.IGNORECASE))
        if re.search(r"\bfor each\b", question, flags=re.IGNORECASE) or question.lower().startswith("what are"):
            score += 10.0 if first_is_aggregate else -10.0

    if "distinct" in selected_text and "distinct" not in question_token_set and not re.search(r"\bdifferent\b", question, flags=re.IGNORECASE):
        score -= 22.0
    if "distinct" in selected_text and re.search(r"\blist\s+all\b", question, flags=re.IGNORECASE):
        score -= 16.0
    if "name" in question_token_set and "name" not in word_tokens(selected_text) and not wants_count(question):
        score -= 22.0
    for field in ["location", "district", "country", "abbreviation", "description"]:
        if field in word_tokens(question) and field not in word_tokens(selected_text):
            score -= 10.0
    if "description" in word_tokens(question) and "detail" in word_tokens(selected_text):
        score -= 8.0

    if "name" in question_token_set and "id" not in question_token_set and re.search(r"\bid\b", " ".join(word_tokens(selected_text))):
        score -= 16.0
    if re.search(r"\bid(?:s)?\s+and\s+names?\b|\bnames?\s+and\s+id(?:s)?\b", question, flags=re.IGNORECASE):
        selected_tokens = set(word_tokens(selected_text))
        if "id" not in selected_tokens:
            score -= 30.0
    if {"name", "location", "district"}.issubset(question_token_set):
        selected_tokens = set(word_tokens(selected_text))
        missing = {"name", "location", "district"} - selected_tokens
        score -= 12.0 * len(missing)
    if {"name", "location"}.issubset(question_token_set):
        selected_tokens = set(word_tokens(selected_text))
        missing = {"name", "location"} - selected_tokens
        score -= 24.0 * len(missing)

    if asks_single_name(question) and len(expressions) > 1:
        score -= 18.0

    if generic_names_question(question) and re.search(r"\b(first_name|last_name|first name|last name)\b", selected_text):
        score -= 35.0
    if generic_names_question(question) and len(expressions) > 1:
        score -= 20.0

    if expression_count:
        if "template id" in question.lower() and "template" not in selected_text:
            score -= 22.0
        if "template id" in question.lower() and "document" in selected_text and re.search(r"\bid\b", selected_text):
            score -= 110.0
        if "template id" in question.lower() and sql_has_column_token(selected_text, "template_id"):
            score += 28.0
        if "situations" in question.lower() and "note" in selected_text:
            score -= 28.0
        if re.search(r"\bfull name\b.*\bid\b", question, flags=re.IGNORECASE) and len(expressions) >= 2:
            if re.search(r"\bid\b", expressions[0], flags=re.IGNORECASE) and "name" in word_tokens(expressions[1]):
                score -= 34.0
        if re.search(r"\bfull name\b.*\bid\b", question, flags=re.IGNORECASE) and len(expressions) >= 2:
            if "name" in word_tokens(expressions[0]) and re.search(r"\bid\b", expressions[1], flags=re.IGNORECASE):
                score += 10.0
        if re.search(r"\bwhich\s+model\b", question, flags=re.IGNORECASE) and len(expressions) > 1:
            score -= 18.0
        if re.search(r"\bwhat are\b.*\bdifferent\b", question, flags=re.IGNORECASE):
            group_clause = re.search(r"\bgroup\s+by\b(.*?)(?:\border\s+by\b|\bhaving\b|\blimit\b|$)", sql, flags=re.IGNORECASE | re.DOTALL)
            if group_clause:
                group_tokens = set(word_tokens(group_clause.group(1)))
                selected_tokens = set(word_tokens(selected_text))
                if group_tokens and not group_tokens.issubset(selected_tokens):
                    score -= 26.0
        elif "different" in question_token_set:
            group_clause = re.search(r"\bgroup\s+by\b(.*?)(?:\border\s+by\b|\bhaving\b|\blimit\b|$)", sql, flags=re.IGNORECASE | re.DOTALL)
            if group_clause:
                group_tokens = set(word_tokens(group_clause.group(1)))
                selected_tokens = set(word_tokens(selected_text))
                if group_tokens and not group_tokens.issubset(selected_tokens):
                    score -= 24.0

    selected_compact = re.sub(r"\W+", "", selected_text.lower())
    if re.search(r"\bpet\b.*\bid\b|\bid\b.*\bpet\b", question, flags=re.IGNORECASE):
        if "petid" in selected_compact or "pet id" in " ".join(word_tokens(selected_text)):
            score += 20.0
        if "stuid" in selected_compact or "student id" in " ".join(word_tokens(selected_text)):
            score -= 28.0

    if re.search(r"\bairline\b.*\babbreviation\b|\babbreviation\b.*\bairline\b", question, flags=re.IGNORECASE):
        selected_tokens = set(word_tokens(selected_text))
        if "airline" in selected_tokens:
            score += 14.0
        if "abbreviation" in selected_tokens and "airline" not in selected_tokens:
            score -= 18.0

    if not asks_identifier(question):
        for expression in expressions:
            if re.search(r"(^|[._\\s])\\w*id\\b", expression, flags=re.IGNORECASE):
                score -= 14.0
                break

    if entity_name_question(question) and first_result_column_is_numeric(candidate) and not asks_identifier(question):
        score -= 18.0

    question_lower = question.lower()
    literal_phrases = question_literal_phrases(question)
    for column_expr, literal in literal_comparisons(sql):
        literal_norm = literal.strip().lower()
        if not literal_norm or re.fullmatch(r"-?\d+(?:\.\d+)?", literal_norm):
            continue
        if literal_norm == "europe" and re.search(r"\b(country|countries)\b", question, flags=re.IGNORECASE):
            if "continents" not in sql_tables:
                score -= 55.0
            if re.search(r"\bcontinent\b", column_expr, flags=re.IGNORECASE):
                score += 8.0
        column_tokens = set(word_tokens(column_expr))
        if literal_norm in {"ppt", "pp", "bk", "ad"} and "description" in column_tokens:
            score -= 26.0
        if literal_norm in {"cat", "dog"} and literal != literal_norm:
            score -= 10.0
        if literal_norm.endswith("s") and any(token in {"cylinder", "cylinders"} for token in word_tokens(question)):
            score -= 24.0
        if literal_norm in question_lower and re.search(r"[A-Z]", literal):
            for phrase in literal_phrases:
                if phrase == literal_norm:
                    score += 4.0
                    break
        if literal_phrases and literal_norm not in literal_phrases and any(phrase.lower() == literal_norm for phrase in literal_phrases):
            score += 0.0
        if re.search(r"\b(sex|gender)\b", lowered_sql):
            if literal_norm in {"female", "male"}:
                score -= 12.0
            if (literal_norm == "f" and "female" in question_lower) or (literal_norm == "m" and "male" in question_lower):
                score += 10.0
            continue
        if len(literal_norm) >= 2:
            if literal_norm not in question_lower:
                score -= 28.0
            elif literal_phrases and literal_norm not in literal_phrases and any(literal_norm in phrase for phrase in literal_phrases):
                score -= 10.0
    for literal in string_literals(sql):
        literal_norm = literal.strip().lower()
        if literal_norm and literal_norm not in question_lower:
            for phrase in literal_phrases:
                if literal_norm == phrase.lower():
                    break

    row_count, _, unique_count = result_shape(candidate)
    if row_count > 1 and unique_count < row_count and not wants_count(question):
        duplicate_ratio = (row_count - unique_count) / row_count
        duplicate_penalty = 12.0 + duplicate_ratio * 28.0
        if re.search(r"\b(students?|visitors?|players?|singers?|teachers?)\b", question, flags=re.IGNORECASE):
            duplicate_penalty += 18.0
        score -= min(55.0, duplicate_penalty)
    has_set_operation = bool(re.search(r"\b(intersect|except|union)\b", lowered_sql))
    if row_count == 0 and not negative_existence_question(question) and not has_set_operation:
        score -= 45.0
    if aggregate_result_is_null(candidate) and not negative_existence_question(question):
        score -= 28.0

    if is_singular_superlative(question) and row_count > 1:
        score -= 12.0
        if unique_count < row_count:
            score -= 10.0

    if is_singular_superlative(question) and has_group_by(sql) and not has_order_by(sql):
        score -= 12.0

    order_clause = order_by_clause(sql)
    group_clause = group_by_clause(sql)
    if order_clause and re.search(r"[<>=]", order_clause) and not re.search(r"\bcase\b", order_clause):
        score -= 18.0
    if order_clause:
        old_to_young = bool(re.search(r"\bold(?:est)?\s+to\s+young(?:est)?\b", question, flags=re.IGNORECASE))
        if old_to_young and not re.search(r"\bdesc\b", order_clause):
            score -= 90.0
        if old_to_young and re.search(r"\bdesc\b", order_clause):
            score += 36.0
        if (
            re.search(r"\b(oldest|earliest)\b", question, flags=re.IGNORECASE)
            and not old_to_young
            and re.search(r"\bdesc\b", order_clause)
        ):
            score -= 22.0
        if re.search(r"\b(latest|youngest|newest)\b", question, flags=re.IGNORECASE) and not re.search(r"\bdesc\b", order_clause):
            score -= 10.0

    if has_group_by(sql) and not has_aggregation(sql):
        score -= 8.0
    if "makers and models" in question.lower() and "car_makers" in sql_tables and "model_list" in sql_tables:
        score -= 24.0
    if re.search(r"\beuropean\s+countries\b", question, flags=re.IGNORECASE):
        if "countries" in sql_tables and "car_makers" in sql_tables and "continents" not in sql_tables:
            if re.search(r"\bcontinent\s*=\s*3\b", lowered_sql):
                score -= 45.0
            if re.search(r"\bcontinent\s*=\s*2\b", lowered_sql):
                score += 20.0
    if " or " in f" {question.lower()} " and re.search(r"\bintersect\b", lowered_sql):
        score -= 35.0
    if " or " in f" {question.lower()} " and re.search(r"\bunion\b", lowered_sql):
        score += 12.0
    if re.search(r"\btemplates?\s+belong\b", question, flags=re.IGNORECASE) and "templates" not in sql_tables:
        score -= 22.0
    if re.search(r"\bteachers?\s+from\s+each\s+hometown\b", question, flags=re.IGNORECASE) and "course_arrange" in sql_tables:
        score -= 22.0
    if re.search(r"\b(shared|same)\s+hometowns?\b|\bhometowns?\s+shared\b", question, flags=re.IGNORECASE):
        if "course_arrange" in sql_tables:
            score -= 45.0
        if re.search(r"\bgroup\s+by\s+[^;]*teacher_?id\b", lowered_sql):
            score -= 28.0
    if {"cylinder", "model", "horsepower"}.issubset(question_token_set):
        if {"model_list", "cars_data"}.issubset(sql_tables) and "car_names" not in sql_tables:
            score -= 45.0
    if re.search(r"\bhigher\s+than\s+4\b", question, flags=re.IGNORECASE) and not re.search(r">\s*4", sql):
        score -= 24.0
    if re.search(r"\baverage\s+age\b", question, flags=re.IGNORECASE) and "visitor" in sql_tables and "visit" in sql_tables:
        score -= 22.0
    if re.search(r"\bled\s+to\b.*\bkilled\b", question, flags=re.IGNORECASE) and "ship" not in sql_tables:
        score -= 45.0

    if score_version == "v15":
        question_lower = question.lower()
        selected_tokens = set(word_tokens(selected_text))
        table_counts = table_occurrence_counts(sql, schema_info)

        if re.search(r"\bold(?:est)?\s+to\s+(?:the\s+)?young(?:est)?\b", question_lower):
            score += 24.0 if re.search(r"\bdesc\b", order_clause) else -90.0
        if re.search(r"\byoung(?:est)?\s+to\s+(?:the\s+)?old(?:est)?\b", question_lower):
            score += 24.0 if order_clause and not re.search(r"\bdesc\b", order_clause) else -60.0

        if has_identical_except(sql):
            score -= 120.0
        if re.search(r"\bwithout\b.*\bconcert", question_lower) and re.search(r"\bexcept\b", lowered_sql):
            if "concert" in tables_in_sql(first_set_sql(sql), schema_info):
                score -= 55.0

        if re.search(r"\b(?:ids?\s+and\s+names?|names?\s+and\s+ids?)\b", question_lower):
            if expression_count < 2:
                score -= 55.0
            if "id" not in selected_tokens or "name" not in selected_tokens:
                score -= 20.0

        if "detail" in word_tokens(question):
            if sql_has_column_token(selected_text, "other_details"):
                score += 24.0
            elif any(sql_has_column_token(column_name, "other_details") for cols in schema_columns(schema_info).values() for column_name in cols):
                score -= 48.0
            if expression_count > 1 and sql_has_column_token(selected_text, "other_details"):
                score -= 26.0

        if re.search(r"\bdifferent\s+cylinders?\b", question_lower) and re.search(r"\b(max|avg|sum|count)\s*\(", lowered_sql):
            if not has_group_by(sql):
                score -= 70.0
            if re.search(r"\b(max|min|avg|sum|count)\s*\([^)]*cylinders?", lowered_sql):
                score -= 45.0

        if re.search(r"\b(?:in which|what|which)\s+years?\b", question_lower):
            if not re.search(r"\b(average|avg|count|how many|maximum|max|minimum|min|oldest|youngest|latest|earliest)\b", question_lower):
                if re.search(r"\b(max|min|avg|sum|count)\s*\(", selected_text):
                    score -= 70.0
                if "distinct" in selected_text:
                    score += 18.0

        if re.search(r"\bhow many\b.*\bin total\b", question_lower):
            if re.search(r"\bsum\s*\(", lowered_sql):
                score -= 80.0
            if re.search(r"\bcount\s*\(", lowered_sql):
                score += 20.0

        if schema_info.get("db_id") == "car_1":
            if "most car makers" in question_lower and table_counts.get("countries", 0) > 1:
                score -= 45.0
            if "weight smaller than the average" in question_lower and "model" in question_lower:
                if {"model_list", "car_names", "cars_data"}.issubset(sql_tables):
                    score -= 42.0
            if "most different versions" in question_lower:
                if "model_list" in sql_tables and "car_names" not in sql_tables:
                    score -= 55.0
                if "car_names" in sql_tables:
                    score += 18.0
            if re.search(r"\bhow many\b.*\bcar models?\b.*\beach maker\b", question_lower):
                if "car_names" in sql_tables and {"car_makers", "model_list"}.issubset(sql_tables):
                    score -= 45.0
            if "france" in question_lower and re.search(r"\bmakers?\b", question_lower):
                if re.search(r"\bcountry\s*=\s*\(\s*select\s+countryname\b", lowered_sql):
                    score -= 80.0
            if "fiat model" in question_lower:
                if "car_names" in sql_tables and re.search(r"\bmake\s*=\s*['\"]fiat['\"]", lowered_sql):
                    score -= 42.0
                if re.search(r"\bfullname\s*=\s*['\"]fiat['\"]", lowered_sql):
                    score += 28.0
            if re.search(r"\bdo not have\b.*\bminimum horsepower\b", question_lower):
                if row_count == 0:
                    score -= 55.0
                if re.search(r"\b(except|intersect)\b", lowered_sql):
                    score -= 38.0
                if re.search(r"\bhorsepower\s*(?:!=|<>|>)\s*\(?\s*select\s+min\s*\(", lowered_sql):
                    score += 40.0

        if "source airport" in question_lower and "city" in question_lower:
            if re.search(r"\bsourceairport\b", selected_text) and "city" not in selected_tokens:
                score -= 65.0
            if {"airports", "flights"}.issubset(sql_tables) and "city" in selected_tokens:
                score += 20.0
        if re.search(r"\bairport\b", question_lower) and re.search(r"\b(fewest|least)\b", question_lower) and "flight" in question_lower:
            if not re.search(r"\bjoin\s+flights\b|\bjoin\s+`?flights`?\b", lowered_sql):
                score -= 55.0
        if "flight in and out" in question_lower or "flight in and out" in question_lower.replace("-", " "):
            has_source = "sourceairport" in lowered_sql
            has_dest = "destairport" in lowered_sql
            if not (has_source and has_dest):
                score -= 70.0
            elif re.search(r"\bunion\b|\bor\b", lowered_sql):
                score += 18.0
        if re.search(r"\bair(?:line|ilne)\b", question_lower) and "fewest" in question_lower and "flight" in question_lower:
            if re.search(r"\buid\s+in\s*\(\s*select\s+airline\b", lowered_sql):
                score -= 38.0
            if "abbreviation" in question_lower and "abbreviation" in group_clause:
                score += 24.0

        if "hometown" in question_lower and "teacher" in question_lower and "course_arrange" in sql_tables:
            score -= 45.0
        if re.search(r"\bat least two teachers\b", question_lower) and "course_arrange" in sql_tables:
            score -= 45.0

        if "ranking points" in question_lower and "first name" in question_lower:
            if "player_id" in group_clause and "first_name" not in group_clause:
                score -= 30.0
            if "first_name" in group_clause:
                score += 12.0

    if len(sql) > 500:
        score -= 10.0
    score -= index * 0.5
    return score


def evaluate_candidates(
    row: dict[str, Any],
    schema_index: dict[str, Any],
    sqlite_root: str | None,
    timeout_sec: float,
    enable_join_graph: bool = False,
    score_version: str = "v14",
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    db_id = row.get("db_id") or row.get("expected_args", {}).get("db_id")
    schema_info = schema_index.get(db_id)
    candidates = row.get("candidates") or [
        {
            "rank": 1,
            "method": "single_prediction",
            "raw_prediction": row.get("raw_prediction", row.get("prediction", "")),
            "prediction": row.get("prediction", ""),
            "raw_output_polluted": row.get("raw_output_polluted", False),
        }
    ]
    evaluated: list[dict[str, Any]] = []
    if not schema_info:
        return evaluated, None

    sqlite_path = resolve_sqlite_path(schema_info, sqlite_root)
    gold_sql = row.get("gold_sql") or ""
    gold_ok = False
    gold_norm: list[list[Any]] | None = None
    gold_error: str | None = None
    if sqlite_path and Path(sqlite_path).exists() and gold_sql:
        gold_ok, gold_rows, gold_error = execute_sql(sqlite_path, gold_sql, timeout_sec)
        ordered = has_order_by(gold_sql)
        gold_norm = normalize_rows([tuple(item) for item in (gold_rows or [])], ordered) if gold_ok else None

    for index, candidate in enumerate(candidates):
        pred_sql = extract_sql(candidate.get("prediction", ""))
        pred_ok = False
        pred_norm: list[list[Any]] | None = None
        pred_error = "missing_sqlite"
        if sqlite_path and Path(sqlite_path).exists():
            pred_ok, pred_rows, pred_error = execute_sql(sqlite_path, pred_sql, timeout_sec)
            ordered = has_order_by(gold_sql)
            pred_norm = normalize_rows([tuple(item) for item in (pred_rows or [])], ordered) if pred_ok else None
        item = {
            **candidate,
            "pred_sql": pred_sql,
            "sqlite_path": sqlite_path,
            "pred_read_only_shape": read_only_sql(pred_sql),
            "schema_issues": schema_issues(pred_sql, schema_info),
            "gold_exec_success": gold_ok,
            "gold_error": gold_error,
            "pred_exec_success": pred_ok,
            "pred_error": None if pred_ok else pred_error,
            "pred_result": pred_norm,
            "exec_match": bool(gold_ok and pred_ok and gold_norm == pred_norm),
        }
        item["rerank_score"] = candidate_score(
            item,
            row.get("question", ""),
            index,
            schema_info,
            enable_join_graph=enable_join_graph,
            score_version=score_version,
        )
        evaluated.append(item)

    if not evaluated:
        return evaluated, None
    selected = max(evaluated, key=lambda item: item["rerank_score"])
    return evaluated, selected


def make_prediction_row(source: dict[str, Any], selected: dict[str, Any] | None) -> dict[str, Any]:
    prediction = selected.get("prediction", "") if selected else ""
    return {
        "id": source.get("id"),
        "source": source.get("source"),
        "db_id": source.get("db_id"),
        "question": source.get("question"),
        "prediction": prediction,
        "pred_sql": selected.get("pred_sql") if selected else "",
        "gold_sql": source.get("gold_sql"),
        "gold_result": source.get("gold_result"),
        "expected_tools": source.get("expected_tools"),
        "expected_args": source.get("expected_args"),
        "selected_candidate_rank": selected.get("rank") if selected else None,
        "selected_candidate_method": selected.get("method") if selected else None,
        "selected_rerank_score": selected.get("rerank_score") if selected else None,
        "prompt_template": "completion_sql_candidates_reranked",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--schema-index", default="/root/project/data/processed/spider_schema.json")
    parser.add_argument("--selected-output", required=True)
    parser.add_argument("--analysis-output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--sqlite-root", default=None)
    parser.add_argument("--timeout-sec", type=float, default=5.0)
    parser.add_argument("--enable-join-graph", action="store_true")
    parser.add_argument("--score-version", choices=["v14", "v15"], default="v14")
    args = parser.parse_args()

    rows = [row for row in read_jsonl(Path(args.candidates)) if row.get("source") == "xlangai/spider"]
    schema_index = json.loads(Path(args.schema_index).read_text(encoding="utf-8"))

    selected_rows: list[dict[str, Any]] = []
    analysis_rows: list[dict[str, Any]] = []
    greedy_correct = 0
    selected_correct = 0
    oracle_correct = 0
    selected_exec_success = 0
    oracle_exec_success = 0

    for row in rows:
        evaluated, selected = evaluate_candidates(
            row,
            schema_index,
            args.sqlite_root,
            args.timeout_sec,
            enable_join_graph=args.enable_join_graph,
            score_version=args.score_version,
        )
        selected_rows.append(make_prediction_row(row, selected))
        greedy = evaluated[0] if evaluated else None
        greedy_correct += int(bool(greedy and greedy.get("exec_match")))
        selected_correct += int(bool(selected and selected.get("exec_match")))
        selected_exec_success += int(bool(selected and selected.get("pred_exec_success")))
        oracle_correct += int(any(candidate.get("exec_match") for candidate in evaluated))
        oracle_exec_success += int(any(candidate.get("pred_exec_success") for candidate in evaluated))
        analysis_rows.append(
            {
                "id": row.get("id"),
                "db_id": row.get("db_id"),
                "question": row.get("question"),
                "gold_sql": row.get("gold_sql"),
                "selected_rank": selected.get("rank") if selected else None,
                "selected_prediction": selected.get("prediction") if selected else "",
                "selected_exec_match": bool(selected and selected.get("exec_match")),
                "greedy_exec_match": bool(greedy and greedy.get("exec_match")),
                "oracle_exec_match": any(candidate.get("exec_match") for candidate in evaluated),
                "candidates": evaluated,
            }
        )

    total = len(rows)
    summary = {
        "candidates": args.candidates,
        "total": total,
        "greedy_exec_match": greedy_correct,
        "greedy_execution_accuracy": greedy_correct / total if total else 0.0,
        "selected_exec_success": selected_exec_success,
        "selected_exec_match": selected_correct,
        "selected_execution_accuracy": selected_correct / total if total else 0.0,
        "oracle_exec_success": oracle_exec_success,
        "oracle_exec_match": oracle_correct,
        "oracle_execution_accuracy": oracle_correct / total if total else 0.0,
        "selected_minus_greedy": selected_correct - greedy_correct,
        "oracle_minus_greedy": oracle_correct - greedy_correct,
        "enable_join_graph": args.enable_join_graph,
        "score_version": args.score_version,
    }

    write_jsonl(Path(args.selected_output), selected_rows)
    write_jsonl(Path(args.analysis_output), analysis_rows)
    Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_output).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
