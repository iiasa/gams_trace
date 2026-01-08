#!/usr/bin/env python3

import argparse
import os
import pickle
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union

class IncludeError(Exception):
    def __init__(self, message: str, include_file: str, include_loc: Optional[Tuple[str, int, str]]):
        self.include_file = include_file
        self.include_loc = include_loc
        if include_loc:
            include_type = '$' + include_loc[2]
            super().__init__(f"{message}\n  {include_type} statement at {include_loc[0]}:{include_loc[1]}")
        else:
            super().__init__(message)

REGION = "REGION59"

# ----------------------------
# Data structures
# ----------------------------

@dataclass
class SourceLoc:
    file: str
    line: int

@dataclass
class Definition:
    kind: str  # 'assignment'|'table'|'declaration'|'equation'|'gdx_load'
    text: str
    loc: SourceLoc
    deps: Set[str] = field(default_factory=set)  # symbols referenced
    lhs: Optional[str] = None
    values: Dict[Tuple[str, ...], float] = field(default_factory=dict)  # for tables
    skipped: bool = False  # for large tables, parsing skipped for performance
    gdx_file: Optional[str] = None  # for GDX-loaded symbols

@dataclass
class SymbolInfo:
    original: str  # original name with case
    stype: str  # 'set','parameter','scalar','table','variable','equation','model'
    decls: List[Definition] = field(default_factory=list)
    defs: List[Definition] = field(default_factory=list)
    dims: List[str] = field(default_factory=list)
    csv_file: Optional[str] = None  # for sets/tables loaded from CSV via $ondelim
    vtype: Optional[str] = None  # Variable type: POSITIVE, FREE, BINARY, etc. (for variables only)
    base_set: Optional[str] = None  # for set aliases, the base set name

@dataclass
class ModelInfo:
    name: str
    equations: List[str]
    loc: SourceLoc

@dataclass
class SolveInfo:
    model: str
    solver: str
    sense: str  # minimizing|maximizing
    objvar: str
    loc: SourceLoc

@dataclass
class LineEntry:
    text: str
    file: str
    line: int

# ----------------------------
# Utility helpers
# ----------------------------

IDENT_RE = re.compile(r"\b([A-Za-z_]\w*)\b")
BUILTINS = {"sum","smin","smax","min","max","ord","card","power","exp","log","abs",
            "uniform","normal","floor","ceil","round","yes","no"}
DECL_START_RE = re.compile(r"^\s*(sets?|parameters?|scalars?|tables?|variables?|equations?|free|positive|negative|binary|integer|semicontinuous|semicont|semiinteger|semiint|sos1|sos2)\b", re.IGNORECASE)
DECL_LINE_RE = re.compile(r'^\s*([A-Za-z_]\w*)\s*\(\s*([^)]*)\s*\)?', re.IGNORECASE)
INCLUDE_RE = re.compile(r"^\s*\$(bat)?include\s+(.+)", re.IGNORECASE | re.DOTALL)
GDXIN_RE = re.compile(r"^\s*\$gdxin\s+(.+)", re.IGNORECASE | re.DOTALL)
LOAD_RE = re.compile(r"^\s*\$load\s+(.+)", re.IGNORECASE | re.DOTALL)
LOADDC_RE = re.compile(r"^\s*\$loaddc\s+(.+)", re.IGNORECASE | re.DOTALL)
SOLVE_RE = re.compile(r"^\s*solve\s+(\w+)\s+using\s+(\w+)\s+(minimizing|minimising|maximizing|maximising)\s+(\w+)", re.IGNORECASE | re.DOTALL)
MODEL_RE = re.compile(r"model\s+(\w+)\s*/\s*([^/]*)/\s*;", re.IGNORECASE | re.DOTALL)
ASSIGN_RE = re.compile(r"^\s*([A-Za-z_]\w*)(\s*\([^)]*\))?\s*=\s*(.+?);\s*$", re.IGNORECASE | re.DOTALL)
EQUATION_RE = re.compile(r"^\s*([A-Za-z_]\w*\s*(?:\([^)]*\))?)\s*(?:\$(.+?))?\s*\.\.\s*(.+?)\s*=(l|L|e|E|g|G)=\s*(.+?);\s*$", re.IGNORECASE | re.DOTALL)
VAR_DECL_RE = re.compile(r"^\s*((negative|positive|free|binary|integer|semicontinuous|semicont|semiinteger|semiint|sos1|sos2)(?:\([^)]*\))?\s+)?\s*variables?\s+(.+);", re.IGNORECASE)
MULTI_VAR_DECL_RE = re.compile(r"^\s*((negative|positive|free|binary|integer|semicontinuous|semicont|semiinteger|semiint|sos1|sos2)(?:\([^)]*\))?\s+)?\s*variables?\s*(.*?);", re.IGNORECASE | re.DOTALL)
TABLE_HEAD_RE = re.compile(r"^\s*tables?\s+([A-Za-z_]\w*)\s*\(([^)]*)\)\s*", re.IGNORECASE | re.DOTALL)
ALIAS_RE = re.compile(r"^\s*alias\s*\(\s*([a-zA-Z_]\w*)\s*,\s*(.+?)\s*\)\s*;?\s*$", re.IGNORECASE | re.DOTALL)

def find_idents_with_aliases(expr: str, aliases: Dict[str, str]) -> Set[str]:
    ids = set()
    for m in IDENT_RE.finditer(expr):
        token = m.group(1)
        if token.isnumeric():
            continue
        if token.lower() in BUILTINS:
            continue
        resolved = aliases.get(token.lower(), token.lower())
        ids.add(resolved)
    return ids


def norm_ident(s: str) -> str:
    return s.strip()


def find_idents(expr: str) -> Set[str]:
    ids = set()
    for m in IDENT_RE.finditer(expr):
        token = m.group(1)
        if token.isnumeric():
            continue
        if token.lower() in BUILTINS:
            continue
        ids.add(token.lower())
    return ids

# ----------------------------
# Loader: read root and resolve includes
# ----------------------------


def load_gms(root_path: str) -> List[LineEntry]:
    """Return list of line entries with original file and line number in merged include order."""
    merged_lines: List[LineEntry] = []
    base_dir = os.path.dirname(os.path.abspath(root_path))

    def load_file(fp: str, depth: int = 0) -> None:
        if depth > 100:
            raise IncludeError(f"Recursion depth exceeded: {fp}", include_file=fp, include_loc=None)

        full = os.path.abspath(fp)
        status_msg = f"Loading: {os.path.basename(full)} ({len(merged_lines)} lines)                "
        print(f"\r{status_msg}", end="", flush=True)

        try:
            with open(full, 'r', encoding='utf-8', errors='ignore') as f:
                raw_lines = f.readlines()
        except FileNotFoundError:
            raise IncludeError(f"Included file not found: {fp}", include_file=fp, include_loc=None)
        # Strip comments, but keep empty lines for line number tracking
        lines = []
        in_comment = False
        for raw_line in raw_lines:
            stripped = raw_line.strip().lower()
            if stripped == '$ontext':
                in_comment = True
            elif stripped == '$offtext':
                in_comment = False
            if raw_line.strip().startswith('*') or in_comment or stripped in ['$ontext', '$offtext']:
                lines.append('')  # Empty line to preserve line numbers
            else:
                lines.append(raw_line.rstrip('\n'))

        # Process lines, handling includes inline and merging equation lines
        line_num = 1
        i = 0
        equation_accumulating = False
        equation_buff = []
        equation_line_num = 1  # Placeholder
        while i < len(lines):
            line = lines[i]
            if not line:  # Empty line (comment)
                # Do not include comment lines in merged output
                pass
            else:
                m = INCLUDE_RE.match(line)
                if m:
                    rest = m.group(2).strip()
                    path_match = re.match(r'''^\s*(?:"([^"]+)"|'([^']+)'|(\S+))''', rest)
                    if path_match:
                        path_str = path_match.group(0)
                        inc_path = next(g for g in path_match.groups() if g is not None)
                        args_start = len(path_str)
                    else:
                        inc_path = rest.split()[0]
                        args_start = len(inc_path)
                    inc_path = inc_path.strip('"\'').replace('%X%', '/').replace('%REGION%', REGION)
                    if inc_path.lower().endswith('.csv'):
                        # Add the $include line for CSV (not processed inline)
                        merged_lines.append(LineEntry(text=line, file=full, line=line_num))
                    else:
                        is_bat = m.group(1) is not None
                        if is_bat:
                            # Extract args after the path
                            after_path = rest[args_start:].strip()
                            args = after_path.split() if after_path else []
                        else:
                            args = []
                        inc_full = os.path.join(base_dir, inc_path)
                        # Recursively load and inline the included file's content
                        start_len = len(merged_lines)
                        load_file(inc_full, depth + 1)
                        # Apply substitutions to the newly added lines from sub-load
                        if args:
                            for j in range(start_len, len(merged_lines)):
                                for idx, arg in enumerate(args, 1):
                                    merged_lines[j].text = merged_lines[j].text.replace(f'%{idx}%', arg)
                else:
                    if not equation_accumulating:
                        if '..' in line:
                            # Check if previous merged entry is potential equation name (starts with word, no ;)
                            if (merged_lines and
                                re.match(r'^\s*[A-Za-z]', merged_lines[-1].text) and
                                ';' not in merged_lines[-1].text):
                                name_entry = merged_lines.pop()
                                equation_buff = [name_entry.text, line]
                                equation_line_num = name_entry.line
                            else:
                                equation_buff = [line]
                                equation_line_num = line_num
                            equation_accumulating = True
                        else:
                            merged_lines.append(LineEntry(text=line, file=full, line=line_num))
                    else:
                        # Accumulating an equation
                        equation_buff.append(line)
                        if line.strip().endswith(';'):
                            # End of equation
                            merged_text = '\n'.join(equation_buff)
                            merged_lines.append(LineEntry(text=merged_text, file=full, line=equation_line_num))
                            equation_accumulating = False
                            equation_buff = []
                            equation_line_num = 1  # Reset
            line_num += 1
            i += 1
            # If end of file and still accumulating, close it (though equations should end with ;)
            if i == len(lines) and equation_accumulating:
                merged_text = '\n'.join(equation_buff)
                merged_lines.append(LineEntry(text=merged_text, file=full, line=equation_line_num))
                equation_accumulating = False
                equation_buff = []

    load_file(root_path, 0)
    print(f"\rLoaded {len(merged_lines)} lines from included files.{'                                    '}", flush=True)
    return merged_lines

# ----------------------------
# Parser
# ----------------------------

NON_ASSIGNABLE_KEYWORDS = frozenset({
    'if', 'else', 'while', 'loop', 'for', 'repeat', 'solve', 'model',
    'set', 'sets', 'parameter', 'parameters', 'scalar', 'scalars', 'table', 'tables',
    'variable', 'variables', 'equation', 'equations', 'function',
    'abort', 'display', 'option', 'options', 'execute', 'put', 'file', 'error', 'system', 'call',
    'gdxin', 'gdout', 'load', 'unload', 'include', 'batinclude', 'ontext', 'offtext',
    'free', 'positive', 'negative', 'binary', 'integer', 'semicontinuous', 'semicont',
    'semiinteger', 'semiint', 'sos1', 'sos2'
})

def parse_code(entries: List[LineEntry]) -> Tuple[Dict[str, SymbolInfo], List[ModelInfo], List[SolveInfo], Dict[str, str]]:
    symbols: Dict[str, SymbolInfo] = {}
    models: List[ModelInfo] = []
    solves: List[SolveInfo] = []
    aliases: Dict[str, str] = {}

    def ensure_symbol(name: str, stype: str) -> SymbolInfo:
        original = name
        name_lower = name.lower()
        if name_lower not in symbols:
            symbols[name_lower] = SymbolInfo(original=original, stype=stype)
        else:
            # upgrade stype if unknown
            if symbols[name_lower].stype == "unknown":
                symbols[name_lower].stype = stype
        return symbols[name_lower]

    status_msg = f"Parsing merged code ({len(entries)} lines)"
    print(f"\r{status_msg}", end="", flush=True)
    current_gdx_file = None
    i = 0
    while i < len(entries):
        if i % 1000 == 0 or i == 0:
            status_msg = f"Parsing merged code ({len(entries)} lines): {i}/{len(entries)}"
            print(f"\r{status_msg}", end="", flush=True)
        line = entries[i].text
        if not line.strip():
            i += 1
            continue
        # Merge continuation lines if trailing comma and next line
        # (simple heuristic)
        if i + 1 < len(entries) and entries[i+1].text.lstrip().startswith(','):
            line += ' ' + entries[i+1].text.strip()
            i += 1

        # GDX input detection
        gm = GDXIN_RE.match(line)
        if gm:
            current_gdx_file = gm.group(1).strip().strip('"\'')
            i += 1
            continue

        # Load detection
        lm = LOAD_RE.match(line) or LOADDC_RE.match(line)
        if lm:
            symbols_list = [s.strip() for s in lm.group(1).strip().split(',') if s.strip()]
            for sym_name in symbols_list:
                sym = ensure_symbol(sym_name, 'unknown')
                sym.defs.append(Definition(kind='gdx_load', text=line, loc=SourceLoc(entries[i].file, entries[i].line), deps=set(), lhs=sym_name.lower(), gdx_file=current_gdx_file))
            i += 1
            continue

        # Alias parsing
        am = ALIAS_RE.match(line)
        if am:
            base = am.group(1).strip()
            aliases_str = am.group(2)
            alias_list = [a.strip().lower() for a in aliases_str.split(',') if a.strip()]
            base_lower = base.lower()
            for a in alias_list:
                sym = ensure_symbol(a.lower(), 'set')
                sym.base_set = base_lower
                sym.decls.append(Definition(kind='alias_declaration', text=line, loc=SourceLoc(entries[i].file, entries[i].line)))
            i += 1
            continue

        # Declarations
        if DECL_START_RE.match(line):
            # Variables special case
            vdm = VAR_DECL_RE.match(line)
            if vdm and vdm.group(3).strip():
                if '\n' in vdm.group(3):
                    # Multi-line declaration, skip single-line parsing
                    pass
                else:
                    # Single-line variable declaration
                    var_type = vdm.group(2).upper() if vdm.group(2) else "POSITIVE"
                    var_list = vdm.group(3)
                    for v in re.split(r",", var_list):
                        vname = v.strip()
                        if not vname:
                            continue
                        m = IDENT_RE.search(vname)
                        if not m:
                            continue
                        name = m.group(1)
                        sym = ensure_symbol(name, 'variable')
                        if sym.vtype is None:
                            sym.vtype = var_type
                        sym.decls.append(Definition(kind='declaration', text=line, loc=SourceLoc(entries[i].file, entries[i].line)))
                    i += 1
                    continue
            # Check for multi-line variable declaration
            if 'variables' in line.lower() and not line.strip().endswith(';'):
                accumulated = line
                j = i + 1
                while j < len(entries):
                    next_line = entries[j].text
                    if not next_line.strip():
                        j += 1
                        continue
                    accumulated += '\n' + next_line.rstrip('\n')
                    if next_line.strip().endswith(';'):
                        # Parse the full variable declaration, handling variables listed on separate lines with descriptions
                        var_match = MULTI_VAR_DECL_RE.search(accumulated)
                        if var_match and var_match.group(3):
                            var_type = var_match.group(2).upper() if var_match.group(2) else "POSITIVE"
                            vars_part = var_match.group(3).strip()
                            lines = vars_part.split('\n')
                            for line_text in lines:
                                line_text = line_text.strip()
                                if line_text and not line_text.startswith('*'):
                                    # Take first token as variable name
                                    parts = line_text.split()
                                    if parts:
                                        vname_str = parts[0]
                                        m = IDENT_RE.search(vname_str)
                                        if m:
                                            name = m.group(1)
                                            sym = ensure_symbol(name, 'variable')
                                            if sym.vtype is None:
                                                sym.vtype = var_type
                                            sym.decls.append(Definition(kind='declaration', text=accumulated, loc=SourceLoc(entries[j].file, entries[j].line)))
                        i = j
                        break
                    j += 1
                i += 1
                continue
            # Tables block
            th = TABLE_HEAD_RE.match(line)
            if th:
                tname = th.group(1)
                dims = [d.strip() for d in th.group(2).split(',')]
                sym = ensure_symbol(tname, 'table')
                sym.dims = dims
                # Read subsequent lines until a lone ';' or next declaration
                j = i + 1
                table_lines: List[str] = []
                while j < len(entries):
                    l2 = entries[j].text
                    if not l2.strip():
                        j += 1
                        continue
                    if ';' in l2:
                        table_lines.append(l2)
                        j += 1
                        break
                    if DECL_START_RE.match(l2) or INCLUDE_RE.match(l2) or SOLVE_RE.search(l2) or MODEL_RE.search(l2):
                        break
                    table_lines.append(l2)
                    j += 1
                # Check if data is from CSV
                if any('$ondelim' in tl.lower() for tl in table_lines) and any('$include' in tl.lower() and '.csv' in tl.lower() for tl in table_lines):
                    for tl in table_lines:
                        if '$include' in tl.lower():
                            imm = INCLUDE_RE.match(tl.strip())
                            if imm:
                                rest = imm.group(2).strip().strip('"\'').replace('%X%', '/').replace('%REGION%', REGION)
                                if rest.lower().endswith('.csv'):
                                    sym.csv_file = rest
                                    break
                # Parse only small tables line by line; for large tables, skip to avoid performance issues
                if len(table_lines) > 100:
                    values: Dict[Tuple[str, ...], float] = {}
                    skipped = True
                else:
                    values = {}
                    skipped = False
                    # Parse a simple table: header row with column keys, then rows as key + numbers
                    # This is heuristic and works for common rectangular numeric tables.
                    header_cols: List[str] = []
                    values = {}
                    # Find first non-empty line as header
                    for idx, tl in enumerate(table_lines):
                        if tl.strip() and not tl.strip().endswith(':') and not tl.strip() == ';':
                            header_line = tl
                            header_cols = [c.strip() for c in re.split(r"\s+", header_line.strip()) if c.strip()]
                            start_row = i + 1 + idx + 1
                            break
                    else:
                        header_cols = []
                        start_row = i + 1
                    # Parse rows
                    r = start_row
                    while r < i + 1 + len(table_lines):
                        row_line = entries[r].text
                        if not row_line.strip() or row_line.strip() == ';':
                            r += 1
                            continue
                        parts = [p for p in re.split(r"\s+", row_line.strip()) if p]
                        if not parts:
                            r += 1
                            continue
                        row_key = parts[0]
                        for k, col in enumerate(header_cols[1:], start=1):
                            if k >= len(parts):
                                continue
                            try:
                                val = float(parts[k])
                                key_tuple = (row_key, col)
                                values[key_tuple] = val
                            except ValueError:
                                pass
                        r += 1
                defn = Definition(kind='table', text=line, loc=SourceLoc(entries[i].file, entries[i].line), deps=set(), values=values, skipped=skipped)
                sym.defs.append(defn)
                i = j
                continue
            # Other declarations (Sets, Parameters, Scalars, Equations)
            # We record the line; detailed parsing of dimensions is skipped.
            first_word = line.strip().split()[0].lower()
            stype_map = {
                'set': 'set', 'sets': 'set',
                'parameter': 'parameter', 'parameters': 'parameter',
                'scalar': 'scalar', 'scalars': 'scalar',
                'equation': 'equation', 'equations': 'equation',
                'variables': 'variable'
            }
            stype = stype_map.get(first_word, 'unknown')
            # Extract names; handle differently for sets (not comma-separated) vs parameters/scalars
            decl_body = line
            decl_parts = decl_body.split(None, 1)
            ondelim_found = False
            csv_found = False
            csv_path = None
            names = []
            if stype == 'set':
                if len(decl_parts) > 1:
                    name_part = decl_parts[1].strip()
                    # Match name optionally followed by (dims...)
                    match = re.match(r'([A-Za-z_]\w*)(?:\s*\(([^)]*)\))?', name_part)
                    if match:
                        name = match.group(1)
                        dims_str = match.group(2)
                        dims = []
                        if dims_str:
                            dims = [d.strip() for d in dims_str.split(',') if d.strip()]
                        sym = ensure_symbol(name, stype)
                        sym.dims = dims
                        sym.decls.append(Definition(kind='declaration', text=line, loc=SourceLoc(entries[i].file, entries[i].line)))
                        names = [name]  # for multi-line compatibility
                        # For sets, check if data is loaded from CSV
                        k = i
                        ondelim_found2 = False
                        while k < len(entries):
                            k += 1
                            l3 = entries[k].text.strip().lower() if k < len(entries) else ''
                            if l3 == '$offdelim':
                                break
                            if l3 == '$ondelim':
                                ondelim_found2 = True
                            if ondelim_found2 and l3.startswith('$include'):
                                inc_m2 = INCLUDE_RE.match(l3)
                                if inc_m2:
                                    rest2 = inc_m2.group(2).strip().strip('"\'').replace('%X%', '/').replace('%REGION__', REGION)
                                    if rest2.lower().endswith('.csv'):
                                        csv_path = rest2
                                        csv_found = True
                        if csv_found:
                            sym.csv_file = csv_path
                else:
                    # malformed, skip
                    pass
            else:
                # For parameters/scalars (can be comma-separated declarations)
                if len(decl_parts) > 1:
                    names = [n.strip() for n in re.split(r",", decl_parts[1])]
                else:
                    names = []
                for raw in names:
                    # name may include dimension suffix (e.g., A(i,j)) — take identifier prefix
                    m = IDENT_RE.search(raw)
                    if m:
                        name = m.group(1)
                        sym = ensure_symbol(name, stype)
                        sym.decls.append(Definition(kind='declaration', text=line, loc=SourceLoc(entries[i].file, entries[i].line)))
            # For parameters, scalars, and sets, handle multi-line declarations
            if stype in ['parameter', 'scalar', 'set']:
                if not line.strip().endswith(';'):
                    # Multi-line declaration: accumulate until ';'
                    accumulated_names = set(n.lower() for n in names)  # To avoid duplicates
                    j = i + 1
                    while j < len(entries):
                        l2 = entries[j].text.strip()
                        if DECL_START_RE.match(entries[j].text) or INCLUDE_RE.match(entries[j].text) or SOLVE_RE.search(entries[j].text) or MODEL_RE.search(entries[j].text):
                            break
                        # Stop accumulation at execution statements (even without semicolon)
                        word_match = re.search(r'^\s*([A-Za-z_]\w*)', l2)
                        first_word = word_match.group(1) if word_match else ''
                        if first_word.lower() in NON_ASSIGNABLE_KEYWORDS:
                            break
                        if not DECL_LINE_RE.match(entries[j].text):
                            break
                        # Parse parameter/scalar names from this line
                        stripped_l2 = entries[j].text.strip()
                        if (stripped_l2 and not stripped_l2.startswith('*') and not stripped_l2.startswith('/')
                            and '.' not in stripped_l2 and not stripped_l2.startswith('(')):
                            parts = stripped_l2.split(None, 1)
                            param_name = parts[0]
                            dims = []
                            if '(' in param_name:
                                name_match = re.match(r'([A-Za-z_]\w*)\s*\(\s*([^)]*)\s*\)?', param_name)
                                if name_match:
                                    param_name = name_match.group(1)
                                    dims_str = name_match.group(2).strip()
                                    dims = [d.strip() for d in dims_str.split(',') if d.strip()]
                            else:
                                m = IDENT_RE.match(param_name)
                                if m:
                                    param_name = m.group(0)
                                else:
                                    param_name = None
                            if param_name and param_name.lower() not in accumulated_names:
                                accumulated_names.add(param_name.lower())
                                psym = ensure_symbol(param_name, stype)
                                psym.decls.append(Definition(kind='declaration', text=entries[j].text, loc=SourceLoc(entries[j].file, entries[j].line)))
                                psym.dims = dims
                        # Check for end of block after parsing
                        if ';' in l2:
                            i = j
                            break
                        j += 1
                    # Also handle potential data blocks after multi-line declarations
                    if j < len(entries):
                        k = i + 1
                        in_data = False
                        while k < len(entries):
                            l2 = entries[k].text.strip()
                            if l2.startswith('/'):
                                in_data = True
                            if in_data and l2.endswith('/'):
                                i = k
                                break
                            if DECL_START_RE.match(entries[k].text):
                                break
                            k += 1
                else:
                    # Single line or data block handling
                    j = i + 1
                    in_data = False
                    while j < len(entries):
                        l2 = entries[j].text.strip()
                        if l2.startswith('/'):
                            in_data = True
                        if in_data and l2.endswith('/'):
                            i = j
                            break
                        if DECL_START_RE.match(entries[j].text):
                            break
                        j += 1
            i += 1
            continue

        # Check for multi-line model start
        if re.match(r'^\s*model\s+', line, re.IGNORECASE) and not line.strip().endswith(';'):
            accumulated = line
            j = i + 1
            while j < len(entries):
                next_line = entries[j].text
                if not next_line.strip():
                    j += 1
                    continue
                accumulated += ' ' + next_line.strip()
                if next_line.strip().endswith(';'):
                    # Parse the full model
                    model_match = MODEL_RE.search(accumulated)
                    if model_match:
                        mname = model_match.group(1)
                        eqs = [e.strip() for e in model_match.group(2).split(',') if e.strip()]
                        models.append(ModelInfo(name=mname, equations=eqs, loc=SourceLoc(entries[i].file, entries[i].line)))
                    i = j
                    break
                j += 1
            i += 1
            continue

        # Model membership (single line)
        mm = MODEL_RE.search(line)
        if mm:
            mname = mm.group(1)
            eqs = [e.strip() for e in mm.group(2).split(',') if e.strip()]
            models.append(ModelInfo(name=mname, equations=eqs, loc=SourceLoc(entries[i].file, entries[i].line)))
            i += 1
            continue

        # Solve detection
        sm = SOLVE_RE.search(line)
        if sm:
            model = sm.group(1)
            solver = sm.group(2)
            sense = sm.group(3).lower()
            objvar = sm.group(4).strip()
            solves.append(SolveInfo(model=model, solver=solver, sense=sense, objvar=objvar, loc=SourceLoc(entries[i].file, entries[i].line)))
            ensure_symbol(objvar, 'variable')
            i += 1
            continue

        # Check for multi-line equation start (has .. but no ; at end)
        eq_start_re = re.compile(r"^\s*([A-Za-z_]\w*\s*(?:\([^)]*\))?)\s*(?:\$(.+?))?\s*\.\.\s*(.+?)\s*=(l|L|e|E|g|G)=\s*(.*)$", re.IGNORECASE)
        em_start = eq_start_re.match(line)
        if em_start and not line.strip().endswith(';'):
            # Accumulate multi-line equation
            ename = em_start.group(1)
            accumulated = line
            j = i + 1
            while j < len(entries):
                next_line = entries[j].text
                if not next_line.strip():
                    j += 1
                    continue
                accumulated += ' ' + next_line.strip()
                if next_line.strip().endswith(';'):
                    # Found the end, parse the full equation
                    em = EQUATION_RE.match(accumulated)
                    if em:
                        m = IDENT_RE.search(ename)
                        if m:
                            ename_ident = m.group(1)
                            lhs, sense, rhs = em.groups()[2:5]  # Skip ename, condition
                            sym = ensure_symbol(ename_ident, 'equation')
                            deps = find_idents(lhs) | find_idents(rhs)
                            sym.defs.append(Definition(kind='equation', text=accumulated, loc=SourceLoc(entries[i].file, entries[i].line), deps=deps, lhs=lhs.strip()))
                    i = j
                    break
                j += 1
            i += 1
            continue

        # Equation definitions (single line)
        em = EQUATION_RE.match(line)
        if em:
            groups = em.groups()
            ename = groups[0]
            lhs = groups[2]
            sense = groups[3]
            rhs = groups[4]
            m = IDENT_RE.search(ename)
            if m:
                ename_ident = m.group(1)
                sym = ensure_symbol(ename_ident, 'equation')
                deps = find_idents(lhs) | find_idents(rhs)
                sym.defs.append(Definition(kind='equation', text=line, loc=SourceLoc(entries[i].file, entries[i].line), deps=deps, lhs=lhs.strip()))
            i += 1
            continue

        # Check for multi-line assignment start
        am_start_re = re.compile(r"^\s*([A-Za-z_]\w*)(\s*\([^)]*\))?", re.IGNORECASE)
        am_start = am_start_re.match(line)
        if am_start:
            atgt = am_start.group(1).strip().lower()
            if atgt in NON_ASSIGNABLE_KEYWORDS:
                i += 1
                continue
            if not line.strip().endswith(';'):
                accumulated = line
                j = i + 1
                while j < len(entries):
                    next_line = entries[j].text
                    if not next_line.strip():
                        j += 1
                        continue
                    accumulated += ' ' + next_line.strip()
                    if next_line.strip().endswith(';'):
                        # Check if it's an assignment by having =
                        if '=' in accumulated:
                            eq_pos = accumulated.find('=')
                            expr = accumulated[eq_pos+1:].strip().rstrip(';')
                            deps = find_idents(expr)
                            sym = ensure_symbol(atgt, symbols.get(atgt.lower(), SymbolInfo(original=atgt, stype='unknown')).stype or 'unknown')
                            sym.defs.append(Definition(kind='assignment', text=accumulated, loc=SourceLoc(entries[i].file, entries[i].line), deps=deps, lhs=atgt.strip()))
                        i = j
                        break
                    j += 1
                i += 1
                continue

        # Assignments (parameters/scalars/etc.)
        am = ASSIGN_RE.match(line)
        if am:
            tgt = am.group(1)
            expr = am.group(3)
            deps = find_idents(expr)
            sym = ensure_symbol(tgt, symbols.get(tgt.lower(), SymbolInfo(original=tgt, stype='unknown')).stype or 'unknown')
            sym.defs.append(Definition(kind='assignment', text=line, loc=SourceLoc(entries[i].file, entries[i].line), deps=deps, lhs=tgt.strip()))
            i += 1
            continue

        i += 1

    return symbols, models, solves

# ----------------------------
# Tracing utilities
# ----------------------------

def trace_symbol(symbols: Dict[str, SymbolInfo], name: str, depth: int = 0, visited: Optional[Set[str]] = None, exclude_sets: bool = False) -> List[str]:
    """Return textual trace for a symbol: where its values come from (assignments, tables, deps)."""
    name = norm_ident(name).lower()
    if visited is None:
        visited = set()
    if name in visited:
        sym = symbols.get(name)
        display_name = sym.original if sym else name
        return ["  "*depth + f"↪ {display_name} (cycle detected)"]
    visited.add(name)
    out: List[str] = []
    sym = symbols.get(name)
    if not sym:
        out.append("  "*depth + f"✖ {name}: not declared/defined in parsed files")
        return out
    display_type = f"{sym.stype}: {sym.vtype}" if sym.vtype and sym.stype == 'variable' else sym.stype
    out.append("  "*depth + f"• {sym.original} [{display_type}]")
    if sym.csv_file:
        out.append("  "*depth + f"  ├─ data loaded from CSV '{sym.csv_file}'")
    if sym.defs:
        for d in sym.defs:
            if d.kind == 'table':
                if d.skipped:
                    out.append("  "*depth + f"  ├─ table at {d.loc.file}:{d.loc.line} (large table, parsing skipped for performance)")
                else:
                    out.append("  "*depth + f"  ├─ table at {d.loc.file}:{d.loc.line} with {len(d.values)} numeric entries")
            elif d.kind == 'assignment':
                out.append("  "*depth + f"  ├─ assignment at {d.loc.file}:{d.loc.line}: {d.text.strip()}")
            elif d.kind == 'equation':
                out.append("  "*depth + f"  ├─ equation at {d.loc.file}:{d.loc.line}: {d.text.strip()}")
            elif d.kind == 'gdx_load':
                out.append("  "*depth + f"  ├─ GDX load at {d.loc.file}:{d.loc.line} from '{d.gdx_file}'")
            elif d.kind == 'alias_declaration':
                out.append("  "*depth + f"  ├─ alias_declaration at {d.loc.file}:{d.loc.line}: {d.text.strip()}")
                if sym.base_set:
                    base_sym = symbols.get(sym.base_set.lower())
                    if base_sym:
                        out.extend(trace_symbol(symbols, sym.base_set, depth+1, visited, exclude_sets))
            else:
                out.append("  "*depth + f"  ├─ {d.kind} at {d.loc.file}:{d.loc.line}")
            dep_symbols = {dep: symbols.get(dep.lower()) for dep in sorted(d.deps)}
            for dep_name, dep_sym in dep_symbols.items():
                if not dep_sym or (exclude_sets and dep_sym.stype == 'set'):
                    continue
                out.extend(trace_symbol(symbols, dep_name, depth+1, visited, exclude_sets))
    elif sym.decls and any(d.kind == 'alias_declaration' for d in sym.decls):
        for d in sym.decls:
            if d.kind == 'alias_declaration':
                out.append("  "*depth + f"  ├─ alias_declaration at {d.loc.file}:{d.loc.line}: {d.text.strip()}")
                if sym.base_set:
                    base_sym = symbols.get(sym.base_set.lower())
                    if base_sym:
                        out.extend(trace_symbol(symbols, sym.base_set, depth+1, visited, exclude_sets))
                break  # assuming only one
    elif sym.decls and any(d.kind == 'declaration' for d in sym.decls):
        for d in sym.decls:
            if d.kind == 'declaration':
                out.append("  "*depth + f"  ├─ declaration at {d.loc.file}:{d.loc.line}: {d.text.strip()}")
                break
    else:
        out.append("  "*depth + f"  ├─ no definitions found (may be loaded via GDX or includes not captured)")
    return out


def extract_objective(symbols: Dict[str, SymbolInfo], solve: Optional[SolveInfo]) -> Tuple[Optional[SolveInfo], Optional[Definition]]:
    """Find objective variable from the solve statement and the equation/assignment that defines it (if present)."""
    if not solve:
        return None, None
    s = solve
    # First, look for assignments/equations where objvar is the target
    for sym in symbols.values():
        for d in sym.defs:
            if d.lhs == s.objvar and d.kind in ['assignment', 'equation']:
                return s, d
    # Then, look for equations that reference objvar in dependencies (e.g., other equations using objvar)
    for sym in symbols.values():
        if sym.stype == 'equation':
            for d in sym.defs:
                if d.kind == 'equation' and s.objvar in d.deps:
                    return s, d
    return s, None


def explain_equation(symbols: Dict[str, SymbolInfo], eq_name: str, exclude_sets: bool = False) -> List[str]:
    eq = symbols.get(eq_name.lower()) if eq_name else None
    if not eq or eq.stype != 'equation':
        return [f"Equation '{eq_name}' not found."]
    out: List[str] = []
    for d in eq.defs:
        if d.kind != 'equation':
            continue
        out.append(f"Equation {eq.original} defined at {d.loc.file}:{d.loc.line}")
        out.append(d.text.strip())
        out.append("Dependencies:")
        dep_symbols = {dep: symbols.get(dep.lower()) for dep in sorted(d.deps)}
        for dep_name, dep_sym in dep_symbols.items():
            if not dep_sym or (exclude_sets and dep_sym.stype == 'set'):
                continue
            out.append(f"  - {dep_name}")
            out.extend(trace_symbol(symbols, dep_name, depth=1, exclude_sets=exclude_sets))
    return out

# ----------------------------
# CLI
# ----------------------------

def print_symbols_histogram(symbols):
    stype_counts = defaultdict(int)
    for sym in symbols.values():
        stype_counts[sym.stype] += 1
    expected_stypes = ['set', 'parameter', 'scalar', 'table', 'variable', 'equation', 'unknown']
    all_stypes = expected_stypes + [s for s in stype_counts if s not in expected_stypes]
    for stype in sorted(all_stypes):
        plural_stype = stype + "s" if stype not in ("solves", "unknown") else stype
        count = stype_counts.get(stype, 0)
        print(f"{plural_stype}: {count}")

def main():
    ap = argparse.ArgumentParser(description="Trace raw data sources in a GAMS model. Use subcommands: parse to load, list to list solves/solve, trace/trace_with_sets for tracing.")
    subparsers = ap.add_subparsers(dest='subcommand', help='Available subcommands')

    # parse subcommand
    parse_sub = subparsers.add_parser('parse', help='Parse root .gms file and save parse data to gams_trace.parse; lists all solve statements')
    parse_sub.add_argument('gms_file', help='Path to root .gms file')

    # save subcommand
    save_sub = subparsers.add_parser('save', help='Save merged decommented source from last parse to specified file')
    save_sub.add_argument('output_file', help='Path to output file (merged decommented source)')

    # list subcommand
    list_sub = subparsers.add_parser('list', help='List solves, models, symbols, or specific details')
    list_subs = list_sub.add_subparsers(dest='list_command')
    list_subs.add_parser('solves', help='List all detected solve statements')

    # Symbol type commands: plural for list names
    for typ in ['sets', 'parameters', 'scalars', 'tables', 'variables', 'equations', 'unknowns']:
        list_subs.add_parser(typ, help=f'List all {typ}')

    # trace subcommand
    trace_sub = subparsers.add_parser('trace', help='Trace objective, symbols, or equations (excludes sets from output)')
    trace_sub.add_argument('target', nargs=argparse.REMAINDER, help='Objective or symbol/equation name to trace; for objective, optional solve number follows')

    # trace_with_sets subcommand
    trace_with_sets_sub = subparsers.add_parser('trace_with_sets', help='Trace objective, symbols, or equations (includes sets in output)')
    trace_with_sets_sub.add_argument('target', nargs=argparse.REMAINDER, help='Objective or symbol/equation name to trace; for objective, optional solve number follows')

    # show subcommand
    show_sub = subparsers.add_parser('show', help='Show details of a specific solve (by number 1+) or symbol')
    show_sub.add_argument('target', help='Solve number (1+) or name of the symbol')

    args = ap.parse_args()

    # If no args provided, show usage
    if args.subcommand is None:
        ap.print_help()
        sys.exit(0)

    symbols = None
    models = None
    solves = None

    if args.subcommand == 'parse':
        # Parse from root
        try:
            files = load_gms(args.gms_file)
        except Exception as e:
            print(f"Error loading files: {e}")
            sys.exit(1)

        symbols, models, solves = parse_code(files)

        # Save parse pickle
        pickled_data = (files, symbols, models, solves)
        with open('gams_trace.parse', 'wb') as f:
            pickle.dump(pickled_data, f)
        print("\rParsing complete, saved parse tree to gams_trace.parse", flush=True)

        # Print summary
        print(f"solves: {len(solves)}")
        print_symbols_histogram(symbols)

    else:
        # Load parse pickle
        if not os.path.exists('gams_trace.parse'):
            print("Error: Parsed data file 'gams_trace.parse' does not exist. Run with 'parse <gms_file>' first.")
            sys.exit(1)
        with open('gams_trace.parse', 'rb') as f:
            pickled_data = pickle.load(f)
            if len(pickled_data) != 4:
                print("Error: gams_trace.parse is in an old format. Please re-run 'parse <gms_file>' to regenerate.")
                sys.exit(1)
            files, symbols, models, solves = pickled_data
        print("Parsed data loaded from gams_trace.parse")

        if args.subcommand == 'list':
            if args.list_command == 'solves':
                print("Solve statements:")
                for idx, s in enumerate(solves, start=1):
                    print(f"{idx}. model={s.model} using {s.solver} {s.sense} objvar={s.objvar} at {s.loc.file}:{s.loc.line}")
                print()
            else:
                    if args.list_command is None:
                        # Show summary
                        print("Parsed symbols summary:")
                        print_symbols_histogram(symbols)
                        sys.exit(0)
            # Handle symbol type lists
            plural_types = ['sets', 'parameters', 'scalars', 'tables', 'variables', 'equations', 'unknowns']
            if args.list_command in plural_types:
                stype = args.list_command[:-1]  # Remove 's' to get singular stype
                if stype == 'variable':
                    # Special handling for variables: group by type
                    vtypes = defaultdict(list)
                    for sym in symbols.values():
                        if sym.stype == 'variable':
                            vtype_str = sym.vtype or "UNKNOWN"
                            vtypes[vtype_str].append(sym.original)
                    for vtype in sorted(vtypes):
                        print(f"{vtype} Variables:")
                        for name in sorted(vtypes[vtype]):
                            print(f"- {name}")
                        print()
                else:
                    names = sorted([sym.original for sym in symbols.values() if sym.stype == stype])
                    if names:
                        print(f"{stype.title()}s:")
                        for name in names:
                            print(f"- {name}")
                        print()
                    else:
                        print(f"No {stype}s found.")
                        print()

        elif args.subcommand == 'trace' or args.subcommand == 'trace_with_sets':
            exclude_sets = (args.subcommand == 'trace')

            target = args.target
            if target and target[0].lower() == 'objective':
                solve_num = int(target[1]) if len(target) > 1 else None
                if solve_num is None:
                    print("Available solves:")
                    for idx, s in enumerate(solves, start=1):
                        print(f"{idx}. model={s.model} using {s.solver} {s.sense} objvar={s.objvar} at {s.loc.file}:{s.loc.line}")
                    while True:
                        try:
                            solve_num = int(input(f"Enter solve number (1-{len(solves)}): "))
                            if 1 <= solve_num <= len(solves):
                                break
                        except ValueError:
                            pass
                else:
                    if solve_num < 1 or solve_num > len(solves):
                        print("Invalid solve number. Available solves:")
                        for idx, s in enumerate(solves, start=1):
                            print(f"{idx}. model={s.model} using {s.solver} {s.sense} objvar={s.objvar} at {s.loc.file}:{s.loc.line}")
                        sys.exit(1)
                target_solve = solves[solve_num - 1]
                s, obj_def = extract_objective(symbols, target_solve)
                if not s:
                    print("No solve detected.")
                else:
                    print(f"Objective: {s.sense} {s.objvar} (from solve at {s.loc.file}:{s.loc.line})")
                    if obj_def:
                        kind_str = "equation" if obj_def.kind == 'equation' else 'assignment'
                        print(f"Objective defining {kind_str} at {obj_def.loc.file}:{obj_def.loc.line}")
                        print(obj_def.text.strip())
                        print("\nTracing dependencies of the objective expression:")
                        dep_symbols = {dep: symbols.get(dep.lower()) for dep in sorted(obj_def.deps)}
                        for dep_name, dep_sym in dep_symbols.items():
                            if dep_name == s.objvar or (exclude_sets and dep_sym and dep_sym.stype == 'set'):
                                continue
                            print(f"- {dep_name}")
                            for ln in trace_symbol(symbols, dep_name, depth=1, exclude_sets=exclude_sets):
                                print(ln)
                    else:
                        print("No explicit objective-defining equation found. Consider tracing parameters used in constraints that reference the objective variable.")
                print()
            else:
                # trace symbol or equation
                name = target[0] if target else None
                if name:
                    if exclude_sets:
                        sym = symbols.get(name.lower())
                        if sym and sym.stype == 'set':
                            print(f"Error: Cannot trace sets with 'trace' command. Use 'trace_with_sets' instead.")
                            sys.exit(1)
                    sym = symbols.get(name.lower())
                    if sym and sym.stype == 'equation':
                        for ln in explain_equation(symbols, name, exclude_sets=exclude_sets):
                            print(ln)
                        print()
                    else:
                        for ln in trace_symbol(symbols, name, exclude_sets=exclude_sets):
                            print(ln)
                        print()
                else:
                    print("Error: No target specified for trace.")
                    sys.exit(1)

        elif args.subcommand == 'show':
            target = args.target
            try:
                solve_num = int(target)
                if solve_num < 1 or solve_num > len(solves):
                    print("Invalid solve number. Available solves:")
                    for idx, s in enumerate(solves, start=1):
                        print(f"{idx}. model={s.model} using {s.solver} {s.sense} objvar={s.objvar} at {s.loc.file}:{s.loc.line}")
                    sys.exit(1)
                selected_solve = solves[solve_num - 1]
                print(f"Solve: model={selected_solve.model} using {selected_solve.solver} {selected_solve.sense}, objvar={selected_solve.objvar} at {selected_solve.loc.file}:{selected_solve.loc.line}")
                print()
            except ValueError:
                # symbol
                symbol_name = target
                sym = symbols.get(symbol_name.lower()) if symbol_name else None
                if sym:
                    # Show definition or declaration
                    d = None
                    loc = None
                    text = None
                    if sym.defs:
                        d = sym.defs[0]  # Take first definition
                        loc = d.loc
                        text = d.text
                    elif sym.decls:
                        d = sym.decls[0]  # Take first declaration
                        loc = d.loc
                        text = d.text
                    if loc:
                        if sym.stype == 'variable':
                            print(f"Variable {sym.original} ({sym.vtype}) declared at {loc.file}:{loc.line}")
                        elif sym.stype == 'unknown':
                            print(f"Unknown symbol {sym.original} last referenced at {loc.file}:{loc.line}")
                        else:
                            print(f"{sym.stype.title()} {sym.original} declared at {loc.file}:{loc.line}")
                        if sym.dims:
                            print(f"Dimensions: {', '.join(sym.dims)}")
                        if sym.stype == 'set' and sym.base_set:
                            base_sym = symbols.get(sym.base_set.lower())
                            if base_sym:
                                print(f"This set is an alias for {base_sym.original}.")
                        # Show first <=5 lines of text
                        if text:
                            lines = text.strip().split('\n')
                            for i in range(min(5, len(lines))):
                                print(lines[i])
                            if len(lines) > 5:
                                print("(...)")
                    else:
                        if sym.stype == 'variable':
                            print(f"Variable {sym.original} ({sym.vtype}) has no parsed definitions or declarations.")
                        elif sym.stype == 'unknown':
                            print(f"Unknown symbol {sym.original} has no parsed definitions or declarations.")
                        else:
                            print(f"{sym.stype.title()} {sym.original} has no parsed definitions or declarations.")
                    print()
                else:
                    print(f"Symbol '{symbol_name}' not found in parsed data.")
                    print()

if __name__ == '__main__':
    main()
