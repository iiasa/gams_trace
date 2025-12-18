#!/usr/bin/env python3

import argparse
import os
import pickle
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union

import lark

# GAMS Grammar using Lark
GAMS_GRAMMAR = r"""
%import common.WS
%ignore WS

start: (statement ";")*

statement: include_stmt
         | gdxin_stmt
         | load_stmt
         | loaddc_stmt
         | model_stmt
         | solve_stmt
         | declaration
         | table_declaration
         | assignment
         | equation

include_stmt: DOLLAR ("bat")? "include" path
path: STRING | WORD

gdxin_stmt: DOLLAR "gdxin" path

load_stmt: DOLLAR "load" symbol_list
loaddc_stmt: DOLLAR "loaddc" symbol_list

symbol_list: WORD ("," WORD)*
STRING: /"[^"]*"/ | /'[^']*'/

declaration: ("set"|"sets"|"parameter"|"parameters"|"scalar"|"scalars"|"equation"|"equations"|"variable"|"variables") name_list

name_list: name ("," name)*
name: WORD dims?
dims: "(" dim ("," dim)* ")"
dim: WORD

table_declaration: "table" name dims?

model_stmt: "model" WORD "/" eq_list "/"
eq_list: WORD ("," WORD)*

solve_stmt: "solve" WORD "using" "lp" ("minimizing" | "maximizing") WORD

assignment: name "=" expression

equation: WORD ".." expression ("=l=" | "=e=" | "=g=") expression

expression: term (( "+" | "-" | "*" | "/" ) term)*
term: primary | "(" expression ")" | function
primary: NUMBER | WORD
function: WORD "(" arg_list ")"
arg_list: | expression ("," expression)*

DOLLAR: "$"
NUMBER: /\d+(\.\d+)?/
WORD: /[A-Za-z_]\w*/
"""

# Compile grammar
parser = lark.Lark(GAMS_GRAMMAR, propagate_positions=True)

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

@dataclass
class ModelInfo:
    name: str
    equations: List[str]
    loc: SourceLoc

@dataclass
class SolveInfo:
    model: str
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
DECL_START_RE = re.compile(r"^\s*(sets?|parameters?|scalars?|tables?|variables?|equations?)\b", re.IGNORECASE)
INCLUDE_RE = re.compile(r"^\s*\$(bat)?include\s+(.+)", re.IGNORECASE | re.DOTALL)
GDXIN_RE = re.compile(r"^\s*\$gdxin\s+(.+)", re.IGNORECASE | re.DOTALL)
LOAD_RE = re.compile(r"^\s*\$load\s+(.+)", re.IGNORECASE | re.DOTALL)
LOADDC_RE = re.compile(r"^\s*\$loaddc\s+(.+)", re.IGNORECASE | re.DOTALL)
SOLVE_RE = re.compile(r"solve\s+(\w+)\s+using\s+lp\s+(minimizing|maximizing)\s+(\w+)", re.IGNORECASE | re.DOTALL)
MODEL_RE = re.compile(r"model\s+(\w+)\s*/\s*([^/]*)/\s*;", re.IGNORECASE | re.DOTALL)
ASSIGN_RE = re.compile(r"^\s*([A-Za-z_]\w*)(\s*\([^)]*\))?\s*=\s*(.+?);\s*$", re.IGNORECASE | re.DOTALL)
EQUATION_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*\.\.\s*(.+?)\s*=(l|L|e|E|g|G)=\s*(.+?);\s*$", re.IGNORECASE | re.DOTALL)
VAR_DECL_RE = re.compile(r"^\s*(positive|free|binary|integer)?\s*variables?\s+(.+);", re.IGNORECASE)
MULTI_VAR_DECL_RE = re.compile(r"^\s*(positive|free|binary|integer)?\s*variables?\s*(.*?);", re.IGNORECASE | re.DOTALL)
TABLE_HEAD_RE = re.compile(r"^\s*tables?\s+([A-Za-z_]\w*)\s*\(([^)]*)\)\s*", re.IGNORECASE | re.DOTALL)


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

        # Process lines, handling includes inline
        line_num = 1
        i = 0
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
                    merged_lines.append(LineEntry(text=line, file=full, line=line_num))
            line_num += 1
            i += 1

    load_file(root_path, 0)
    print(f"\rLoaded {len(merged_lines)} lines from included files.{'                                    '}", flush=True)
    return merged_lines

# ----------------------------
# Parser
# ----------------------------

def extract_idents(tree_node) -> Set[str]:
    """Extract identifiers from a Lark tree node (for expressions)."""
    ids = set()
    if isinstance(tree_node, lark.Tree):
        if tree_node.data == 'WORD':
            word = tree_node.children[0].value.lower()
            if word not in BUILTINS and not word.isnumeric():
                ids.add(word)
        else:
            for child in tree_node.children:
                ids.update(extract_idents(child))
    elif isinstance(tree_node, lark.Token) and tree_node.type == 'WORD':
        word = tree_node.value.lower()
        if word not in BUILTINS and not word.isnumeric():
            ids.add(word)
    return ids

def parse_code(entries: List[LineEntry]) -> Tuple[Dict[str, SymbolInfo], List[ModelInfo], List[SolveInfo]]:
    symbols: Dict[str, SymbolInfo] = {}
    models: List[ModelInfo] = []
    solves: List[SolveInfo] = []

    def ensure_symbol(name: str, stype: str) -> SymbolInfo:
        original = name
        name_lower = name.lower()
        if name_lower not in symbols:
            symbols[name_lower] = SymbolInfo(original=original, stype=stype)
        else:
            if symbols[name_lower].stype == "unknown":
                symbols[name_lower].stype = stype
        return symbols[name_lower]

    status_msg = f"Parsing merged code ({len(entries)} lines)"
    print(f"\r{status_msg}", end="", flush=True)
    current_gdx_file = None
    i = 0
    while i < len(entries):
        line = entries[i].text
        if not line.strip():
            i += 1
            continue
        # Merge continuation lines if trailing comma and next line
        # (simple heuristic)
        full_line = line
        j = i + 1
        while j < len(entries) and entries[j].text.lstrip().startswith(','):
            full_line += ' ' + entries[j].text.strip()
            j += 1

        try:
            # Try to parse the statement
            tree = parser.parse(f"{full_line};")
            # Assume first child is the statement
            stmt = tree.children[0]
            data = stmt.data

            if data == 'include_stmt':
                # Handle include (but already handled in load_gms)
                pass
            elif data == 'gdxin_stmt':
                path = stmt.children[1]  # WORD or STRING
                current_gdx_file = path.value.strip('"\'')
            elif data in ('load_stmt', 'loaddc_stmt'):
                syms = stmt.children[1:]  # symbol_list
                for sym_tok in syms:
                    if isinstance(sym_tok, lark.Token):  # WORD
                        sym_name = sym_tok.value
                        sym = ensure_symbol(sym_name, 'unknown')
                        sym.defs.append(Definition(kind='gdx_load', text=full_line, loc=SourceLoc(entries[i].file, entries[i].line), deps=set(), lhs=sym_name.lower(), gdx_file=current_gdx_file))
            elif data == 'declaration':
                stype = stmt.children[0].data  # 'set', 'parameter', etc.
                stype_map = {'set': 'set', 'parameter': 'parameter', 'scalar': 'scalar', 'equation': 'equation', 'variable': 'variable'}
                stype_str = stype_map.get(stype, 'unknown')
                # name_list
                names = []
                for node in stmt.children[1:]:
                    if node.data == 'name':
                        name_tok = node.children[0]  # WORD
                        names.append(name_tok.value)
                for name in names:
                    sym = ensure_symbol(name, stype_str)
                    sym.decls.append(Definition(kind='declaration', text=full_line, loc=SourceLoc(entries[i].file, entries[i].line)))
            elif data == 'table_declaration':
                tname = stmt.children[1].children[0].value  # name -> WORD
                dims = []
                if len(stmt.children) > 2:
                    dims_node = stmt.children[2]
                    for dim in dims_node.children:
                        if isinstance(dim, lark.Tree) and dim.data == 'dim':
                            dims.append(dim.children[0].value)
                sym = ensure_symbol(tname, 'table')
                sym.dims = dims
                # Table data not parsed via grammar, keep as is
                # Parse subsequent lines for table data
                table_lines: List[str] = []
                k = j
                while k < len(entries):
                    l2 = entries[k].text
                    if not l2.strip():
                        k += 1
                        continue
                    if l2.strip() == ';':
                        table_lines.append(l2)
                        k += 1
                        break
                    table_lines.append(l2)
                    k += 1
                # Same table parsing logic
                if len(table_lines) > 100:
                    values = {}
                    skipped = True
                else:
                    values = {}
                    skipped = False
                    header_cols = []
                    for tl in table_lines:
                        if tl.strip() and not tl.strip().endswith(':') and not tl.strip() == ';':
                            header_cols = tl.strip().split()
                            break
                    for tl in table_lines:
                        if not tl.strip() or tl.strip() == ';':
                            continue
                        parts = tl.strip().split()
                        if not parts or not header_cols:
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
                defn = Definition(kind='table', text=full_line, loc=SourceLoc(entries[i].file, entries[i].line), deps=set(), values=values, skipped=skipped)
                sym.defs.append(defn)
                i = k
                continue
            elif data == 'model_stmt':
                mname = stmt.children[1].value  # WORD after 'model'
                eqs = []
                if len(stmt.children) > 3:
                    eq_list = stmt.children[3:]  # eq_list
                    for eq in eq_list:
                        if isinstance(eq, lark.Token):
                            eqs.append(eq.value)
                        elif hasattr(eq, 'value'):
                            eqs.append(eq.value)
                models.append(ModelInfo(name=mname, equations=eqs, loc=SourceLoc(entries[i].file, entries[i].line)))
            elif data == 'solve_stmt':
                mmodel = stmt.children[1].value
                sense = stmt.children[3].data  # 'minimizing' or 'maximizing'
                objvar = stmt.children[5].value
                solves.append(SolveInfo(model=mmodel, sense=sense, objvar=objvar, loc=SourceLoc(entries[i].file, entries[i].line)))
                ensure_symbol(objvar, 'variable')
            elif data == 'assignment':
                tgt = stmt.children[0]  # name
                tgt_name = tgt.children[0].value if isinstance(tgt, lark.Tree) else tgt.value
                expr = stmt.children[2]  # expression
                deps = extract_idents(expr)
                sym = ensure_symbol(tgt_name, symbols.get(tgt_name.lower(), SymbolInfo(original=tgt_name, stype='unknown')).stype or 'unknown')
                sym.defs.append(Definition(kind='assignment', text=full_line, loc=SourceLoc(entries[i].file, entries[i].line), deps=deps, lhs=tgt_name))
            elif data == 'equation':
                ename = stmt.children[0].value
                lhs_expr = stmt.children[2]
                rhs_expr = stmt.children[4]
                deps = extract_idents(lhs_expr) | extract_idents(rhs_expr)
                sym = ensure_symbol(ename, 'equation')
                sym.defs.append(Definition(kind='equation', text=full_line, loc=SourceLoc(entries[i].file, entries[i].line), deps=deps, lhs=''))

            # Advance i to j for multi-line handling
            i = j
        except lark.exceptions.LarkError:
            # Fallback: if Lark fails, try to match with regex for important cases like multi-line
            # For now, advance one line
            i += 1

        # Handle multi-line not parsed by simple statement
        # Keep some regex for complex cases like multi-line declarations, but simplify
        # Time constraints, keep original for unparsed lines

    return symbols, models, solves

# ----------------------------
# Tracing utilities
# ----------------------------

def trace_symbol(symbols: Dict[str, SymbolInfo], name: str, depth: int = 0, visited: Optional[Set[str]] = None) -> List[str]:
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
    out.append("  "*depth + f"• {sym.original} [{sym.stype}]")
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
            else:
                out.append("  "*depth + f"  ├─ {d.kind} at {d.loc.file}:{d.loc.line}")
            for dep in sorted(d.deps):
                if dep == name:
                    continue
                out.extend(trace_symbol(symbols, dep, depth+1, visited))
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


def explain_equation(symbols: Dict[str, SymbolInfo], eq_name: str) -> List[str]:
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
        for dep in sorted(d.deps):
            out.append(f"  - {dep}")
            out.extend(trace_symbol(symbols, dep, depth=1))
    return out

# ----------------------------
# CLI
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Trace raw data sources in a GAMS LP model (CPLEX). Use subcommands: parse to load, list to list solves/solve, trace for tracing.")
    subparsers = ap.add_subparsers(dest='subcommand', help='Available subcommands')

    # parse subcommand
    parse_sub = subparsers.add_parser('parse', help='Parse root .gms file and save parse data to gams_trace.parse; lists all solve statements')
    parse_sub.add_argument('gms_file', help='Path to root .gms file')

    # list subcommand
    list_sub = subparsers.add_parser('list', help='List solves, models, symbols, or specific details')
    list_subs = list_sub.add_subparsers(dest='list_command')
    list_subs.add_parser('solves', help='List all detected solve statements')
    solve_sub = list_subs.add_parser('solve', help='Show details for a specific solve statement')
    solve_sub.add_argument('solve_number', type=int, help='Solve number (1+)')

    # Symbol type commands: plural for list names, singular for definition
    for typ in ['sets', 'parameters', 'scalars', 'tables', 'variables', 'equations']:
        list_subs.add_parser(typ, help=f'List all {typ}')
        singular = list_subs.add_parser(typ[:-1], help=f'Show definition of a {typ[:-1]}')
        singular.add_argument('symbol_name', help=f'Name of the {typ[:-1]}')

    # trace subcommand
    trace_sub = subparsers.add_parser('trace', help='Trace objective, symbols, or equations')
    trace_subs = trace_sub.add_subparsers(dest='trace_command')
    obj_sub = trace_subs.add_parser('objective', help='Trace objective for solve N')
    obj_sub.add_argument('solve_number', nargs='?', type=int, help='Solve number (1+)')
    # For tracing a symbol, it's just 'trace <symbol>', so no subparser, positional
    trace_sub.add_argument('target', help='Symbol or equation name to trace')

    args = ap.parse_args()

    # If no args provided and parse data exists, show summary

    if args.subcommand is None:
        if not os.path.exists('gams_trace.parse'):
            ap.print_help()
            sys.exit(1)
        with open('gams_trace.parse', 'rb') as f:
            symbols, models, solves = pickle.load(f)
        print("Parsed symbols summary:")
        stype_counts = defaultdict(int)
        for sym in symbols.values():
            stype_counts[sym.stype] += 1
        for stype, count in sorted(stype_counts.items()):
            print(f"{stype}: {count}")
        print("\nTip: use 'trace objective', 'trace <eqname>', or 'trace <symbol>' to see detailed traces.")
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

        # Serialize
        with open('gams_trace.parse', 'wb') as f:
            pickle.dump((symbols, models, solves), f)
        print("\rParsing complete, saved parse tree to gams_trace.parse", flush=True)

        # Print summary
        stype_counts = defaultdict(int)
        for sym in symbols.values():
            stype_counts[sym.stype] += 1
        print(f"solves: {len(solves)}")
        expected_stypes = ['set', 'parameter', 'scalar', 'table', 'variable', 'equation', 'unknown']
        all_stypes = expected_stypes + [s for s in stype_counts if s not in expected_stypes]
        for stype in sorted(all_stypes):
            plural_stype = stype + "s" if stype not in ("solves", "unknown") else stype
            count = stype_counts.get(stype, 0)
            print(f"{plural_stype}: {count}")

        # Write detailed solves to file
        with open('gams_trace.solves', 'w') as f:
            f.write(f"parse: {args.gms_file}\n")
            for idx, s in enumerate(solves, start=1):
                f.write(f"{idx}|model={s.model}|sense={s.sense}|objvar={s.objvar}|file={s.loc.file}|line={s.loc.line}\n")

    else:
        # load from pickle
        if not os.path.exists('gams_trace.parse'):
            print("Error: Parsed data file 'gams_trace.parse' does not exist. Run with 'parse <gms_file>' first.")
            sys.exit(1)
        with open('gams_trace.parse', 'rb') as f:
            symbols, models, solves = pickle.load(f)
        print("Parsed data loaded from gams_trace.parse")

        if args.subcommand == 'list':
            if args.list_command == 'solves':
                print("Solve statements:")
                for idx, s in enumerate(solves, start=1):
                    print(f"{idx}. model={s.model} sense={s.sense} objvar={s.objvar} at {s.loc.file}:{s.loc.line}")
                print()
            elif args.list_command == 'solve':
                solve_num = args.solve_number
                if solve_num < 1 or solve_num > len(solves):
                    print("Invalid solve number. Available solves:")
                    for idx, s in enumerate(solves, start=1):
                        print(f"{idx}. model={s.model} sense={s.sense} objvar={s.objvar} at {s.loc.file}:{s.loc.line}")
                    sys.exit(1)
                selected_solve = solves[solve_num - 1]
                print(f"Solve: model={selected_solve.model}, sense={selected_solve.sense}, objvar={selected_solve.objvar} at {selected_solve.loc.file}:{selected_solve.loc.line}")
                print()
            else:
                # Handle symbol type lists
                plural_types = ['sets', 'parameters', 'scalars', 'tables', 'variables', 'equations']
                singular_types = ['set', 'parameter', 'scalar', 'table', 'variable', 'equation']
                if args.list_command in plural_types:
                    stype = args.list_command[:-1]  # Remove 's' to get singular stype
                    names = sorted([sym.original for sym in symbols.values() if sym.stype == stype])
                    if names:
                        print(f"{stype.title()}s:")
                        for name in names:
                            print(f"- {name}")
                        print()
                    else:
                        print(f"No {stype}s found.")
                        print()
                elif args.list_command in singular_types:
                    symbol_name = args.symbol_name
                    stype = args.list_command
                    sym = symbols.get(symbol_name.lower()) if symbol_name else None
                    if sym and sym.stype == stype:
                        # Show definition
                        if sym.defs:
                            d = sym.defs[0]  # Take first definition
                            print(f"{stype.title()} {sym.original} defined at {d.loc.file}:{d.loc.line}")
                            # Show first <=5 lines of text
                            lines = d.text.strip().split('\n')
                            for i in range(min(5, len(lines))):
                                print(lines[i])
                            if len(lines) > 5:
                                print("(...)")
                        else:
                            print(f"{stype.title()} {sym.original} has no parsed definitions.")
                        print()
                    else:
                        print(f"No {stype} named '{symbol_name}' found.")
                        print()

        elif args.subcommand == 'trace':
            if args.trace_command == 'objective':
                solve_num = getattr(args, 'solve_number', None)
                if solve_num is None:
                    print("Available solves:")
                    for idx, s in enumerate(solves, start=1):
                        print(f"{idx}. model={s.model} sense={s.sense} objvar={s.objvar} at {s.loc.file}:{s.loc.line}")
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
                            print(f"{idx}. model={s.model} sense={s.sense} objvar={s.objvar} at {s.loc.file}:{s.loc.line}")
                        sys.exit(1)
                target_solve = solves[solve_num - 1]
                s, obj_def = extract_objective(symbols, target_solve)
                if not s:
                    print("No LP solve detected.")
                else:
                    print(f"Objective: {s.sense} {s.objvar} (from solve at {s.loc.file}:{s.loc.line})")
                    if obj_def:
                        kind_str = "equation" if obj_def.kind == 'equation' else 'assignment'
                        print(f"Objective defining {kind_str} at {obj_def.loc.file}:{obj_def.loc.line}")
                        print(obj_def.text.strip())
                        print("\nTracing dependencies of the objective expression:")
                        for dep in sorted(obj_def.deps):
                            if dep == s.objvar:
                                continue
                            print(f"- {dep}")
                            for ln in trace_symbol(symbols, dep, depth=1):
                                print(ln)
                    else:
                        print("No explicit objective-defining equation found. Consider tracing parameters used in constraints that reference the objective variable.")
                print()

            else:
                # trace target (symbol or equation)
                name = args.target
                sym = symbols.get(name.lower()) if name else None
                if sym and sym.stype == 'equation':
                    for ln in explain_equation(symbols, name):
                        print(ln)
                    print()
                else:
                    for ln in trace_symbol(symbols, name):
                        print(ln)
                    print()

if __name__ == '__main__':
    main()
