#!/usr/bin/env python3

import argparse
import os
import pickle
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

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
    kind: str  # 'assignment'|'table'|'declaration'|'equation'
    text: str
    loc: SourceLoc
    deps: Set[str] = field(default_factory=set)  # symbols referenced
    lhs: Optional[str] = None
    values: Dict[Tuple[str, ...], float] = field(default_factory=dict)  # for tables
    skipped: bool = False  # for large tables, parsing skipped for performance

@dataclass
class Symbol:
    name: str
    stype: str  # 'set','parameter','scalar','table','variable','equation','model'
    dims: List[str] = field(default_factory=list)
    decls: List[Definition] = field(default_factory=list)
    defs: List[Definition] = field(default_factory=list)

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

# ----------------------------
# Utility helpers
# ----------------------------

IDENT_RE = re.compile(r"\b([A-Za-z_]\w*)\b")
BUILTINS = {"sum","smin","smax","min","max","ord","card","power","exp","log","abs",
            "uniform","normal","floor","ceil","round","yes","no"}
DECL_START_RE = re.compile(r"^\s*(sets?|parameters?|scalars?|tables?|variables?|equations?)\b", re.IGNORECASE)
INCLUDE_RE = re.compile(r"^\s*\$(bat)?include\s+(.+)", re.IGNORECASE)
SOLVE_RE = re.compile(r"solve\s+(\w+)\s+using\s+lp\s+(minimizing|maximizing)\s+(\w+)", re.IGNORECASE)
MODEL_RE = re.compile(r"model\s+(\w+)\s*/\s*([^/]*)/\s*;", re.IGNORECASE)
ASSIGN_RE = re.compile(r"^\s*([A-Za-z_]\w*)(\s*\([^)]*\))?\s*=\s*(.+?);\s*$", re.IGNORECASE)
EQUATION_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*\.\.\s*(.+?)\s*=(l|L|e|E|g|G)=\s*(.+?);\s*$", re.IGNORECASE)
VAR_DECL_RE = re.compile(r"^\s*(positive|free|binary|integer)?\s*variables?\s+(.+);", re.IGNORECASE)
TABLE_HEAD_RE = re.compile(r"^\s*tables?\s+([A-Za-z_]\w*)\s*\(([^)]*)\)\s*", re.IGNORECASE)


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
        ids.add(token)
    return ids

# ----------------------------
# Loader: read root and resolve includes
# ----------------------------


def load_gms(root_path: str) -> List[Tuple[str, List[str]]]:
    """Return list of (file_path, lines) in dependency order."""
    visited: Set[str] = set()
    ordered: List[Tuple[str, List[str]]] = []
    base_dir = os.path.dirname(os.path.abspath(root_path))

    def _load(fp: str, include_loc: Optional[Tuple[str, int, str]] = None, base_dir: str = base_dir):
        full = os.path.abspath(fp)
        if full in visited:
            return
        status_msg = f"Loading: {os.path.basename(full)}                 "
        print(f"\r{status_msg}", end="", flush=True)
        visited.add(full)
        try:
            with open(full, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
        except FileNotFoundError:
            raise IncludeError(f"Included file not found: {fp}", include_file=fp, include_loc=include_loc)
        # Process includes first (pre-order to mimic compilation)
        in_comment = False
        for i, line in enumerate(lines):
            stripped = line.strip().lower()
            if stripped == '$ontext':
                in_comment = True
                continue
            elif stripped == '$offtext':
                in_comment = False
                continue
            elif line.strip().startswith('*') or in_comment:
                continue
            m = INCLUDE_RE.match(line)
            if m:
                rest = m.group(2).strip()  # Everything after $include / $batinclude

                # Extract the first "argument" — either quoted string or first token
                # Handle both quoted and unquoted paths
                path_match = re.match(r'''^\s*(?:"([^"]+)"|'([^']+)'|(\S+))''', rest)
                if path_match:
                    # Take the first non-None group: "..." or '...' or bare word
                    inc_path = next(g for g in path_match.groups() if g is not None)
                else:
                    # Fallback: take up to first space (rare)
                    inc_path = rest.split()[0]

                # Clean quotes if still present (in case fallback used)
                inc_path = inc_path.strip('"\'')
                
                # Substitute %X% with '/' as path separator
                inc_path = inc_path.replace('%X%', '/')
                
                # Substitute %REGION% with REGION value
                inc_path = inc_path.replace('%REGION%', REGION)
                
                inc_full = os.path.join(base_dir, inc_path)
                include_type = 'batinclude' if m.group(1) else 'include'
                _load(inc_full, include_loc=(fp, i+1, include_type), base_dir=base_dir)
        ordered.append((full, lines))

    _load(root_path, base_dir=base_dir)
    print(f"\rLoaded {len(ordered)} file(s).{'                                    '}", flush=True)
    return ordered

# ----------------------------
# Parser
# ----------------------------

def parse_code(files: List[Tuple[str, List[str]]]) -> Tuple[Dict[str, Symbol], List[ModelInfo], List[SolveInfo]]:
    symbols: Dict[str, Symbol] = {}
    models: List[ModelInfo] = []
    solves: List[SolveInfo] = []

    def ensure_symbol(name: str, stype: str) -> Symbol:
        name = norm_ident(name)
        if name not in symbols:
            symbols[name] = Symbol(name=name, stype=stype)
        else:
            # upgrade stype if unknown
            if symbols[name].stype == "unknown":
                symbols[name].stype = stype
        return symbols[name]

    num_files = len(files)
    for fidx, (fp, lines) in enumerate(files, start=1):
        status_msg = f"Parsing: {os.path.basename(fp)} ({fidx}/{num_files})                "
        print(f"\r{status_msg}", end="", flush=True)
        in_comment = False
        i = 0
        while i < len(lines):
            line = lines[i].rstrip('\n')
            stripped = line.strip().lower()
            if stripped == '$ontext':
                in_comment = True
                i += 1
                continue
            elif stripped == '$offtext':
                in_comment = False
                i += 1
                continue
            elif line.strip().startswith('*') or in_comment:
                i += 1
                continue
            # Merge continuation lines if trailing comma and next line
            # (simple heuristic)
            if i + 1 < len(lines) and lines[i+1].lstrip().startswith(','):
                line += ' ' + lines[i+1].strip()
                i += 1

            # Declarations
            if DECL_START_RE.match(line):
                # Variables special case
                vdm = VAR_DECL_RE.match(line)
                if vdm:
                    var_list = vdm.group(2)
                    for v in re.split(r",", var_list):
                        vname = v.strip()
                        if not vname:
                            continue
                        sym = ensure_symbol(vname, 'variable')
                        sym.decls.append(Definition(kind='declaration', text=line, loc=SourceLoc(fp, i+1)))
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
                    while j < len(lines):
                        l2 = lines[j].rstrip('\n')
                        if l2.strip() == ';':
                            table_lines.append(l2)
                            j += 1
                            break
                        if DECL_START_RE.match(l2) or INCLUDE_RE.match(l2) or SOLVE_RE.search(l2) or MODEL_RE.search(l2):
                            break
                        table_lines.append(l2)
                        j += 1
                    # Parse only small tables line by line; for large tables, skip to avoid performance issues
                    if len(table_lines) > 100:
                        values = {}
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
                            row_line = lines[r].rstrip('\n')
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
                    defn = Definition(kind='table', text=line, loc=SourceLoc(fp, i+1), deps=set(), values=values, skipped=skipped)
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
                # Extract names (split by commas until ';')
                decl_body = line
                decl_parts = decl_body.split(None, 1)
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
                        sym.decls.append(Definition(kind='declaration', text=line, loc=SourceLoc(fp, i+1)))
                # For parameters and scalars, skip over data definitions bounded by / ... /
                if stype in ['parameter', 'scalar']:
                    j = i + 1
                    in_data = False
                    while j < len(lines):
                        l2 = lines[j].rstrip('\n').strip()
                        if l2.startswith('/'):
                            in_data = True
                        if in_data and l2.endswith('/'):
                            i = j
                            break
                        if DECL_START_RE.match(lines[j]):
                            break
                        j += 1
                i += 1
                continue

            # Model membership
            mm = MODEL_RE.search(line)
            if mm:
                mname = mm.group(1)
                eqs = [e.strip() for e in mm.group(2).split(',') if e.strip()]
                models.append(ModelInfo(name=mname, equations=eqs, loc=SourceLoc(fp, i+1)))
                i += 1
                continue

            # Solve detection
            sm = SOLVE_RE.search(line)
            if sm:
                solves.append(SolveInfo(model=sm.group(1), sense=sm.group(2).lower(), objvar=sm.group(3), loc=SourceLoc(fp, i+1)))
                i += 1
                continue

            # Check for multi-line equation start (has .. but no ; at end)
            eq_start_re = re.compile(r"^\s*([A-Za-z_]\w*)\s*\.\.\s*(.+?)\s*=(l|L|e|E|g|G)=\s*(.*)$", re.IGNORECASE)
            em_start = eq_start_re.match(line)
            if em_start and not line.strip().endswith(';'):
                # Accumulate multi-line equation
                ename = em_start.group(1)
                accumulated = line
                j = i + 1
                while j < len(lines):
                    next_line = lines[j].rstrip('\n')
                    accumulated += ' ' + next_line.strip()
                    if next_line.strip().endswith(';'):
                        # Found the end, parse the full equation
                        em = EQUATION_RE.match(accumulated)
                        if em:
                            lhs, sense, rhs = em.groups()[1:4]  # Skip ename
                            sym = ensure_symbol(ename, 'equation')
                            deps = find_idents(lhs) | find_idents(rhs)
                            sym.defs.append(Definition(kind='equation', text=accumulated, loc=SourceLoc(fp, i+1), deps=deps, lhs=lhs.strip()))
                        i = j
                        break
                    j += 1
                else:
                    # No semicolon found, treat as single line attempt
                    pass
                i += 1
                continue

            # Equation definitions (single line)
            em = EQUATION_RE.match(line)
            if em:
                ename, lhs, sense, rhs = em.groups()
                sym = ensure_symbol(ename, 'equation')
                deps = find_idents(lhs) | find_idents(rhs)
                sym.defs.append(Definition(kind='equation', text=line, loc=SourceLoc(fp, i+1), deps=deps, lhs=lhs.strip()))
                i += 1
                continue

            # Check for multi-line assignment start
            am_start_re = re.compile(r"^\s*([A-Za-z_]\w*)(\s*\([^)]*\))?", re.IGNORECASE)
            am_start = am_start_re.match(line)
            if am_start and not line.strip().endswith(';'):
                atgt = am_start.group(1)
                accumulated = line
                j = i + 1
                while j < len(lines):
                    next_line = lines[j].rstrip('\n')
                    accumulated += ' ' + next_line.strip()
                    if next_line.strip().endswith(';'):
                        # Check if it's an assignment by having =
                        if '=' in accumulated:
                            eq_pos = accumulated.find('=')
                            expr = accumulated[eq_pos+1:].strip().rstrip(';')
                            deps = find_idents(expr)
                            sym = ensure_symbol(atgt, symbols.get(atgt, Symbol(atgt, 'unknown')).stype or 'unknown')
                            sym.defs.append(Definition(kind='assignment', text=accumulated, loc=SourceLoc(fp, i+1), deps=deps, lhs=atgt.strip()))
                        i = j
                        break
                    j += 1
                else:
                    # No semicolon found, skip
                    pass
                i += 1
                continue

            # Assignments (parameters/scalars/etc.)
            am = ASSIGN_RE.match(line)
            if am:
                tgt = am.group(1)
                expr = am.group(3)
                deps = find_idents(expr)
                sym = ensure_symbol(tgt, symbols.get(tgt, Symbol(tgt, 'unknown')).stype or 'unknown')
                sym.defs.append(Definition(kind='assignment', text=line, loc=SourceLoc(fp, i+1), deps=deps, lhs=tgt.strip()))
                i += 1
                continue

            i += 1

    return symbols, models, solves

# ----------------------------
# Tracing utilities
# ----------------------------

def trace_symbol(symbols: Dict[str, Symbol], name: str, depth: int = 0, visited: Optional[Set[str]] = None) -> List[str]:
    """Return textual trace for a symbol: where its values come from (assignments, tables, deps)."""
    name = norm_ident(name)
    if visited is None:
        visited = set()
    if name in visited:
        return ["  "*depth + f"↪ {name} (cycle detected)"]
    visited.add(name)
    out: List[str] = []
    sym = symbols.get(name)
    if not sym:
        out.append("  "*depth + f"✖ {name}: not declared/defined in parsed files")
        return out
    out.append("  "*depth + f"• {name} [{sym.stype}]")
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
            else:
                out.append("  "*depth + f"  ├─ {d.kind} at {d.loc.file}:{d.loc.line}")
            for dep in sorted(d.deps):
                if dep == name:
                    continue
                out.extend(trace_symbol(symbols, dep, depth+1, visited))
    else:
        out.append("  "*depth + f"  ├─ no definitions found (may be loaded via GDX or includes not captured)")
    return out


def extract_objective(symbols: Dict[str, Symbol], solve: Optional[SolveInfo]) -> Tuple[Optional[SolveInfo], Optional[Definition]]:
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


def explain_equation(symbols: Dict[str, Symbol], eq_name: str) -> List[str]:
    eq = symbols.get(eq_name)
    if not eq or eq.stype != 'equation':
        return [f"Equation '{eq_name}' not found."]
    out: List[str] = []
    for d in eq.defs:
        if d.kind != 'equation':
            continue
        out.append(f"Equation {eq_name} defined at {d.loc.file}:{d.loc.line}")
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
    ap = argparse.ArgumentParser(description="Trace raw data sources in a GAMS LP model (CPLEX). Parses with --parse, then loads from gams_trace.parse for queries.")
    ap.add_argument('--parse', help='Parse root .gms file and save parse data to gams_trace.parse; lists all solve statements')
    ap.add_argument('--list-solves', action='store_true', help='List all detected solve statements')
    ap.add_argument('--solve', action='store_true', help='Show details for a specific solve statement')
    ap.add_argument('--objective', action='store_true', help='Trace objective variable/equation')
    ap.add_argument('--equation', help='Trace a specific equation by name')
    ap.add_argument('--dump-symbol', help='Trace a symbol (parameter/scalar/table/variable)')
    ap.add_argument('solve_number', nargs='?', type=int, help='Solve number (1+) when using --solve or --objective (optional: will prompt if omitted)')
    args = ap.parse_args()

    if len(sys.argv) == 1 and not os.path.exists('gams_trace.parse'):
        ap.print_help()
        sys.exit(1)

    symbols = None
    models = None
    solves = None

    if args.parse:
        if any([args.solve_number, args.list_solves, args.solve, args.objective, args.equation, args.dump_symbol]):
            print("Error: --parse is standalone for parsing. Other flags require loading from gams_trace.parse.")
            ap.print_help()
            sys.exit(1)
        # Parse from root
        try:
            files = load_gms(args.parse)
        except Exception as e:
            print(f"Error loading files: {e}")
            sys.exit(1)

        symbols, models, solves = parse_code(files)

        # Serialize
        with open('gams_trace.parse', 'wb') as f:
            pickle.dump((symbols, models, solves), f)
        print("\rParsing complete, saved parse tree to gams_trace.parse", flush=True)

        # Do list-solves
        print("\nSolve statements:")
        with open('gams_trace.solves', 'w') as f:
            f.write(f"parse: {args.parse}\n")
            for idx, s in enumerate(solves, start=1):
                print(f"{idx}. model={s.model} sense={s.sense} objvar={s.objvar} at {s.loc.file}:{s.loc.line}")
                f.write(f"{idx}|model={s.model}|sense={s.sense}|objvar={s.objvar}|file={s.loc.file}|line={s.loc.line}\n")

    else:
        if any([args.list_solves, args.solve, args.objective, args.equation, args.dump_symbol, args.solve_number]):
            if not os.path.exists('gams_trace.parse'):
                print("Error: Parsed data file 'gams_trace.parse' does not exist. Run with --parse first to parse and save data.")
                sys.exit(1)
            with open('gams_trace.parse', 'rb') as f:
                symbols, models, solves = pickle.load(f)
            print("Parsed data loaded from gams_trace.parse")
        else:
            # no flags, print summary only if parse exists, else help
            if os.path.exists('gams_trace.parse'):
                with open('gams_trace.parse', 'rb') as f:
                    symbols, models, solves = pickle.load(f)
                print("Parsed data loaded from gams_trace.parse")
            else:
                ap.print_help()
                sys.exit(1)

    if args.list_solves:
        print("Solve statements:")
        for idx, s in enumerate(solves, start=1):
            print(f"{idx}. model={s.model} sense={s.sense} objvar={s.objvar} at {s.loc.file}:{s.loc.line}")
        print()
        sys.exit(0)

    selected_solve = None
    if args.solve or args.objective:
        if args.solve_number is None:
            print("Available solves:")
            for idx, s in enumerate(solves, start=1):
                print(f"{idx}. model={s.model} sense={s.sense} objvar={s.objvar} at {s.loc.file}:{s.loc.line}")
            while True:
                try:
                    solve_num = int(input(f"Enter solve number (1-{len(solves)}): "))
                    if 1 <= solve_num <= len(solves):
                        selected_solve = solves[solve_num - 1]
                        break
                except ValueError:
                    pass
        else:
            idx = args.solve_number
            if idx < 1 or idx > len(solves):
                print("Invalid solve number. Available solves:")
                for idx, s in enumerate(solves, start=1):
                    print(f"{idx}. model={s.model} sense={s.sense} objvar={s.objvar} at {s.loc.file}:{s.loc.line}")
                sys.exit(1)
            selected_solve = solves[idx - 1]

    if args.solve:
        s = selected_solve
        print(f"Solve: model={s.model}, sense={s.sense}, objvar={s.objvar} at {s.loc.file}:{s.loc.line}")
        print()

    if args.objective:
        target_solve = selected_solve
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

    if args.equation:
        for ln in explain_equation(symbols, args.equation):
            print(ln)
        print()

    if args.dump_symbol:
        for ln in trace_symbol(symbols, args.dump_symbol):
            print(ln)
        print()

    if not (args.parse or args.list_solves or args.solve or args.objective or args.equation or args.dump_symbol or args.solve_number):
        print("Parsed symbols summary:")
        for name, sym in sorted(symbols.items()):
            print(f"- {name} [{sym.stype}] defs={len(sym.defs)}")
        print("\nTip: use --objective, --equation EQNAME, or --dump-symbol NAME to see detailed traces.")

if __name__ == '__main__':
    main()
