#!/usr/bin/env python3
"""
GAMS → CPLEX LP Data Trace Tool
--------------------------------

This script performs static analysis on a GAMS codebase (.gms files) to help you
trace *where raw data values come from* for key aspects of an LP model solved
with CPLEX. It:

1) Recursively loads GAMS source (resolves $include / $batinclude).
2) Parses declarations (Sets, Parameters/Scalars, Tables, Variables, Equations, Model).
3) Extracts assignments and equation definitions, and builds a dependency graph.
4) Identifies the LP solve (solve ... using lp minimizing/maximizing ...).
5) Provides query functions to trace:
   - Objective data sources (parameters in the objective expression).
   - Constraint RHS/LHS parameter sources.
   - Variable bounds if assigned via parameters.

NOTE: This is a heuristic/static parser. GAMS is flexible and rich; the tool does
not evaluate conditional compilation ($if, $eval), embedded code, GDX I/O, nor the
full grammar. It focuses on common patterns to provide transparent provenance.

Usage examples:

  python gams_trace.py --root main.gms --show-solve
  python gams_trace.py --root main.gms --objective
  python gams_trace.py --root main.gms --equation eq_demand
  python gams_trace.py --root main.gms --dump-symbol A
"""

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

# REGION substitution for include paths
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
    values: Dict[Tuple[str, ...], float] = field(default_factory=dict)  # for tables

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
EQUATION_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*\.\.\s*(.+?)\s*=(l|e|g)=\s*(.+?);\s*$", re.IGNORECASE)
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

    def _load(fp: str):
        full = os.path.abspath(fp)
        if full in visited:
            return
        if not os.path.exists(full):
            raise FileNotFoundError(f"Included file not found: {fp}")
        visited.add(full)
        with open(full, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        # Process includes first (pre-order to mimic compilation)
        dirn = os.path.dirname(full)
        for i, line in enumerate(lines, start=1):
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
                
                inc_full = os.path.join(dirn, inc_path)
                _load(inc_full)
            ordered.append((full, lines))

    _load(root_path)
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

    for fp, lines in files:
        i = 0
        while i < len(lines):
            line = lines[i].rstrip('\n')
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
                    # Parse a simple table: header row with column keys, then rows as key + numbers
                    # This is heuristic and works for regular rectangular numeric tables.
                    header_cols: List[str] = []
                    values: Dict[Tuple[str, ...], float] = {}
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
                    defn = Definition(kind='table', text='\n'.join([line] + table_lines), loc=SourceLoc(fp, i+1), deps=set(), values=values)
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
                names = [n.strip() for n in re.split(r",", decl_body.split(None, 1)[1])]
                for raw in names:
                    # name may include dimension suffix (e.g., A(i,j)) — take identifier prefix
                    m = IDENT_RE.search(raw)
                    if m:
                        name = m.group(1)
                        sym = ensure_symbol(name, stype)
                        sym.decls.append(Definition(kind='declaration', text=line, loc=SourceLoc(fp, i+1)))
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

            # Equation definitions
            em = EQUATION_RE.match(line)
            if em:
                ename, lhs, sense, rhs = em.groups()
                sym = ensure_symbol(ename, 'equation')
                deps = find_idents(lhs) | find_idents(rhs)
                sym.defs.append(Definition(kind='equation', text=line, loc=SourceLoc(fp, i+1), deps=deps))
                i += 1
                continue

            # Assignments (parameters/scalars/etc.)
            am = ASSIGN_RE.match(line)
            if am:
                tgt = am.group(1)
                expr = am.group(3)
                deps = find_idents(expr)
                sym = ensure_symbol(tgt, symbols.get(tgt, Symbol(tgt, 'unknown')).stype or 'unknown')
                sym.defs.append(Definition(kind='assignment', text=line, loc=SourceLoc(fp, i+1), deps=deps))
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


def extract_objective(symbols: Dict[str, Symbol], solves: List[SolveInfo]) -> Tuple[Optional[SolveInfo], Optional[Definition]]:
    """Find objective variable from the solve statement and the equation that defines it (if present)."""
    if not solves:
        return None, None
    # Use the last solve statement
    s = solves[-1]
    objvar = symbols.get(s.objvar)
    # Find an equation that directly references objvar on LHS (pattern: obj .. Z =e= ...)
    for sym in symbols.values():
        if sym.stype == 'equation':
            for d in sym.defs:
                if d.kind == 'equation' and s.objvar in d.deps:
                    # heuristic: if objvar appears, assume this is the objective-defining equation
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
    ap = argparse.ArgumentParser(description="Trace raw data sources in a GAMS LP model (CPLEX).")
    ap.add_argument('--root', required=True, help='Root .gms file to analyze')
    ap.add_argument('--show-solve', action='store_true', help='Show detected solve statement(s)')
    ap.add_argument('--objective', action='store_true', help='Trace objective variable/equation')
    ap.add_argument('--equation', help='Trace a specific equation by name')
    ap.add_argument('--dump-symbol', help='Trace a symbol (parameter/scalar/table/variable)')
    args = ap.parse_args()

    try:
        files = load_gms(args.root)
    except Exception as e:
        print(f"Error loading files: {e}")
        sys.exit(1)

    symbols, models, solves = parse_code(files)

    if args.show_solve:
        if not solves:
            print("No 'solve ... using lp ...' statement found.")
        else:
            for s in solves:
                print(f"Solve: model={s.model}, sense={s.sense}, objvar={s.objvar} at {s.loc.file}:{s.loc.line}")
        print()

    if args.objective:
        s, obj_def = extract_objective(symbols, solves)
        if not s:
            print("No LP solve detected.")
        else:
            print(f"Objective: {s.sense} {s.objvar} (from solve at {s.loc.file}:{s.loc.line})")
            if obj_def:
                print(f"Objective equation at {obj_def.loc.file}:{obj_def.loc.line}")
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

    if not (args.show_solve or args.objective or args.equation or args.dump_symbol):
        print("Parsed symbols summary:")
        for name, sym in sorted(symbols.items()):
            print(f"- {name} [{sym.stype}] defs={len(sym.defs)}")
        print("\nTip: use --objective, --equation EQNAME, or --dump-symbol NAME to see detailed traces.")

if __name__ == '__main__':
    main()
