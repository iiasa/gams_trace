The `gams_trace.py` script:

*   Recursively loads your GAMS sources and resolves `$include` / `$batinclude`
*   Parses common declarations: `Sets`, `Parameters/Scalars`, `Tables`, `Variables` (including multi-line declarations where variable lists may be comma-separated across lines, and classifying by type such as FREE, POSITIVE, etc.), `Equations`, `Model`
*   Detects your `solve ... using ... minimizing|maximizing ...` (any solver)
*   Builds a **dependency graph** of symbols referenced by assignments/equations
*   Recognizes and resolves `alias` declarations for set alternative names
*   Lets you query and **trace** the origin of data (e.g., parameters feeding the objective or a constraint’s RHS)

> ⚠️ **Warning**
> The parser is not a full GAMS parser, the script does not compile nor execute GAMS code. Hence:
> - Conditional code execution is not resolved when tracing.
> - Exceptional GAMS syntax constructs will be handled incorrectly.
> - Results should be taken with a grain of salt: useful for exploration and discovery, not useful for obtaining hard truth.

***

## What the script does

**Capabilities** (static analysis):

*   **Includes:** Recursively resolves `$include` and `$batinclude`, inlining the included file contents at the point of inclusion (matching GAMS compilation behavior). Allows multiple inclusions of the same file. Ignores comments (* lines and $ontext/$offtext blocks) during parsing to avoid interference with code analysis. Excludes .csv files bracketed in $ondelim/$offdelim. For $batinclude, handles argument substitution (e.g., `%1%` replaced with first argument).
*   **Declaration Parsing:** Handles multi-line declarations for parameters, scalars, and sets interrupted by comments or whitespace, but stops accumulation when encountering execution statements (e.g., LOOP, IF) even without semicolons, following GAMS rules that declarations may not contain flow-control blocks.
*   **Line Tracking:** Maintains original source file and line number for every line in the merged code, ensuring accurate source attribution in output.
*   **GDX Loading:** Parses `$GDXIN`, `$LOAD`, and `$LOADDC` statements, recording symbol origins from external GDX files.
*   **Tables:** Parses simple rectangular `Table` blocks and captures numeric entries (row/column keyed values). Captures dimensions where specified.
*   **Sets/Parameters/Scalars:** Parses declarations and captures dimensions where specified (e.g., `set A(i,j)` records dimensions `['i', 'j']`).
*   **Assignments:** Records parameter/scalar assignments and their dependencies (e.g., `a(i) = b(i) + 0.1*c;` → `b` and `c`).
*   **Equations:** Extracts equation definitions, handling multi-line equations where the '..' and sense may span across lines, and their symbol dependencies.
*   **Solve detection:** Finds solve statements, solver types, sense (`minimizing` or `maximizing`), and objective variable.
*   **Tracing:** Given a target symbol or equation, recursively traces dependencies down to raw sources (e.g., table entries, literals, or GDX files).

**Limitations** (by design, to keep it broadly usable):

*   Does not evaluate compile-time conditionals (`$if`, `$eval`) or macros.
*   Does not process other GDX I/O (e.g., `execute_load`, `execute_unload`).
*   Does not load or parse .csv files; tracks CSV file paths for set/table sources.
*   Table parsing supports common rectangular numeric tables; unusual layouts may need adjustments.
*   Coefficient extraction in expressions is heuristic—it traces *parameters used* rather than numerically building the full matrix.

**GDX Loading**:
*   Traces symbol assignments from GDX files loaded via `$GDXIN`, `$LOAD`, and `$LOADDC`.
*   Records GDX file path, source file location, and line number for tracing.

***

## Usage

Running without arguments or passing `-h` or --help` as a first arguement displays usage information. The first arguement selects a subcommand that invokes a particular script action. The subcommands are `parse`, `save`, `list`, `show`, `trace`, and `trace_with_sets`.

## Parse


First, parse the GAMS model (saves data to `gams_trace.parse`):

```bash
python gams_trace.py parse path/to/main.gms
```

This parses the provided root and included `.gms` files for solve statements (any solver), saves parsed data to `gams_trace.parse`, and lists aggregate counts of solves and symbols (including unidentified symbols); detailed solve information is persisted in `gams_trace.solves`.

## Save

After parsing, you can save the merged and decommented GAMS code:

```bash
python gams_trace.py save path/to/merged.gms
```

This saves the merged decommented GAMS code (all `$include` and `$batinclude` files inlined, comments removed) to the specified output file for review.

## List

The `list` subcomments lists parsed symbols and solve statements. Without futher parameters, this subcommand displays a summary of symbol types and counts:

```bash
python gams_trace.py list
```

Example output:
```
Parsed data loaded from gams_trace.parse
Parsed symbols summary:
equations: 73
parameters: 1370
scalars: 9
sets: 219
tables: 61
unknown: 254
variables: 76
```

To list the parsed symbols of a given type, invoke:

```bash
python gams_trace.py list solves
python gams_trace.py list sets
python gams_trace.py list parameters
python gams_trace.py list scalars
python gams_trace.py list tables
python gams_trace.py list variables
python gams_trace.py list equations
python gams_trace.py list unknowns
```

To list parsed solve statements, invoke:

```bash
python gams_trace.py list solves
```

Detailed explanation:

*   `list sets`:
    Lists all parsed sets alphabetically.

*   `list parameters`:
    Lists all parsed parameters alphabetically.

*   `list scalars`:
    Lists all parsed scalars alphabetically.

*   `list tables`:
    Lists all parsed tables alphabetically.

*   `list variables`:
    Lists all parsed variables grouped by variable type (e.g., Free Variables: - X, - Y; Positive Variables: - Z).

*   `list equations`:
    Lists all parsed equations alphabetically.

*   `list unknowns`:
    Lists all symbols encountered that could not be classified or linked to declarations.

*   `list solves`:
    Lists all detected solve statements with index numbers.

## Show

Shows the definition location, type, dimensions (if any), and first ≤5 lines of the declaration or definition for the specified symbol or solve statement. Works for any symbol type including thoe of unknown type. Solve statements are specified by an index number.

Note: Symbol name lookups are case-insensitive. The original case from the code is displayed in output.

```bash
python gams_trace.py show MY_SYMBOL
python gams_trace.py show SOLVE_INDEX_NUMBER
```

## Trace

The `trace` and `trace_with_sets` subcommands trace dependencies. The former excludes sets from the tracing for brevity. Invocations:

```bash
python gams_trace.py trace objective
python gams_trace.py trace objective 1
python gams_trace.py trace eq_supply
python gams_trace.py trace A
```

Trace dependencies (including sets):

```bash
python gams_trace.py trace_with_sets objective
python gams_trace.py trace_with_sets objective 1
python gams_trace.py trace_with_sets eq_supply
python gams_trace.py trace_with_sets A
python gams_trace.py trace_with_sets set_name
```

Typical outputs for tracing:

*   `trace objective`:
    Prompts for a solve number if not provided, then prints the objective variable and attempts to locate the **objective-defining equation** (e.g., `obj .. Z =e= sum(i, c(i) * x(i));`).
    Then it **traces** all parameters in that expression (e.g., `c(i)`) back to table entries or assignment lines, excluding any sets from the output.

*   `trace <eq_name>`:
    Prints the equation definition and traces **all parameters** appearing on LHS/RHS, excluding sets. Works globally without solve context.

*   `trace <symbol>`:
    Traces a parameter/scalar/table/variable symbol back to its **raw sources** (table values and/or assignment lines), following any layers of dependencies, excluding sets. Works globally without solve context.

*   `trace_with_sets objective`:
    Same as `trace objective`, but includes sets in dependency outputs if present.

*   `trace_with_sets <eq_name>`:
    Prints the equation definition and traces **all parameters** appearing on LHS/RHS, including sets. Works globally without solve context.

*   `trace_with_sets <symbol>`:
    Traces a parameter/scalar/table/variable symbol back to its **raw sources** (table values and/or assignment lines), following any layers of dependencies, including sets. Works globally without solve context.

*   `trace_with_sets <set>`:
    Traces a set to its raw sources, following dependencies including other sets.

**Note on Cycle Detection:**
*   If tracing outputs `↪ {symbol_name} (cycle detected)`, it indicates a circular dependency in the symbol reference graph.
*   This prevents infinite recursion when symbols reference each other directly or indirectly (e.g., `param a = b;` and `param b = a + 1;`).
*   Cycles are rare in valid models but may signal potential logical errors.

***

### Example (mini demonstration)

Suppose part of your GAMS code looks like:

```gams
Sets
    i /i1*i3/
    j /j1*j2/;

Table cost(i,j)
         j1   j2
i1       4    5
i2       6    3
i3       2    7
;

Parameters demand(j) / j1 8, j2 6 /;

Positive Variables x(i,j), Z;
Equations obj, supply(i), satisfy(j);

obj..     Z =e= sum((i,j), cost(i,j)*x(i,j));
supply(i).. sum(j, x(i,j)) =l= 10;
satisfy(j).. sum(i, x(i,j)) =g= demand(j);

Model m / obj, supply, satisfy /;
solve m using lp minimizing Z;
```

*   `trace objective`: will show the `obj` equation and trace `cost` to the `Table` block (and print entries).
*   `trace satisfy`: will trace `demand(j)` to the inline parameter assignment.

***

## How it’s built (technical notes)

*   **Parsing** uses regular expressions for common GAMS patterns, with line accumulation for multi-line constructs to handle cases where semicolons appear on different lines than the start:
    *   Declarations (`Sets`, `Parameters`, `Tables`, `Variables`, `Equations`)
    *   Assignments: `name(index?) = expression;`
    *   Equations: `eq .. LHS =l|e|g= RHS;`
    *   Model membership: `model m / eq1, eq2 /;`
    *   Solve: `solve m using ... minimizing|maximizing Z;`
    *   GDX loads: `$gdxin`, `$load`, `$loaddc`
*   It computes and prints **dependency chains**. For example, if `c(i) = base(i) + delta;` and `base(i)` comes from a `Table`, you’ll see:
```
        • c [parameter]
          ├─ assignment at mymodel.gms:42: c(i) = base(i) + delta;
            • base [parameter]
              ├─ table at data.gms:12 with 6 numeric entries
            • delta [scalar]
              ├─ assignment at params.gms:7: delta = 0.15;
```

***

## Installation / Run

1.  Save the scruot locally.
2.  Run it with Python 3:

```bash
python gams_trace.py parse /path/to/your/main.gms
python gams_trace.py trace objective 1
```

> If your codebase has nested includes, the script will follow `$include` and `$batinclude` relative to the root file’s directory.

***

## Possible future features

*   For symbols loaded from GDX files, tracing now shows the GDX file path and load location. Could analyze GDX contents to obtain further detail.
*   Identify **variable terms** (e.g., `param(i,j)*x(i,j)`) and separate **parameter** vs. **variable** tokens
*   Expand common aggregates (`sum(i, ...)`) and track indices to map to rows/columns
*   Optionally emit a **symbolic sparse matrix** listing constraints vs variables and the **source** of each coefficient (assignment/table and file:line).
*   Produce a per-constraint JSON (variables → coefficients → source lines).

***

## References (conceptual background)

While the script is standalone and doesn’t rely on external APIs, its logic mirrors common GAMS practices:

*   GAMS documentation on data declarations and equations (Sets/Parameters/Tables/Variables/Equations; solves) explains how model components are defined and referenced, which is what we statically trace here (GAMS User’s Guide—Language Concepts, GAMS—Modeling Basics).
*   The GAMS solve invocation `solve m using ...` and objective sense `minimizing|maximizing` follows the standard GAMS solve syntax (GAMS—Solve Statement).

> These references describe the constructs the parser targets and the conventions (e.g., `obj .. Z =e= expr;`) that let us identify and trace objective and constraints.

*   GAMS documentation on data declarations and equations (Sets/Parameters/Tables/Variables/Equations; solves) explains how model components are defined and referenced, which is what we statically trace here (GAMS User’s Guide—Language Concepts, GAMS—Modeling Basics).

