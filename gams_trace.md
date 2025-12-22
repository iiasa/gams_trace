The `gams_trace.py` script:

*   Recursively loads your GAMS sources and resolves `$include` / `$batinclude`
*   Parses common declarations: `Sets`, `Parameters/Scalars`, `Tables`, `Variables` (including multi-line declarations where variable lists may be comma-separated across lines, and classifying by type such as FREE, POSITIVE, etc.), `Equations`, `Model`
*   Detects your `solve ... using lp minimizing|maximizing ...`
*   Builds a **dependency graph** of symbols referenced by assignments/equations
*   Lets you query and **trace** the origin of numbers (e.g., parameters feeding the objective or a constraint’s RHS)

***

## What the script does

**Capabilities** (static analysis):

*   **Includes:** Recursively resolves `$include` and `$batinclude`, inlining the included file contents at the point of inclusion (matching GAMS compilation behavior). Allows multiple inclusions of the same file. Ignores comments (* lines and $ontext/$offtext blocks) during parsing to avoid interference with code analysis. Excludes .csv files bracketed in $ondelim/$offdelim. For $batinclude, handles argument substitution (e.g., `%1%` replaced with first argument).
*   **Line Tracking:** Maintains original source file and line number for every line in the merged code, ensuring accurate source attribution in output.
*   **GDX Loading:** Parses `$GDXIN`, `$LOAD`, and `$LOADDC` statements, recording symbol origins from external GDX files.
*   **Tables:** Parses simple rectangular `Table` blocks and captures numeric entries (row/column keyed values).
*   **Assignments:** Records parameter/scalar assignments and their dependencies (e.g., `a(i) = b(i) + 0.1*c;` → `b` and `c`).
*   **Equations:** Extracts equation definitions, handling multi-line equations where the '..' and sense may span across lines, and their symbol dependencies.
*   **Solve detection:** Finds the LP solve statement, sense (`minimizing` or `maximizing`), and objective variable.
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

If parse data (from a prior run) exists, running without arguments displays a summary of symbol types and counts:

```bash
python gams_trace.py
```

Example output:
```
Parsed symbols summary:
equations: 6
parameters: 164
scalars: 9
sets: 173
tables: 61
variables: 0
unknown: 1348

Tip: use 'trace objective', 'trace <eqname>', or 'trace <symbol>' to see detailed traces.
```

First, parse the GAMS model (this will save parsed data to `gams_trace.parse` and list aggregate counts of solves and symbols, including the number of unidentified symbols; detailed solve information is saved to `gams_trace.solves`):

```bash
python gams_trace.py parse path/to/main.gms
python gams_trace.py save path/to/merged.gms
```

First, parse the GAMS model (saves data to `gams_trace.parse`):

```bash
python gams_trace.py parse path/to/main.gms
```

Then, save the merged and decommented source to a file:

```bash
python gams_trace.py save path/to/merged.gms
```

This saves the merged decommented source (all `$include` and `$batinclude` files inlined, comments removed) to the specified output file.

### Listing Commands

First, parse the model as above. Then list components. Note: Symbol name lookups (e.g., `list scalar w4`) are case-insensitive. The original case from the code is displayed in output.

```bash
python gams_trace.py list solves
python gams_trace.py list solve 1
python gams_trace.py list sets
python gams_trace.py list parameters
python gams_trace.py list scalars
python gams_trace.py list tables
python gams_trace.py list variables
python gams_trace.py list equations
python gams_trace.py list set MY_SET
python gams_trace.py list parameter MY_PARAM
python gams_trace.py list scalar MY_SCALAR
python gams_trace.py list table MY_TABLE
python gams_trace.py list variable MY_VAR
python gams_trace.py list equation MY_EQ
```

Typical outputs for listing:

*   `parse path/to/main.gms`:
    Parses the provided root and included `.gms` files for LP solve statements, saves parsed data to `gams_trace.parse`, and lists aggregate counts of solves and symbols (including unidentified symbols); detailed solve information is persisted in `gams_trace.solves`.

*   `list solves`:
    Lists all detected solve statements with IDs.

*   `list solve N`:
    Shows details for the N-th solve statement: model name, sense, objective variable, and file:line. Requires solve number.

*   `list sets`:
    Lists all defined sets (e.g., - Myset).

*   `list parameters`:
    Lists all defined parameters (e.g., - Myparam).

*   `list scalars`:
    Lists all defined scalars (e.g., - Mysc).

*   `list tables`:
    Lists all defined tables (e.g., - Mycost).

*   `list variables`:
    Lists all defined variables grouped by variable type (e.g., Free Variables: - X, - Y; Positive Variables: - Z).

*   `list equations`:
    Lists all defined equations (e.g., - Balance_eq).

*   `list set MY_SET`:
    Shows the definition location and first ≤5 lines of the set declaration.

*   `list parameter MY_PARAM`:
    Shows the definition location and first ≤5 lines of the parameter assignment/declaration.

*   Similarly for `scalar`, `table`, `variable`, `equation`: shows definition location and first ≤5 lines.

### Tracing Commands

Trace dependencies:

```bash
python gams_trace.py trace objective
python gams_trace.py trace objective 1
python gams_trace.py trace eq_supply
python gams_trace.py trace A
```

Typical outputs for tracing:

*   `trace objective`:
    Prompts for a solve number if not provided, then prints the objective variable and attempts to locate the **objective-defining equation** (e.g., `obj .. Z =e= sum(i, c(i) * x(i));`).
    Then it **traces** all parameters in that expression (e.g., `c(i)`) back to table entries or assignment lines.

*   `trace <eq_name>`:
    Prints the equation definition and traces **all parameters** appearing on LHS/RHS. Works globally without solve context.

*   `trace <symbol>`:
    Traces a parameter/scalar/table/variable symbol back to its **raw sources** (table values and/or assignment lines), following any layers of dependencies. Works globally without solve context.

***

## Example (mini demonstration)

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
    *   Solve: `solve m using lp minimizing Z;`
    *   GDX loads: `$gdxin`, `$load`, `$loaddc`
*   It computes and prints **dependency chains**. For example, if `c(i) = base(i) + delta;` and `base(i)` comes from a `Table`, you’ll see:
        • c [parameter]
          ├─ assignment at mymodel.gms:42: c(i) = base(i) + delta;
            • base [parameter]
              ├─ table at data.gms:12 with 6 numeric entries
            • delta [scalar]
              ├─ assignment at params.gms:7: delta = 0.15;

***

## Installation / Run

1.  Save the file locally (download link above).
2.  Run it with Python 3:

```bash
python gams_trace.py parse /path/to/your/main.gms
python gams_trace.py trace objective 1
```

> If your codebase has nested includes, the script will follow `$include` and `$batinclude` relative to the root file’s directory.

***

## Extending the script to your needs

If you want deeper matrix-level details (e.g., extract exact coefficients per variable per constraint), we can extend the parser to:

*   Identify **variable terms** (e.g., `param(i,j)*x(i,j)`) and separate **parameter** vs. **variable** tokens
*   Expand common aggregates (`sum(i, ...)`) and track indices to map to rows/columns
*   Optionally emit a **symbolic sparse matrix** listing constraints vs variables and the **source** of each coefficient (assignment/table and file:line).

***

## Next steps (tailored to IBF workflows)

*   If you share a *small anonymized snippet* of your `.gms` (especially your objective and 2–3 representative constraints), I can tweak the parser to your style (e.g., handling `alias`, special functions, or your typical table layouts).
*   For symbols loaded from GDX files, tracing now shows the GDX file path and load location. No parsing of GDX contents itself.

***

## References (conceptual background)

While the script is standalone and doesn’t rely on external APIs, its logic mirrors common GAMS practices:

*   GAMS documentation on data declarations and equations (Sets/Parameters/Tables/Variables/Equations; LP solves) explains how model components are defined and referenced, which is what we statically trace here (GAMS User’s Guide—Language Concepts, GAMS—Modeling Basics).
*   The CPLEX LP solve invocation `solve m using lp` and objective sense `minimizing|maximizing` follows the standard GAMS solve syntax (GAMS—Solve Statement, GAMS—LP with CPLEX).

> These references describe the constructs the parser targets and the conventions (e.g., `obj .. Z =e= expr;`) that let us identify and trace objective and constraints. (See **GAMS User’s Guide** sections and **CPLEX solver docs** above.)

***

Would you like me to adapt the script to:

*   produce a per-constraint JSON (variables → coefficients → source lines), or
*   generate a Markdown report summarizing objective and top constraints with provenance?

If you can share the names of the **objective variable** and a few **equation names** you care most about, I’ll tailor the output for your current project.
