Great question, Albert! Since you’re solving a **linear program (LP) with CPLEX from GAMS**, the core of what you’re asking for is a **static provenance trace**: given your `.gms` codebase, show *where* the numbers in the objective, constraints, and bounds **come from** (tables, parameter assignments, includes, etc.).

Below I’ve created a Python script that:

*   Recursively loads your GAMS sources and resolves `$include` / `$batinclude`
*   Parses common declarations: `Sets`, `Parameters/Scalars`, `Tables`, `Variables`, `Equations`, `Model`
*   Detects your `solve ... using lp minimizing|maximizing ...`
*   Builds a **dependency graph** of symbols referenced by assignments/equations
*   Lets you query and **trace** the origin of numbers (e.g., parameters feeding the objective or a constraint’s RHS)

You can download it here:

[📄 gams\_trace.py](blob:https://outlook.office.com/ff50ec8f-5232-43d4-9cab-b6e7515e6457)

***

## What the script does

**Capabilities** (static analysis):

*   **Includes:** Resolves `$include` and `$batinclude` to create a unified view of the codebase.
*   **Tables:** Parses simple rectangular `Table` blocks and captures numeric entries (row/column keyed values).
*   **Assignments:** Records parameter/scalar assignments and their dependencies (e.g., `a(i) = b(i) + 0.1*c;` → `b` and `c`).
*   **Equations:** Extracts equation definitions (`eq.. LHS =l= RHS;`) and their symbol dependencies.
*   **Solve detection:** Finds the LP solve statement, sense (`minimizing` or `maximizing`), and objective variable.
*   **Tracing:** Given a target symbol or equation, recursively traces dependencies down to raw sources (e.g., table entries or literals).

**Limitations** (by design, to keep it broadly usable):

*   Does not evaluate compile-time conditionals (`$if`, `$eval`) or macros.
*   Does not process GDX I/O (e.g., `execute_load`, `execute_unload`).
*   Table parsing supports common rectangular numeric tables; unusual layouts may need adjustments.
*   Coefficient extraction in expressions is heuristic—it traces *parameters used* rather than numerically building the full matrix.

***

## Usage

Once the script is on your machine:

```bash
python gams_trace.py --root path/to/main.gms --show-solve
python gams_trace.py --root path/to/main.gms --objective
python gams_trace.py --root path/to/main.gms --equation eq_supply
python gams_trace.py --root path/to/main.gms --dump-symbol A
```

### Typical outputs

*   `--show-solve`:  
    Shows where the LP solve is declared: model name, sense, objective variable, and file:line.

*   `--objective`:  
    Prints the objective variable and attempts to locate the **objective-defining equation** (e.g., `obj .. Z =e= sum(i, c(i) * x(i));`).  
    Then it **traces** all parameters in that expression (e.g., `c(i)`) back to table entries or assignment lines.

*   `--equation eq_name`:  
    Prints the equation definition and traces **all parameters** appearing on LHS/RHS. Useful for seeing where a constraint’s RHS originates (e.g., `demand(j)` coming from a table or assignment).

*   `--dump-symbol A`:  
    Traces a parameter/scalar/table/variable symbol back to its **raw sources** (table values and/or assignment lines), following any layers of dependencies.

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

*   `--objective`: will show the `obj` equation and trace `cost` to the `Table` block (and print entries).
*   `--equation satisfy`: will trace `demand(j)` to the inline parameter assignment.

***

## How it’s built (technical notes)

*   **Parsing** uses regular expressions for common GAMS patterns:
    *   Declarations (`Sets`, `Parameters`, `Tables`, `Variables`, `Equations`)
    *   Assignments: `name(index?) = expression;`
    *   Equations: `eq .. LHS =l|e|g= RHS;`
    *   Model membership: `model m / eq1, eq2 /;`
    *   Solve: `solve m using lp minimizing Z;`
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
python gams_trace.py --root /path/to/your/main.gms --objective
```

> If your codebase has nested includes, the script will follow `$include` and `$batinclude` relative to the included file’s directory.

***

## Extending the script to your needs

If you want deeper matrix-level details (e.g., extract exact coefficients per variable per constraint), we can extend the parser to:

*   Identify **variable terms** (e.g., `param(i,j)*x(i,j)`) and separate **parameter** vs. **variable** tokens
*   Expand common aggregates (`sum(i, ...)`) and track indices to map to rows/columns
*   Optionally emit a **symbolic sparse matrix** listing constraints vs variables and the **source** of each coefficient (assignment/table and file:line).

***

## Next steps (tailored to IBF workflows)

*   If you share a *small anonymized snippet* of your `.gms` (especially your objective and 2–3 representative constraints), I can tweak the parser to your style (e.g., handling `alias`, special functions, or your typical table layouts).
*   Do you also use **GDX** loads for data? If yes, I can add a light **GDX reference detector** (to flag where data leaves the code into external files), even if we don’t parse the GDX contents here.

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
