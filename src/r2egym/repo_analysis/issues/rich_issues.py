rich_issues = [
    ### issue 1
    r"""
Description

When rendering a `Table` with `show_lines=True` and a title, the title row does not
account for the extra horizontal lines between rows. This causes the table to render
with misaligned borders when the title is longer than the combined column widths.

Steps to Reproduce

```python
from rich.table import Table
from rich.console import Console

console = Console(width=60)
table = Table(title="A Long Title For This Table", show_lines=True)
table.add_column("Name")
table.add_column("Value")
table.add_row("foo", "1")
table.add_row("bar", "2")
console.print(table)
```

Expected behavior: The table renders with properly aligned borders and the title
centered above the table.

Actual behavior: The top border of the table is misaligned or the title is truncated
incorrectly.
""",
    ### issue 2
    r"""
Description

`Syntax` highlight with `line_range` raises an `IndexError` when the specified range
extends beyond the actual number of lines in the source code.

Steps to Reproduce

```python
from rich.syntax import Syntax

code = "x = 1\ny = 2\n"
syntax = Syntax(code, "python", line_numbers=True, line_range=(1, 10))

from rich.console import Console
console = Console()
console.print(syntax)
```

Expected behavior: Lines within range are displayed; out-of-range lines are silently
ignored or the output stops at the last available line.

Actual behavior: An `IndexError` is raised because the code tries to access lines
that do not exist.
""",
]
