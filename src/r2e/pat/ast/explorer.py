import ast as _ast


def build_ast(source_code: str):
    """Parse source code and annotate parent references."""
    tree = _ast.parse(source_code)
    for node in _ast.walk(tree):
        for child in _ast.iter_child_nodes(node):
            child.parent = node  # type: ignore
    tree.parent = None  # type: ignore
    return tree


def find_def_in_ast(tree, name: str):
    """Find a function or class definition by name in the AST."""
    for node in _ast.walk(tree):
        if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef, _ast.ClassDef)):
            if node.name == name:
                return node
    return None
