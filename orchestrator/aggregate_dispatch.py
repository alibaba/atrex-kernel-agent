"""Build static, self-contained source blocks for workload-bucket aggregation."""

from __future__ import annotations

import ast
import symtable
from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddedBuckets:
    blocks: tuple[str, ...]
    entry_symbols: tuple[str, ...]
    future_features: tuple[str, ...]


class _ModuleImportCollector(ast.NodeVisitor):
    """Collect imports evaluated in a module body, excluding nested scopes."""

    def __init__(self) -> None:
        self.bindings: dict[str, set[tuple[str, ...]]] = {}
        self.future_features: set[str] = set()

    def _record(self, name: str, target: tuple[str, ...]) -> None:
        self.bindings.setdefault(name, set()).add(target)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            bound = alias.asname or alias.name.split(".", 1)[0]
            target = alias.name if alias.asname else alias.name.split(".", 1)[0]
            self._record(bound, ("import", target))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module == "__future__" and node.level == 0:
            self.future_features.update(alias.name for alias in node.names)
            return
        if node.level:
            raise ValueError("relative imports cannot be embedded in aggregate kernel.py")
        for alias in node.names:
            if alias.name == "*":
                raise ValueError("star imports cannot be embedded in aggregate kernel.py")
            bound = alias.asname or alias.name
            self._record(bound, ("from", node.module or "", alias.name))

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return

    def visit_ListComp(self, node: ast.ListComp) -> None:
        return

    def visit_SetComp(self, node: ast.SetComp) -> None:
        return

    def visit_DictComp(self, node: ast.DictComp) -> None:
        return

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        return


class _GlobalRenamer(ast.NodeTransformer):
    """Prefix one module's globals while preserving locals and import aliases."""

    def __init__(
        self,
        source: str,
        filename: str,
        mapping: dict[str, str],
    ) -> None:
        self.mapping = mapping
        self.table = symtable.symtable(source, filename, "exec")
        self._child_offsets: dict[tuple[int, str, str, int], int] = {}

    def _is_global(self, name: str) -> bool:
        if self.table.get_type() == "module":
            return True
        try:
            return self.table.lookup(name).is_global()
        except KeyError:
            return False

    def _renamed_binding(self, name: str) -> str:
        return self.mapping.get(name, name) if self._is_global(name) else name

    def _child_table(self, name: str, lineno: int, kind: str) -> symtable.SymbolTable:
        candidates = [
            child
            for child in self.table.get_children()
            if child.get_type() == kind
            and child.get_name() == name
            and child.get_lineno() == lineno
        ]
        if not candidates:
            candidates = [
                child
                for child in self.table.get_children()
                if child.get_type() == kind and child.get_name() == name
            ]
        key = (self.table.get_id(), kind, name, lineno)
        offset = self._child_offsets.get(key, 0)
        if offset >= len(candidates):
            raise ValueError(
                f"could not resolve symbol table for embedded {kind} {name!r} "
                f"at line {lineno}"
            )
        self._child_offsets[key] = offset + 1
        return candidates[offset]

    def _visit_body_in_child(
        self,
        body: list[ast.stmt],
        *,
        name: str,
        lineno: int,
        kind: str,
    ) -> list[ast.stmt]:
        parent = self.table
        self.table = self._child_table(name, lineno, kind)
        try:
            return [self.visit(item) for item in body]
        finally:
            self.table = parent

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in self.mapping and self._is_global(node.id):
            node.id = self.mapping[node.id]
        return node

    def visit_Global(self, node: ast.Global) -> ast.AST:
        node.names = [self.mapping.get(name, name) for name in node.names]
        return node

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> ast.AST:
        if node.name:
            node.name = self._renamed_binding(node.name)
        return self.generic_visit(node)

    def visit_MatchAs(self, node: ast.MatchAs) -> ast.AST:
        if node.name:
            node.name = self._renamed_binding(node.name)
        return self.generic_visit(node)

    def visit_MatchStar(self, node: ast.MatchStar) -> ast.AST:
        if node.name:
            node.name = self._renamed_binding(node.name)
        return node

    def _visit_function(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> ast.AST:
        source_name = node.name
        node.name = self._renamed_binding(node.name)
        node.decorator_list = [self.visit(item) for item in node.decorator_list]
        node.returns = self.visit(node.returns) if node.returns is not None else None
        arguments = (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
        for argument in arguments:
            if argument.annotation is not None:
                argument.annotation = self.visit(argument.annotation)
        if node.args.vararg and node.args.vararg.annotation is not None:
            node.args.vararg.annotation = self.visit(node.args.vararg.annotation)
        if node.args.kwarg and node.args.kwarg.annotation is not None:
            node.args.kwarg.annotation = self.visit(node.args.kwarg.annotation)
        node.args.defaults = [self.visit(item) for item in node.args.defaults]
        node.args.kw_defaults = [
            self.visit(item) if item is not None else None
            for item in node.args.kw_defaults
        ]
        node.body = self._visit_body_in_child(
            node.body,
            name=source_name,
            lineno=node.lineno,
            kind="function",
        )
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        return self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        return self._visit_function(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        source_name = node.name
        node.name = self._renamed_binding(node.name)
        node.decorator_list = [self.visit(item) for item in node.decorator_list]
        node.bases = [self.visit(item) for item in node.bases]
        node.keywords = [self.visit(item) for item in node.keywords]
        node.body = self._visit_body_in_child(
            node.body,
            name=source_name,
            lineno=node.lineno,
            kind="class",
        )
        return node

    def visit_Lambda(self, node: ast.Lambda) -> ast.AST:
        node.args.defaults = [self.visit(item) for item in node.args.defaults]
        node.args.kw_defaults = [
            self.visit(item) if item is not None else None
            for item in node.args.kw_defaults
        ]
        parent = self.table
        self.table = self._child_table("lambda", node.lineno, "function")
        try:
            node.body = self.visit(node.body)
        finally:
            self.table = parent
        return node

    def _visit_comprehension(self, node: ast.AST, name: str) -> ast.AST:
        parent = self.table
        self.table = self._child_table(name, node.lineno, "function")
        try:
            return self.generic_visit(node)
        finally:
            self.table = parent

    def visit_ListComp(self, node: ast.ListComp) -> ast.AST:
        return self._visit_comprehension(node, "listcomp")

    def visit_SetComp(self, node: ast.SetComp) -> ast.AST:
        return self._visit_comprehension(node, "setcomp")

    def visit_DictComp(self, node: ast.DictComp) -> ast.AST:
        return self._visit_comprehension(node, "dictcomp")

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> ast.AST:
        return self._visit_comprehension(node, "genexpr")


def _embed_one(
    source: str,
    *,
    bucket_name: str,
    bucket_index: int,
    entry_name: str,
) -> tuple[str, str, dict[str, set[tuple[str, ...]]], set[str]]:
    filename = f"<aggregate bucket {bucket_name}>"
    tree = ast.parse(source, filename=filename)
    symbols = symtable.symtable(source, filename, "exec")
    imported = {
        symbol.get_name() for symbol in symbols.get_symbols() if symbol.is_imported()
    }
    rebound_imports = {
        symbol.get_name()
        for symbol in symbols.get_symbols()
        if symbol.is_imported() and symbol.is_assigned()
    }
    if rebound_imports:
        raise ValueError(
            f"bucket {bucket_name} rebinds imported names: {sorted(rebound_imports)}"
        )
    globals_to_prefix = {
        symbol.get_name()
        for symbol in symbols.get_symbols()
        if symbol.get_name() not in imported
        and (symbol.is_assigned() or symbol.is_namespace())
    }
    if entry_name not in globals_to_prefix:
        raise ValueError(f"bucket {bucket_name} kernel has no top-level {entry_name}")
    reserved = sorted(
        name
        for name in globals_to_prefix | imported
        if name.startswith("_atrex_bucket_")
    )
    if reserved:
        raise ValueError(
            f"bucket {bucket_name} uses reserved aggregate names: {reserved}"
        )

    prefix = f"_atrex_bucket_{bucket_index}_"
    mapping = {name: prefix + name for name in sorted(globals_to_prefix)}
    collector = _ModuleImportCollector()
    collector.visit(tree)
    tree.body = [
        node
        for node in tree.body
        if not (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and node.module == "__future__"
        )
    ]
    renamed = _GlobalRenamer(source, filename, mapping).visit(tree)
    ast.fix_missing_locations(renamed)
    rendered = ast.unparse(renamed).strip()
    block = (
        f"# BEGIN embedded bucket: {bucket_name}\n"
        f"{rendered}\n"
        f"# END embedded bucket: {bucket_name}"
    )
    return block, mapping[entry_name], collector.bindings, collector.future_features


def embed_bucket_sources(
    module_sources: dict[str, str], *, entry_name: str
) -> EmbeddedBuckets:
    """Namespace bucket modules into static blocks suitable for one kernel.py."""
    blocks: list[str] = []
    entries: list[str] = []
    future_features = {"annotations"}
    import_bindings: dict[str, tuple[str, ...]] = {}

    for index, name in enumerate(sorted(module_sources)):
        block, entry, bindings, features = _embed_one(
            module_sources[name],
            bucket_name=name,
            bucket_index=index,
            entry_name=entry_name,
        )
        for binding, targets in bindings.items():
            if len(targets) != 1:
                raise ValueError(
                    f"bucket {name} imports {binding!r} from multiple targets: "
                    f"{sorted(targets)}"
                )
            target = next(iter(targets))
            previous = import_bindings.get(binding)
            if previous is not None and previous != target:
                raise ValueError(
                    f"embedded bucket import collision for {binding!r}: "
                    f"{previous} vs {target}"
                )
            import_bindings[binding] = target
        future_features.update(features)
        blocks.append(block)
        entries.append(entry)

    return EmbeddedBuckets(
        blocks=tuple(blocks),
        entry_symbols=tuple(entries),
        future_features=tuple(sorted(future_features)),
    )
