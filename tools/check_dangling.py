"""Static check for names that will not resolve at run time.

    python tools/check_dangling.py [path ...]     # default: src/train.py

WHY THIS EXISTS
---------------
E10 crashed 15 minutes into a Kaggle run on

    SAMPLING_WEIGHTS = weights_for_ratio(SAMPLER_RATIO)   # line 1585
    NameError: name 'weights_for_ratio' is not defined

Neither of the checks in use caught it:

  * `py_compile` only checks SYNTAX. A call to a name that does not exist
    is perfectly valid syntax.
  * the `ast.dump` diff compares function BODIES. It reported
    `weights_for_ratio` as IDENTICAL - correctly, because the function was
    never touched. The call site had been moved above the definition.

That second point is the reason this file checks two things, not one. A
checker that only asks "does this name resolve to a module-level def?"
would have passed line 1585, because `weights_for_ratio` IS defined at
module level - eleven lines further down. `src/train.py` is a script that
executes top to bottom, so for a call in module scope, "defined later" and
"not defined" fail identically.

WHAT IS REPORTED
----------------
  UNRESOLVED   a called name that is not a module binding, not a local in
               any enclosing function, not an import and not a builtin.
  TOO EARLY    a call in MODULE scope whose target is bound later in the
               file. This is the E10 bug.
  LOAD ORDER   (advisory) a module-scope read of a module-level name that
               is only bound further down.

Locals are over-approximated inside functions: any name stored anywhere in
a function body counts as bound there. That biases toward silence inside
functions and keeps the module-scope result trustworthy, which is where
this class of bug actually bites.

Exit code 1 if anything in the first two categories is found.
"""

import ast
import builtins
import io
import sys

BUILTINS = set(dir(builtins))

# Bound by the interpreter, not by any statement in the file.
IMPLICIT = {"__name__", "__file__", "__doc__", "__builtins__", "__spec__",
            "__package__", "__loader__"}


def _stored_names(node):
    """Every name a statement subtree binds."""

    out = []

    for sub in ast.walk(node):

        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
            out.append((sub.id, sub.lineno))

        elif isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef,
                              ast.ClassDef)):
            out.append((sub.name, sub.lineno))

        elif isinstance(sub, (ast.Import, ast.ImportFrom)):
            for alias in sub.names:
                name = alias.asname or alias.name.split(".")[0]
                out.append((name, sub.lineno))

        elif isinstance(sub, ast.ExceptHandler) and sub.name:
            out.append((sub.name, sub.lineno))

        elif isinstance(sub, ast.arg):
            out.append((sub.arg, sub.lineno))

        elif isinstance(sub, ast.Global):
            for name in sub.names:
                out.append((name, sub.lineno))

    return out


def module_bindings(tree):
    """Module-scope names -> earliest line that binds them.

    Descends into if/for/while/try/with at module level, but never into a
    function or class body: those bind in their own scope.
    """

    first = {}

    def bind(name, lineno):
        if name not in first or lineno < first[name]:
            first[name] = lineno

    def walk(body):
        for stmt in body:

            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
                bind(stmt.name, stmt.lineno)
                continue

            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                for alias in stmt.names:
                    bind(alias.asname or alias.name.split(".")[0],
                         stmt.lineno)
                continue

            for name, lineno in _stored_names(stmt):
                bind(name, lineno)

            for field in ("body", "orelse", "finalbody"):
                inner = getattr(stmt, field, None)
                if isinstance(inner, list):
                    walk([s for s in inner if isinstance(s, ast.stmt)])

            for handler in getattr(stmt, "handlers", []) or []:
                walk(handler.body)

    walk(tree.body)
    return first


def function_locals(fn):
    """Over-approximate the names bound inside one function."""

    names = {a.arg for a in fn.args.args}
    names |= {a.arg for a in fn.args.kwonlyargs}
    names |= {a.arg for a in fn.args.posonlyargs}
    if fn.args.vararg:
        names.add(fn.args.vararg.arg)
    if fn.args.kwarg:
        names.add(fn.args.kwarg.arg)

    for stmt in fn.body:
        for name, _lineno in _stored_names(stmt):
            names.add(name)

    return names


def parent_map(tree):
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def enclosing_functions(node, parents):
    """Function scopes containing `node`, innermost first."""

    out = []
    cur = parents.get(node)
    while cur is not None:
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef,
                            ast.Lambda)):
            out.append(cur)
        cur = parents.get(cur)
    return out


def comprehension_vars(node, parents):
    """Names bound by any comprehension containing `node`."""

    out = set()
    cur = parents.get(node)
    while cur is not None:
        if isinstance(cur, (ast.ListComp, ast.SetComp, ast.DictComp,
                            ast.GeneratorExp)):
            for gen in cur.generators:
                for sub in ast.walk(gen.target):
                    if isinstance(sub, ast.Name):
                        out.add(sub.id)
        cur = parents.get(cur)
    return out


def check(path):
    src = io.open(path, encoding="utf-8").read()
    tree = ast.parse(src, filename=path)

    mod = module_bindings(tree)
    parents = parent_map(tree)

    scope_locals = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scope_locals[node] = function_locals(node)
        elif isinstance(node, ast.Lambda):
            args = node.args
            names = {a.arg for a in args.args + args.kwonlyargs
                     + args.posonlyargs}
            if args.vararg:
                names.add(args.vararg.arg)
            if args.kwarg:
                names.add(args.kwarg.arg)
            scope_locals[node] = names

    unresolved, too_early, load_order = [], [], []

    for node in ast.walk(tree):

        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)):
            continue

        name = node.func.id
        chain = enclosing_functions(node, parents)

        if name in BUILTINS or name in IMPLICIT:
            continue
        if any(name in scope_locals.get(fn, ()) for fn in chain):
            continue
        if name in comprehension_vars(node, parents):
            continue

        if name not in mod:
            unresolved.append((name, node.lineno))
        elif not chain and mod[name] > node.lineno:
            # Module scope executes top to bottom, so a later binding is
            # exactly as useful as no binding.
            too_early.append((name, node.lineno, mod[name]))

    # Advisory: module-scope reads of module names bound further down.
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)):
            continue
        if enclosing_functions(node, parents):
            continue
        if node.id in BUILTINS or node.id in IMPLICIT:
            continue
        if node.id in comprehension_vars(node, parents):
            continue
        if node.id in mod and mod[node.id] > node.lineno:
            parent = parents.get(node)
            if isinstance(parent, ast.Call) and parent.func is node:
                continue          # already reported above
            load_order.append((node.id, node.lineno, mod[node.id]))

    return unresolved, too_early, load_order


def main(argv):
    paths = argv[1:] or ["src/train.py"]
    bad = 0

    for path in paths:
        unresolved, too_early, load_order = check(path)

        print("=" * 72)
        print("DANGLING REFERENCE AUDIT: %s" % path)
        print("=" * 72)

        print("\nUNRESOLVED calls (no module binding, no local, no builtin): "
              "%d" % len(unresolved))
        for name, lineno in sorted(unresolved, key=lambda x: x[1]):
            print("  line %-6d %s(...)" % (lineno, name))

        print("\nTOO EARLY - module-scope call above its definition: %d"
              % len(too_early))
        for name, lineno, defline in sorted(too_early, key=lambda x: x[1]):
            print("  line %-6d %s(...) is defined at line %d"
                  % (lineno, name, defline))

        print("\nLOAD ORDER (advisory) - module-scope read before binding: %d"
              % len(load_order))
        seen = set()
        for name, lineno, defline in sorted(load_order, key=lambda x: x[1]):
            if (name, defline) in seen:
                continue
            seen.add((name, defline))
            print("  line %-6d %s is bound at line %d"
                  % (lineno, name, defline))

        n = len(unresolved) + len(too_early)
        bad += n
        print("\n%s: %s" % (path, "CLEAN" if n == 0
                            else "%d BLOCKING PROBLEM(S)" % n))

    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
