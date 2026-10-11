import os, ast, sys

print("=== DEEP ANALYSIS OF utils/ FILES ===\n")

utils_files = [
    "utils/__init__.py",
    "utils/gradcam.py",
    "utils/segmentation.py",
    "utils/dosage_calculator.py",
    "utils/pmfby_verifier.py",
    "utils/pdf_generator.py",
]

for fn in utils_files:
    print(f"--- {fn} ---")
    try:
        with open(fn, encoding="utf-8") as fh:
            src = fh.read()
    except Exception as e:
        print(f"  CANNOT READ: {e}")
        continue

    lines = src.splitlines()

    # 1. Syntax check
    try:
        tree = ast.parse(src)
        print(f"  Syntax: OK")
    except SyntaxError as e:
        print(f"  SYNTAX ERROR: line {e.lineno}: {e.msg} -> {e.text}")
        print()
        continue

    # 2. Collect all names imported
    imports = set()
    from_imports = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imports.add(alias.asname or alias.name)

    # 3. Check for type annotation issues (Python 3.9 vs 3.10)
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        # Check for X | Y type union in function annotations (requires Python 3.10)
        if ") -> " in line or ":" in line:
            if " | " in stripped and "import" not in stripped and "#" not in stripped.split("|")[0]:
                print(f"  WARN line {i}: Union type using '|' requires Python 3.10+ -> {stripped}")

    # 4. Check for usage of undefined names (basic heuristic)
    used_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            used_names.add(node.id)
        elif isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name):
                used_names.add(node.value.id)

    builtins = set(dir(__builtins__)) if isinstance(__builtins__, dict) else set(dir(__builtins__))
    builtins.update(["__name__", "__file__", "__doc__", "TYPE_CHECKING"])

    # Check key modules
    key_modules = ["os", "cv2", "np", "torch", "nn", "F", "models", "transforms",
                   "logger", "math", "io", "datetime", "colors", "inch"]
    for mod in key_modules:
        if mod in used_names and mod not in imports and mod not in builtins:
            print(f"  ERROR: '{mod}' is USED but NOT IMPORTED")

    print()

print("=== Python version ===")
print(f"  {sys.version}")
