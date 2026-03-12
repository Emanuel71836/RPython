import sys
import os
import rpython

def execute_file(path: str) -> str:
    code = open(path).read()
    return rpython.compile_to_native(code)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 run.py <file.ry> [--debug]", file=sys.stderr)
        sys.exit(1)

    path = sys.argv[1]
    debug = "--debug" in sys.argv

    if debug:
        code = open(path).read()
        print(f"=== {path} ({len(code)} bytes, {len(code.splitlines())} lines) ===",
              file=sys.stderr)
        for i, line in enumerate(code.splitlines()[:15], 1):
            print(f"  {i:3}: {line!r}", file=sys.stderr)
        print("  ...", file=sys.stderr)

    result = execute_file(path)
    print(result)