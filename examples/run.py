import rpython

with open("example.ry") as f:
    code = f.read()

result = rpython.compile_to_native(code)
print(f"\nProgram returned: {result}")