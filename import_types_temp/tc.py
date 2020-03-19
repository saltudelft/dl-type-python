import type_cases
import inspect
import sys

mems = inspect.getmembers(sys.modules["type_cases"])

for m in mems:
    if (inspect.isclass(m[1])):
        print(dir(m[1]), m[1].__class__)