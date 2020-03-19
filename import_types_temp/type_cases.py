from typing import Dict, NewType
import pandas as pd

# Type alias
Alias = Dict[(int, int)]

# NewType
NewInt = NewType('NewInt', int)

def random_func():
    print("X")

# Type 1
class test():
    def __init__(self):
        print("X")