# save this as py2ipynb.py
import sys, nbformat
from nbformat.v4 import new_notebook, new_code_cell
src = open(sys.argv[1]).read()
nb = new_notebook(cells=[new_code_cell(src)])
nbformat.write(nb, sys.argv[2])
