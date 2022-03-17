#!/usr/bin/env python3

import json
import os
import sys

EXTKEY1="#!/usr/bin/env python"
EXTKEY2="#extract"

if "NBFILE" in os.environ:
    nbfile = os.environ["NBFILE"]
else:
    print("No NBFILE given in environment")
    sys.exit(1)

if "PYFILE" in os.environ:
    pyfile = os.environ["PYFILE"]
else:
    print("No PYFILE given in environment")
    sys.exit(1)

nb_content=""
py_content=""
with open(nbfile,"r") as nbf:
    nb_content = nbf.read()
    nb_json = json.loads(nb_content)
    for c in nb_json["cells"]:
        if c["cell_type"]=="code":
            if len(c["source"]) == 0:
                continue
            start_line = c["source"][0]
            #print(start_line)
            if ( EXTKEY1 in start_line) or ( EXTKEY2 in start_line):
                #print(c)
                for line in c["source"]:
                    #print(line)
                    py_content += line
                py_content += "\n"
                #Alternateive way:
                #py_content = "".join(c["source"])
    #print(py_content)

with open(pyfile,"w+") as pyf:
    pyf.write(py_content)
