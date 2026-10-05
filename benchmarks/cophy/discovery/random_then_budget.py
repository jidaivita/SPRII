"""Prioritize the requested Random references, then finish the budget exploration."""
import subprocess
import sys
from pathlib import Path

root=Path(__file__).parent
for script,args in [('random_reference.py',['dispatch']),('extend_advantage100.py',[])]:
    subprocess.run([sys.executable,'-u',str(root/script),*args],check=True)
