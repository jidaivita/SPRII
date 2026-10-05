"""Logged continuation repair; original objectives, states, RNG and binding retained.
Only remove a duplicated event-log keyword. The original binding remains the
scientific base; this wrapper is separately hash-bound by task and repair receipt.
"""

import os
import hashlib,json,sys
from pathlib import Path
ORIGINAL=Path((os.environ.get("SPRII_COPHY_ROOT", "runs/cophy") + '/source/cophy_complete_v7/legacy/supervised_continue.py'))
EXPECTED='66e6322cd8d393d4289939b6e7bc50943bfeb8a4e1087c3e9b608162ed20996d'
raw=ORIGINAL.read_bytes()
if hashlib.sha256(raw).hexdigest()!=EXPECTED:raise ValueError('Unexpected original continuation source')
old="io.emit('SOURCE_EPOCH',scene=a.scene,method=a.method,**row)"
new="io.emit('SOURCE_EPOCH',**dict(row,scene=a.scene,method=a.method))"
text=raw.decode()
if text.count(old)!=1:raise ValueError('Unexpected log call layout')
text=text.replace(old,new)
# Keep the original scientific binding so epoch51 checkpoints resume exactly.
# Actual executed wrapper and the one-statement change are separately recorded.
sys.path.insert(0,str(ORIGINAL.parent))
exec(compile(text,str(ORIGINAL),'exec'),{'__name__':'__main__','__file__':str(ORIGINAL)})
