"""Bind the three existing verified Both sources; no training and no test access."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--legacy-root',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();root=Path(a.legacy_root).resolve();output=Path(a.output).resolve()
    output.mkdir(parents=True,exist_ok=False);descriptors=[]
    cwd=Path(__file__).resolve().parents[1]
    for seed in range(3):
        source=root/'spring_reference'/f'seed{seed}'/'sources/Both/COMPLETE.json'
        head=root/'spring_reference'/f'seed{seed}'/'heads/Both/head0'
        path=output/f'source{seed}.json';descriptors.append(str(path))
        command=[sys.executable,'-m','sprii_next','bind-spring','--native-root',str(root/'native'),
            '--completion',str(source),'--manifest',str(root/'bank/MANIFEST.private.json'),
            '--features',str(head/'features/FEATURES.json'),'--targets',str(head/'targets/SUPERVISION.json'),
            '--method','Both','--seed',str(seed),'--output',str(path)]
        subprocess.run(command,cwd=cwd,check=True)
    subprocess.run([sys.executable,'-m','sprii_next','assemble','--sources',*descriptors,'--output',str(output/'spring.json')],cwd=cwd,check=True)


if __name__=='__main__':main()
