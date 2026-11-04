#!/usr/bin/env python3
"""Point generated recipe copies at a local source tree, for validation only."""
import argparse
import json
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipes', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    for file in sorted(args.recipes.glob('*/meta.yaml')):
        text = file.read_text()
        text, count = re.subn(r'\nsource:\n.*?(?=\nbuild:)',
                             lambda _: '\nsource:\n  path: ' + json.dumps(str(args.source.resolve())) + '\n',
                             text, count=1, flags=re.S)
        if count != 1:
            parser.error('Cannot locate source block: ' + str(file))
        file.write_text(text)
        print(file)


if __name__ == '__main__':
    main()
