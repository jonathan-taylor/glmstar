"""
Experiments that are too slow to run when the docs are built.

A page defines a function (by default ``calibration``) that runs the
experiment and returns a DataFrame. Run it locally with

    python docs/experiments.py probit_regression [--arg n_rep=1000]

or rebuild the results of every page that defines it with

    python docs/experiments.py --all

Pages run in parallel, one process each (``--jobs`` sets how many at once).

This executes the page's code cells in order until the function is
defined, calls it and saves the result as ``docs/results/<page>.csv``.
The CSV starts with ``# key: value`` lines recording the commit, the
package versions and the arguments; the page reads it back with
`load_results`.

The commit recorded is HEAD, so commit the code first, then the results.
The script refuses to run with uncommitted changes to tracked files, or
with glmstar imported from somewhere other than this checkout (install
it with ``pip install -e .``).
"""

import argparse
import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from html import escape
from importlib.metadata import version
import os
from pathlib import Path
import subprocess
import sys

from IPython.core.inputtransformer2 import TransformerManager
from IPython.display import display, HTML, Markdown
import jupytext
import pandas as pd

import glmnet

DOCS = Path(__file__).resolve().parent
RESULTS = DOCS / 'results'
# the commit identifies glmstar (the version of an editable install is the
# one at install time)
PACKAGES = ['lassoinf', 'coxdev', 'numpy', 'scipy', 'statsmodels', 'pandas']
# as project.github in myst.yml
GITHUB = 'https://github.com/jonathan-taylor/glmstar'


def results_path(page, function='calibration'):
    name = page if function == 'calibration' else f'{page}-{function}'
    return RESULTS / f'{name}.csv'


def load_results(page, function='calibration'):
    """
    The saved results of ``function`` on ``page``, with the header in
    ``.attrs``.
    """
    path = results_path(page, function)
    metadata = {}
    with open(path) as f:
        for line in f:
            if not line.startswith('#'):
                break
            key, _, value = line[1:].partition(':')
            metadata[key.strip()] = value.strip()
    # keep_default_na=False: strings such as 'null' (a kind of signal) or ''
    # are read back as written, not as missing values
    results = pd.read_csv(path, skiprows=len(metadata), keep_default_na=False)
    results.attrs.update(metadata)
    return results


def provenance(results):
    """
    A table of where saved results came from, with the commit linked on
    GitHub.
    """
    m = results.attrs
    commit = m['commit']
    rows = {'computed by': f'<code>{escape(m["function"])}({escape(m["arguments"])})</code>'
                           f' in <code>{escape(m["page"])}</code>',
            'commit': f'<a href="{GITHUB}/commit/{commit}"><code>{commit}</code></a>',
            'date': escape(m['date'])}
    rows.update({p: escape(m[p]) for p in PACKAGES if p in m})
    body = ''.join(f'<tr><th style="text-align: left">{k}</th><td>{v}</td></tr>'
                   for k, v in rows.items())
    return HTML(f'<table>{body}</table>')


def reproduce(results):
    """
    The commands that check out the commit that computed ``results``, install
    it and rerun the experiment, as a bash code block.
    """
    m = results.attrs
    page = Path(m['page']).with_suffix('').as_posix().removeprefix('docs/')
    command = f'python docs/experiments.py {page}'
    if m['function'] != 'calibration':
        command += f' --function {m["function"]}'
    for argument in filter(None, (a.strip() for a in m['arguments'].split(','))):
        command += f' --arg {argument}'
    lines = [f'git clone {GITHUB}.git glmstar',
             'cd glmstar',
             # the commit may not be on a branch any more (e.g. after a squash merge):
             # GitHub still serves it by its hash
             f'git fetch origin {m["commit"]}',
             f'git checkout {m["commit"]}',
             'git submodule update --init --recursive',
             'pip install -e ".[doc]"',
             command]
    return Markdown('To reproduce these results:\n\n```bash\n' + '\n'.join(lines) + '\n```')


def _git(*args):
    return subprocess.run(['git', *args], cwd=DOCS, capture_output=True,
                          text=True, check=True).stdout.rstrip()


def save_results(results, page, function, arguments):
    metadata = {'page': f'docs/{page}.md',
                'function': function,
                'arguments': ', '.join(f'{k}={v!r}' for k, v in arguments.items()),
                'commit': _git('rev-parse', 'HEAD'),
                'date': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}
    metadata.update({p: version(p) for p in PACKAGES})
    path = results_path(page, function)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        for key, value in metadata.items():
            f.write(f'# {key}: {value}\n')
        results.to_csv(f, index=False)
    return path


def run(page, function='calibration', arguments=None):
    """
    Execute the code cells of ``docs/<page>.md`` until ``function`` is
    defined, then call it with ``arguments`` and save the result.
    """
    notebook = jupytext.read(DOCS / f'{page}.md')
    transform = TransformerManager().transform_cell
    namespace = {'__name__': '__main__', 'display': display}
    os.chdir((DOCS / page).parent)      # as when the docs are built
    for cell in notebook.cells:
        if cell.cell_type != 'code' or 'skip-execution' in cell.metadata.get('tags', []):
            continue
        exec(compile(transform(cell.source), f'{page}.md', 'exec'), namespace)
        if function in namespace:
            break
    else:
        raise ValueError(f'docs/{page}.md does not define {function}')
    arguments = arguments or {}
    results = namespace[function](**arguments)
    return save_results(results, page, function, arguments)


def pages_defining(function):
    """The pages in docs/ whose code defines ``function``."""
    return sorted(p.relative_to(DOCS).with_suffix('').as_posix()
                  for p in DOCS.rglob('*.md')
                  if '_build' not in p.parts and f'def {function}(' in p.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[1],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('pages', nargs='*', help='pages in docs/, e.g. probit_regression')
    parser.add_argument('--all', action='store_true',
                        help='every page in docs/ that defines the function')
    parser.add_argument('--function', default='calibration')
    parser.add_argument('--jobs', type=int, default=None,
                        help='pages to run at once (default: all, up to the number of CPUs)')
    parser.add_argument('--arg', action='append', default=[], metavar='NAME=VALUE',
                        help='keyword argument for the function (a Python literal)')
    parser.add_argument('--force', action='store_true',
                        help='run anyway (the commit recorded may not be the '
                             'code that ran)')
    args = parser.parse_args()
    if args.all:
        args.pages = pages_defining(args.function)
    if not args.pages:
        parser.error('give some pages, or --all')

    arguments = {}
    for a in args.arg:
        name, _, value = a.partition('=')
        arguments[name] = ast.literal_eval(value)

    if not args.force:
        changed = _git('status', '--porcelain', '--untracked-files=no',
                       '--', ':(top)', ':(top,exclude)docs/results')
        if changed:
            sys.exit(f'uncommitted changes, so the commit recorded would be wrong:\n{changed}\n'
                     'commit them first, or pass --force')
        root = Path(_git('rev-parse', '--show-toplevel')).resolve()
        if root not in Path(glmnet.__file__).resolve().parents:
            sys.exit(f'glmstar is imported from {Path(glmnet.__file__).parent}, not {root}: '
                     'install this checkout with pip install -e ., or pass --force')

    if not _git('branch', '--remotes', '--contains', 'HEAD'):
        print('warning: HEAD is not on a remote branch; push it, or the commit '
              'link in the docs will not resolve', file=sys.stderr)
    pages = [Path(p).with_suffix('').as_posix().removeprefix('docs/') for p in args.pages]
    jobs = args.jobs or min(len(pages), os.cpu_count())
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(run, page, args.function, arguments): page for page in pages}
        for future in as_completed(futures):
            print(f'{futures[future]}: {future.result()}')


if __name__ == '__main__':
    main()
