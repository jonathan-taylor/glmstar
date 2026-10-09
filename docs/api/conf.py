# Sphinx configuration for the GLMStar API reference.
#
# The narrative docs are a Jupyter Book 2 (MyST) site, which has no autodoc;
# this small Sphinx project builds the API reference from docstrings into
# docs/_build/html/api, next to the book. See .github/workflows/build_docs.yml.
#
# Build:  sphinx-build -b html docs/api docs/_build/html/api

project = 'GLMStar API Reference'
author = 'GLMStar Team'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
]

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}

html_theme = 'sphinx_book_theme'
html_title = 'GLMStar API Reference'
html_theme_options = {
    'repository_url': 'https://github.com/jonathan-taylor/glmstar',
    'use_repository_button': True,
    'home_page_in_toc': True,
}


def _skip_sklearn_metadata_routing(app, what, name, obj, skip, options):
    # scikit-learn adds set_{fit,predict,score,...}_request methods to every
    # estimator subclass; they are not part of the GLMStar API
    if name.startswith('set_') and name.endswith('_request'):
        return True
    return None


def setup(app):
    app.connect('autodoc-skip-member', _skip_sklearn_metadata_routing)
