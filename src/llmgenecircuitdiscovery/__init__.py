import anndata as ad

from llmgenecircuitdiscovery.cli import main

ad.settings.allow_write_nullable_strings = True

__all__ = ["main"]
