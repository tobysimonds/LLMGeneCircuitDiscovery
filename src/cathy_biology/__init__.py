import anndata as ad

from cathy_biology.cli import main

ad.settings.allow_write_nullable_strings = True

__all__ = ["main"]
