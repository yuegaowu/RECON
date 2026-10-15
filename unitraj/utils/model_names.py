"""Public architecture names and compatibility with historical cache metadata."""


def canonical_model_key(name):
    return {'wayformerx_ci_v2': 'recon', 'wayformerx': 'recon_lp'}.get(name, name)


def legacy_model_key(name):
    return {'recon': 'wayformerx_ci_v2', 'recon_lp': 'wayformerx'}.get(name, name)
