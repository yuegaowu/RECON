"""Definitions and guards for the three approved RECON group ablations."""
import inspect
from pathlib import Path

ARMS = ('NoScoreBranch', 'NoAuxGeo', 'NoProto')


def validate_ablation_structure(model, cfg):
    import torch
    from tools.recon_training import U, sha
    arm = cfg.recon_arm
    assert arm in ARMS and type(model).__name__ == 'RECON'
    assert Path(inspect.getfile(type(model))).resolve() == U / 'models/recon/recon.py'
    expected = dict(causal_v1_enabled=False, causal_v2_enabled=arm != 'NoScoreBranch',
                    clean_norm_v2=False, score_head_only=False, rank_loss_weight=0., cls_loss_weight=1.,
                    best6_ade_weight=0. if arm == 'NoAuxGeo' else .2,
                    best6_fde_weight=0. if arm == 'NoAuxGeo' else .1,
                    graph_sparsity_weight=0. if arm == 'NoScoreBranch' else .001,
                    prototype_entropy_weight=.0005 if arm == 'NoAuxGeo' else 0.,
                    confounder_subtract_scale=.5 if arm == 'NoAuxGeo' else 0.,
                    causal_logit_residual_weight=1., kinematic_filter=False,
                    num_modes=6, hidden_size=256, past_len=11, future_len=80,
                    num_queries_enc=192, num_queries_dec=64,
                    num_encoder_layers=2, num_decoder_layers=8)
    for key, value in expected.items():
        assert cfg[key] == value, (arm, key, cfg[key], value)
    assert not model.causal_v1 and model.causal_v2 == (arm != 'NoScoreBranch')
    assert not model.clean_norm_v2 and not hasattr(model, 'deconf_prob_v2')
    assert all(p.requires_grad for p in model.parameters())
    removed = ('confounder_prototypes', 'confounder_q', 'confounder_gate',
               'prototype_value_proj', 'prototype_value_norm', 'prototype_residual_norm')
    if arm == 'NoScoreBranch':
        removed += ('causal_graph', 'graph_q', 'graph_k', 'graph_v', 'graph_out',
                    'graph_type_embedding', 'risk_token_encoder', 'mode_intervention',
                    'deconfounded_prob_predictor')
    if arm in ('NoProto', 'NoScoreBranch'):
        assert all(not hasattr(model, name) for name in removed)
    if arm != 'NoScoreBranch':
        assert model.mode_intervention[0].in_features == 1030
        assert model.deconfounded_prob_predictor[0].in_features == 1286
    dictionary = None
    if arm == 'NoAuxGeo':
        proto = model.confounder_prototypes
        assert proto.shape == (64, 256) and not proto.requires_grad
        assert torch.allclose(proto.norm(dim=-1), torch.ones(64, device=proto.device), atol=1e-5)
        path = model._resolve_unitraj_path(cfg.confounder_init_path).resolve()
        assert path == (U / 'models/wayformer/prototypes_large_gpu/normal_mode_prototypes_k64.pt').resolve()
        dictionary = {'path': str(path), 'sha256': sha(path)}
    return {'arm': arm, 'class': type(model).__name__, 'checks': expected,
            'trainable_parameters': sum(p.numel() for p in model.parameters()),
            'dictionary': dictionary,
            'removed_attributes': list(removed) if arm != 'NoAuxGeo' else []}
