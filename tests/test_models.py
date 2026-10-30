import pytest
import torch

from adfusion.models import MODEL_NAMES, HybridFusion, build_model, count_params

# Parameter counts reported in the paper (millions, rounded to 3 decimals).
EXPECTED_NAIVE = {
    "mri": 0.316,
    "pet": 0.316,
    "early": 0.316,
    "late": 0.632,
    "intermediate": 0.632,
    "hybrid": 0.781,
    "hybrid_noattn": 0.649,
}
EXPECTED_MATCHED = {
    "mri": 0.607,
    "pet": 0.607,
    "early": 0.608,
    "late": 0.632,
    "intermediate": 0.632,
    "hybrid": 0.781,
    "hybrid_noattn": 0.649,
}


@pytest.fixture
def batch():
    return torch.rand(2, 1, 16, 16, 16), torch.rand(2, 1, 16, 16, 16)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_forward_returns_one_logit_per_sample(name, batch):
    model = build_model(name).eval()
    out = model(*batch)
    assert out.shape == (2,)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_backward_reaches_every_parameter(name, batch):
    model = build_model(name).train()
    model(*batch).sum().backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, missing


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_parameter_counts_match_paper(name):
    assert round(count_params(build_model(name)) / 1e6, 3) == EXPECTED_NAIVE[name]
    matched = count_params(build_model(name, capacity_matched=True)) / 1e6
    assert round(matched, 3) == EXPECTED_MATCHED[name]


def test_capacity_matching_closes_the_gap():
    single = count_params(build_model("mri", capacity_matched=True))
    double = count_params(build_model("intermediate"))
    assert abs(single - double) / double < 0.05


def test_single_modality_ignores_other_input():
    model = build_model("mri").eval()
    mri = torch.rand(1, 1, 16, 16, 16)
    a = model(mri, torch.zeros(1, 1, 16, 16, 16))
    b = model(mri, torch.ones(1, 1, 16, 16, 16))
    torch.testing.assert_close(a, b)


def test_hybrid_returns_attention_and_branch_weights(batch):
    model = HybridFusion(use_attn=True).eval()
    out, (w1, w2, w) = model(*batch, return_attn=True)
    assert out.shape == (2,)
    assert w1.shape == w2.shape == (2, 1, 1)
    torch.testing.assert_close(w.sum(), torch.tensor(1.0))
    torch.testing.assert_close(model.branch_weights(), torch.full((3,), 1 / 3))


def test_ablation_has_no_attention_modules(batch):
    model = HybridFusion(use_attn=False).eval()
    _, (w1, w2, _) = model(*batch, return_attn=True)
    assert w1 is None and w2 is None
    assert not hasattr(model, "x_m2p")


def test_unknown_model_name():
    with pytest.raises(ValueError, match="unknown model"):
        build_model("transformer")
