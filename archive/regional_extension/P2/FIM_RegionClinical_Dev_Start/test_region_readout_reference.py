"""CPU tensor tests only; no medical data, trained weights or performance claims."""
from __future__ import annotations
import json
import torch
from region_readout_reference import region_tokens, RegionReadout, SpatialCrossAttentionReadout


def main():
    torch.set_num_threads(1)
    torch.manual_seed(171)
    tests = []
    f = torch.randn(2, 16, 7, 7, requires_grad=True)
    masks = torch.randn(2, 1, 7, 7, requires_grad=True)
    g = f.mean((-2, -1))
    h = torch.randn(2, 16, requires_grad=True)
    tokens, area, valid = region_tokens(f, masks)
    assert tokens.shape == (2, 3, 16)
    torch.testing.assert_close(area.sum(1), torch.ones(2), atol=2e-6, rtol=2e-6)
    tests.append("soft_partition_shape_and_area_sum")
    for constant in [-100., 0., 100.]:
        t, a, v = region_tokens(f, torch.full((2, 1, 7, 7), constant))
        assert torch.isfinite(t).all() and v.any(1).all()
    tests.append("all_background_uniform_all_foreground_are_finite")
    r = RegionReadout(16, 8, True)
    out, diagnostics = r(f, masks, g, h)
    torch.testing.assert_close(out, g, rtol=0, atol=0)
    tests.append("zero_residual_initialization_is_identity")
    r.eta.data.fill_(0.5)
    out, diagnostics = r(f, masks, g, h)
    out.square().mean().backward()
    assert masks.grad is None
    assert f.grad is not None and f.grad.abs().sum() > 0
    assert h.grad is not None and h.grad.abs().sum() > 0
    tests.append("classification_gradient_reaches_features_and_clinical_gate_not_mask_weights")
    torch.testing.assert_close(diagnostics["region_weights"].sum(1), torch.ones(2))
    tests.append("region_attention_normalized")
    r_img = RegionReadout(16, 8, False)
    r_img.eta.data.fill_(0.5)
    out1, _ = r_img(f.detach(), masks.detach(), g.detach(), h.detach())
    out2, _ = r_img(f.detach(), masks.detach(), g.detach(), h.detach() + 9)
    torch.testing.assert_close(out1, out2, rtol=0, atol=0)
    tests.append("image_only_region_gate_ignores_metadata_input")
    out1, _ = r(f.detach(), masks.detach(), g.detach(), h.detach())
    out2, _ = r(f.detach(), masks.detach(), g.detach(), h.detach().roll(1, dims=1))
    assert (out1 - out2).abs().max() > 1e-7
    tests.append("clinical_region_gate_can_respond_to_metadata")
    x = SpatialCrossAttentionReadout(16, 8, 4)
    out, _ = x(f.detach(), g.detach(), h.detach())
    torch.testing.assert_close(out, g.detach(), rtol=0, atol=0)
    x.eta.data.fill_(0.5)
    out, _ = x(f.detach(), g.detach(), h.detach())
    assert torch.isfinite(out).all()
    tests.append("grid_attention_shape_identity_and_finite_forward")
    try:
        region_tokens(f, torch.zeros(2, 2, 7, 7))
    except ValueError:
        tests.append("invalid_mask_shape_rejected")
    else:
        raise AssertionError("invalid input was not rejected")
    print(json.dumps({"status":"PASS", "test_count":len(tests), "tests":tests,
                      "torch_version":torch.__version__, "device":"cpu",
                      "scope":"standalone synthetic-tensor tests only; not trainer integration or clinical validation"},
                      ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
