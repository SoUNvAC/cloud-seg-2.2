import copy

import pytest
import torch

from cloud_adapter.models.backbones.cloud_adapter_dinov2 import (
    CloudAdapterDinoVisionTransformer,
)
from cloud_adapter.models.backbones.cloud_adapter import CloudAdapter
from cloud_adapter.models.backbones.frequency_routed_adapter import (
    FrequencyRoutedInteractiveModule,
    TokenFrequencyRouter,
)


@pytest.mark.parametrize("mode", ["fixed", "image", "token"])
def test_router_weights_sum_to_one(mode):
    router = TokenFrequencyRouter(
        token_dim=32, context_dim=8, router_dim=4, temperature=1.0, mode=mode
    )
    weights = router(torch.randn(2, 24, 32), torch.randn(2, 8, 3, 5))
    assert weights.shape == (2, 24, 3)
    assert (weights.sum(dim=-1) - 1.0).abs().max().item() < 1e-6


def _make_module(device="cpu"):
    return FrequencyRoutedInteractiveModule(
        emd_dim=32,
        context_dim=8,
        ranks=(2, 1, 1),
        cutoff_ratio=0.3,
        router_dim=4,
        router_mode="token",
        alpha_init=0.1,
        heads=4,
        dim_head=8,
    ).to(device)


@pytest.mark.parametrize(
    "batch,height,width", [(2, 32, 32), (2, 16, 16), (1, 24, 32)]
)
def test_rectangular_forward_backward(batch, height, width):
    module = _make_module()
    tokens = torch.randn(height * width, batch, 32, requires_grad=True)
    context = torch.randn(batch, 8, height, width, requires_grad=True)
    output = module(tokens, context, index=0, spatial_shape=(height, width))
    assert output.shape == tokens.shape
    output.square().mean().backward()
    assert tokens.grad is not None and torch.isfinite(tokens.grad).all()
    assert context.grad is not None and torch.isfinite(context.grad).all()


def test_amp_forward_is_finite():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.float16 if device == "cuda" else torch.bfloat16
    module = _make_module(device)
    tokens = torch.randn(24 * 32, 1, 32, device=device)
    context = torch.randn(1, 8, 12, 16, device=device)
    with torch.autocast(device_type=device, dtype=amp_dtype):
        output = module(tokens, context, index=0, spatial_shape=(24, 32))
    assert torch.isfinite(output).all()


def test_adapter_checkpoint_round_trip():
    torch.manual_seed(7)
    module = _make_module().eval()
    tokens = torch.randn(24 * 32, 1, 32)
    context = torch.randn(1, 8, 12, 16)
    with torch.no_grad():
        expected = module(tokens, context, index=0, spatial_shape=(24, 32))
    checkpoint = copy.deepcopy(module.state_dict())
    restored = _make_module().eval()
    restored.load_state_dict(checkpoint)
    with torch.no_grad():
        actual = restored(tokens, context, index=0, spatial_shape=(24, 32))
    assert (actual - expected).abs().max().item() < 1e-5
    assert "alpha" in checkpoint
    assert "decomposer.cutoff_ratio" in checkpoint
    assert "attention.rank_budget" in checkpoint
    assert "router.temperature" in checkpoint
    assert "router.mode_id" in checkpoint


def test_dinov2_is_frozen_while_adapter_gets_gradients():
    model = CloudAdapterDinoVisionTransformer(
        cloud_adapter_config=dict(
            type="CloudAdapter",
            cnn_type="pmaa",
            int_type="frequency_routed",
            emd_dim=32,
            num_layers=4,
            context_dim=8,
            hidden_channels=8,
            depth=4,
            return_multi_feats=False,
            return_last_feature=False,
            frequency_ranks=(2, 1, 1),
            router_dim=4,
            frequency_heads=4,
            frequency_dim_head=8,
        ),
        adapter_index=[0, 1, 2, 3],
        img_size=(64, 96),
        patch_size=16,
        embed_dim=32,
        depth=4,
        num_heads=4,
        out_indices=[0, 1, 2, 3],
        block_chunks=0,
        init_values=1e-5,
    )
    model.train()
    outputs, _ = model.forward_features(torch.randn(1, 3, 64, 96))
    sum(output.mean() for output in outputs).backward()
    frozen = [
        parameter
        for name, parameter in model.named_parameters()
        if "cloud_adapter" not in name
    ]
    adapter = [
        parameter
        for name, parameter in model.named_parameters()
        if "cloud_adapter" in name
    ]
    assert frozen and all(not parameter.requires_grad for parameter in frozen)
    assert all(parameter.grad is None for parameter in frozen)
    assert any(parameter.grad is not None for parameter in adapter)


def test_main_adapter_parameter_budget():
    adapter = CloudAdapter(
        cnn_type="pmaa",
        int_type="frequency_routed",
        emd_dim=1024,
        num_layers=24,
        context_dim=64,
        hidden_channels=64,
        depth=4,
        return_multi_feats=False,
        return_last_feature=False,
        frequency_ranks=(8, 4, 4),
        router_dim=8,
    )
    assert sum(parameter.numel() for parameter in adapter.parameters()) <= 2_100_000
