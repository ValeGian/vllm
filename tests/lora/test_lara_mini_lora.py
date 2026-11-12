# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lara Mini LoRA in the PEFT layout of the Transformers implementation.

Adapters name the gated delta net input projections `in_proj_{qkv,z,b,a}` (the
checkpoint fuses them into one `in_proj`) and target the routed experts as 3D
parameters (`experts.base_layer.lora_{A,B}` for `gate_up_proj`,
`experts.lora_{A,B}` for `down_proj`). Serving an adapter must match serving the
base model with the LoRA delta merged into its weights.
"""

import json

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import vllm
from tests.models.registry import HF_EXAMPLE_MODELS
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.lora.request import LoRARequest

from ..utils import multi_gpu_test

RANK = 8
ALPHA = 16


@pytest.fixture(scope="module")
def lara_mini_lora_models(tmp_path_factory) -> tuple[str, str, str]:
    """A random tiny Lara Mini with every token mixer, a LoRA adapter on every
    projection and a copy of the model with that adapter merged."""
    model_info = HF_EXAMPLE_MODELS.get_hf_info("LaraMiniForCausalLM")
    model_info.check_available_online(on_fail="skip")
    model_info.check_transformers_version(on_fail="skip")
    config = AutoConfig.from_pretrained(
        model_info.default,
        revision=model_info.revision,
        trust_remote_code=True,
        hidden_size=256,
        **model_info.hf_overrides,
        num_attention_heads=4,
        num_key_value_heads=2,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        num_experts=8,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=128,
        sliding_window=16,
    )
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(
        config, trust_remote_code=True, code_revision=model_info.revision
    )
    model = model.to(torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(
        model_info.default, revision=model_info.revision, trust_remote_code=True
    )
    root = tmp_path_factory.mktemp("lara_mini_lora")
    base_dir, merged_dir, lora_dir = (root / n for n in ("base", "merged", "lora"))
    for model_dir in (base_dir, merged_dir):
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)

    weights = load_file(merged_dir / "model.safetensors")
    scale = ALPHA / RANK
    gdn_heads = config.linear_num_value_heads
    gdn_q = config.linear_num_key_heads * config.linear_key_head_dim
    gdn_v = gdn_heads * config.linear_value_head_dim
    # The checkpoint fuses the GDN input projections, rows [q | k | v | z | b | a]
    gdn_in_proj = {
        "in_proj_qkv": 2 * gdn_q + gdn_v,
        "in_proj_z": gdn_v,
        "in_proj_b": gdn_heads,
        "in_proj_a": gdn_heads,
    }
    adapter: dict[str, torch.Tensor] = {}

    def lora_pair(name: str, out_dim: int, in_dim: int) -> torch.Tensor:
        lora_a = torch.randn(RANK, in_dim) / in_dim**0.5
        lora_b = torch.randn(out_dim, RANK) * 0.05
        adapter[f"base_model.model.{name}.lora_A.weight"] = lora_a
        adapter[f"base_model.model.{name}.lora_B.weight"] = lora_b
        return scale * lora_b @ lora_a

    for i, layer_type in enumerate(config.layer_types):
        prefix = f"model.layers.{i}"
        if layer_type == "linear_attention":
            in_proj = f"{prefix}.linear_attn.in_proj.weight"
            deltas = [
                lora_pair(f"{prefix}.linear_attn.{name}", rows, config.hidden_size)
                for name, rows in gdn_in_proj.items()
            ]
            weights[in_proj] = weights[in_proj].float() + torch.cat(deltas)
            projs = ["linear_attn.out_proj"]
        else:
            projs = [f"self_attn.{p}_proj" for p in "qkvo"]
        projs += [
            f"mlp.shared_expert.{p}" for p in ("gate_proj", "up_proj", "down_proj")
        ]
        for proj in projs:
            name = f"{prefix}.{proj}"
            out_dim, in_dim = weights[f"{name}.weight"].shape
            weights[f"{name}.weight"] = weights[f"{name}.weight"].float() + lora_pair(
                name, out_dim, in_dim
            )

        # Routed experts: one flat LoRA pair per 3D projection (E, out, in)
        experts = f"{prefix}.mlp.experts"
        for proj, module in (("gate_up_proj", "base_layer."), ("down_proj", "")):
            num_experts, out_dim, in_dim = weights[f"{experts}.{proj}"].shape
            lora_a = torch.randn(num_experts * RANK, in_dim) / in_dim**0.5
            lora_b = torch.randn(out_dim, RANK * num_experts) * 0.05
            adapter[f"base_model.model.{experts}.{module}lora_A.weight"] = lora_a
            adapter[f"base_model.model.{experts}.{module}lora_B.weight"] = lora_b
            a = lora_a.view(num_experts, RANK, in_dim)
            b = lora_b.view(out_dim, RANK, num_experts).permute(2, 0, 1)
            weights[f"{experts}.{proj}"] = weights[
                f"{experts}.{proj}"
            ].float() + scale * torch.bmm(b, a)

    weights = {k: v.to(torch.bfloat16) for k, v in weights.items()}
    save_file(weights, merged_dir / "model.safetensors", metadata={"format": "pt"})
    lora_dir.mkdir()
    save_file(adapter, lora_dir / "adapter_model.safetensors")
    (lora_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "r": RANK,
                "lora_alpha": ALPHA,
                "target_modules": sorted(
                    {k.split(".")[-3] for k in adapter if ".experts." not in k}
                ),
                "target_parameters": [
                    "mlp.experts.gate_up_proj",
                    "mlp.experts.down_proj",
                ],
                "bias": "none",
            }
        )
    )
    return str(base_dir), str(merged_dir), str(lora_dir)


def _prompts() -> list[list[int]]:
    # Longer than the 16-token window of the sliding attention layers
    generator = torch.Generator().manual_seed(1)
    return [
        torch.randint(1000, 10000, (64,), generator=generator).tolist()
        for _ in range(4)
    ]


def _prompt_logprobs(
    model: str, lora_path: str | None = None, **kwargs
) -> dict[str, list[list[float]]]:
    """Prompt logprobs without LoRA; with `lora_path` also with the adapter, and
    in a batch that alternates between the adapter and no adapter."""
    llm = vllm.LLM(
        model,
        trust_remote_code=True,
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=256,
        max_num_seqs=4,
        gpu_memory_utilization=0.3,
        enable_lora=True,
        max_lora_rank=RANK,
        **kwargs,
    )
    try:
        prompts = _prompts()
        params = vllm.SamplingParams(max_tokens=1, prompt_logprobs=0, detokenize=False)
        requests: dict[str, LoRARequest | list[LoRARequest | None] | None] = {
            "base": None
        }
        if lora_path is not None:
            lora = LoRARequest("lara-mini-lora", 1, lora_path)
            requests["lora"] = lora
            requests["mixed"] = [lora, None] * (len(prompts) // 2)
        results = {}
        for name, lora_request in requests.items():
            outputs = llm.generate(
                [{"prompt_token_ids": p} for p in prompts],
                params,
                lora_request=lora_request,
            )
            results[name] = [
                _token_logprobs(out, p) for out, p in zip(outputs, prompts)
            ]
        return results
    finally:
        del llm
        cleanup_dist_env_and_memory()


def _token_logprobs(output: vllm.RequestOutput, prompt: list[int]) -> list[float]:
    assert output.prompt_logprobs is not None
    logprobs = []
    for token, position in zip(prompt[1:], output.prompt_logprobs[1:]):
        assert position is not None
        logprobs.append(position[token].logprob)
    return logprobs


def _mean_abs_diff(x: list[list[float]], y: list[list[float]]) -> float:
    diffs = [abs(a - b) for xs, ys in zip(x, y) for a, b in zip(xs, ys)]
    return sum(diffs) / len(diffs)


def _check(lara_mini_lora_models, **kwargs) -> None:
    base, merged, lora = lara_mini_lora_models
    served = _prompt_logprobs(base, lora, **kwargs)
    merged_lp = _prompt_logprobs(merged, **kwargs)["base"]

    signal = _mean_abs_diff(merged_lp, served["base"])
    assert signal > 0.1
    assert _mean_abs_diff(served["lora"], merged_lp) < 0.1 * signal
    # In a mixed batch, each prompt gets its own adapter (or none)
    expected = [
        served["lora"][i] if i % 2 == 0 else served["base"][i]
        for i in range(len(served["base"]))
    ]
    assert _mean_abs_diff(served["mixed"], expected) < 0.1 * signal


def test_lara_mini_lora_matches_merged(lara_mini_lora_models):
    _check(lara_mini_lora_models)


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("fully_sharded_loras", [False, True])
def test_lara_mini_lora_matches_merged_tp2(lara_mini_lora_models, fully_sharded_loras):
    _check(
        lara_mini_lora_models,
        tensor_parallel_size=2,
        fully_sharded_loras=fully_sharded_loras,
    )
