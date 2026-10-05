"""Check that lambda_align=0 preserves the released NOD-Hier update."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch import nn

from fhn_minimal.train import create_locs, model_checksum, predict_four, set_seed, training_step


def reference_step(model, optimizer, batch, criterion, locs) -> float:
    x, c1, _, target, indicators, _ = batch
    device = locs.device
    prediction, _ = predict_four(
        model, x.to(device), c1.to(device), indicators.to(device), locs
    )
    loss = criterion(prediction, target.to(device))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-code", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    sys.path.insert(0, str(args.official_code))
    from ngs.neuralnetworks import NGS_metaNet_Hier

    device = torch.device(args.device)
    set_seed(42)
    reference = NGS_metaNet_Hier(64, 3, 2, 128).to(device)
    fork = NGS_metaNet_Hier(64, 3, 2, 128).to(device)
    fork.load_state_dict(reference.state_dict())
    count = sum(p.numel() for p in reference.parameters() if p.requires_grad)
    lr = 0.5 / math.sqrt(count)
    opt_reference = torch.optim.RMSprop(reference.parameters(), lr=lr)
    opt_fork = torch.optim.RMSprop(fork.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(1234)
    batch_size = 2
    batch = (
        torch.randn(batch_size, 2, 128, 128, generator=generator),
        torch.randn(batch_size, 5, 2, 128, 128, generator=generator),
        torch.randn(batch_size, 5, 2, 128, 128, generator=generator),
        torch.randn(batch_size, 4, 2, 128, 128, generator=generator),
        torch.randint(0, 4, (batch_size, 4), generator=generator),
        torch.zeros(batch_size, 2),
    )
    criterion = nn.MSELoss()
    locs = create_locs(device)
    reference_loss = reference_step(reference, opt_reference, batch, criterion, locs)
    fork_metrics = training_step(fork, opt_fork, batch, criterion, locs, align_weight=0.0)
    max_parameter_difference = max(
        float((a - b).abs().max())
        for a, b in zip(reference.parameters(), fork.parameters())
    )
    receipt = {
        "status": "PASS" if max_parameter_difference == 0.0 and reference_loss == fork_metrics["base_loss"] else "FAIL",
        "reference_base_loss": reference_loss,
        "fork_base_loss": fork_metrics["base_loss"],
        "fork_align_loss": fork_metrics["align_loss"],
        "max_parameter_difference": max_parameter_difference,
        "reference_checksum": model_checksum(reference),
        "fork_checksum": model_checksum(fork),
        "device": str(device),
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if receipt["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
