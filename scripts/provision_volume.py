#!/usr/bin/env python3
"""Create the study's network volume in a datacenter that has GPU capacity.

US-KS-2, where the pre-existing tara_gcg volume lives, has no GPU capacity for
any candidate type. A network volume only attaches to pods in its own
datacenter, so that volume cannot serve a GPU study at all. This provisions a
dedicated volume where the GPUs actually are.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pgeom.ops import RunPodClient  # noqa: E402

NAME = "pgeom-clean"
SIZE_GB = 30          # ~16 GB model weights plus checkpoints and artifacts
# Only three storage-capable datacenters had GPU capacity when probed:
# EU-RO-1 (RTX PRO 4500 Blackwell 32GB, RTX 4090), US-CA-2 and US-IL-1 (RTX 4090
# only). EU-RO-1 wins on VRAM headroom for the GCG backward pass at the same
# price. US-KS-2, where the pre-existing tara_gcg volume lives, had none.
DATACENTER = "EU-RO-1"


def main() -> int:
    client = RunPodClient()
    existing = [v for v in client.list_network_volumes() if v.get("name") == NAME]
    if existing:
        print(json.dumps({"already_exists": existing[0]}, indent=2, sort_keys=True))
        return 0
    vol = client._call("POST", "/networkvolumes", {
        "name": NAME, "size": SIZE_GB, "dataCenterId": DATACENTER,
    })
    print(json.dumps(vol, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
