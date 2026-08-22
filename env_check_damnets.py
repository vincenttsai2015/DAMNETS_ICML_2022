import importlib
import os
import sys

MODS = ["torch", "torch_geometric", "networkx", "networkx_temporal", "pandas",
        "numpy", "scipy", "sklearn", "matplotlib", "tqdm", "yaml", "easydict",
        "tensorboard", "pyemd", "pygsp", "gensim", "igraph", "msgpack", "powerlaw"]

bad = []
for m in MODS:
    try:
        mod = importlib.import_module(m)
        print("  OK   {:20s} {}".format(m, getattr(mod, "__version__", "")))
    except Exception as e:
        bad.append(m)
        print("  FAIL {:20s} {}: {}".format(m, type(e).__name__, e))

import torch  # noqa: E402

print()
print("torch path :", os.path.dirname(torch.__file__))
print("cuda avail :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device     :", torch.cuda.get_device_name(0))
print()
print("缺少:", bad if bad else "無")
sys.exit(1 if bad else 0)
