import sys
import timeit
from functools import partial, reduce
from itertools import product
from typing import Optional, Union

import torch
from torch._inductor.compile_fx import compile_fx_inner, cudagraphify_impl, compile_fx
from torch._inductor.decomposition import decompositions
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils.benchmark import Compare, Language, Timer, timer as torch_default_timer

device = "cuda"

torch._logging.set_logs(output_code=True)

# SZ = [2**i for i in range(13, 20)]
IN_SZ = [
        100, 
        300
        ]


# RATIOS = [0.25, 0.5, 0.75]
RATIOS = [0.1, 
          0.15, 
          0.2
          ]

def benchmark(name, label, f, x, out_size, would_fallback):
    """Update signature and sub label as needed"""
    sub_label = f"{tuple(x.shape)}, adaptive_max_pool2d({out_size})"
    if would_fallback:
        sub_label = sub_label + "*"

    return Timer(
        "f([x, out_size])",
        globals=locals(),
        label=name,
        description=label,
        sub_label=sub_label,
        num_threads=torch.get_num_threads(),
    ).blocked_autorange(min_run_time=2)


def gen_inputs():
    """Modify this to generate the correct args for function"""
    make_arg = partial(torch.randn, dtype=torch.float32, device=device)
    for n, c in [(512, 3), (1024, 3)]:
        for h_in, w_in in product(IN_SZ, IN_SZ):
            for scale in RATIOS:
                h_out, w_out = int(scale * h_in), int(scale * w_in)
                x = make_arg(n, c, h_in, w_in)
                h_kmax = -((h_in + h_out - 1) // -h_out)
                w_kmax = -((w_in + w_out - 1) // -w_out)
                would_fallback = (h_kmax * w_kmax) > 25
                yield x, (h_out, w_out), would_fallback



def gen_compare(name, x, out_size, would_fallback):
    """Fix signature as needed"""

    def f(args):
        """Unpack args as needed, update val=line to call correct function"""
        x, out_size = args
        val = torch.ops.aten.adaptive_avg_pool2d(x, out_size)
        return val

    sys.stderr.write(f"{x.shape} avgpool2d({out_size})\n")
    args = [x, out_size]

    decomposed = make_fx(f, decomposition_table=decompositions, tracing_mode="fake")(
        args
    )
    compiled_decomposed = compile_fx(decomposed, args, inner_compile=partial(compile_fx_inner, cudagraphs=False))
    yield benchmark(name, f"Compile", compiled_decomposed, *args, would_fallback)

    # Just show the first two generated kernels
    torch._logging.set_logs(output_code=False)
    if device == "cuda":
        eager_f = cudagraphify_impl(f, args, static_input_idxs=tuple(range(len(args))))
    else:
        eager_f = f
    yield benchmark(name, f"Eager", eager_f, *args, would_fallback)


for d in ('cuda','cpu'):
    device = d
    results = []
    name = f"adaptive_avg_pool2d-{device}"
    for args in gen_inputs():
        for res in gen_compare(name, *args):
            results.append(res)

    compare = Compare(results)
    compare.trim_significant_figures()
    compare.print()

