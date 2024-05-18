import torch
from torch.testing import make_tensor
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils.benchmark import Timer, Compare
from torch._inductor.compile_fx import compile_fx_inner, cudagraphify_impl
from torch._inductor.decomposition import decompositions
from itertools import product
from functools import partial

aten = torch.ops.aten

torch._logging.set_logs(output_code=True)

benchmark_name = "avgpool_3d_backwards"
Ss = [
    # batchsize, numchannels, depth, height, width
    [3, 5, 400, 200, 200],
    [3, 5, 300, 200, 200],
    [3, 5, 200, 200, 200],
    [3, 5, 300, 300, 300],
    [3, 5, 100, 100, 100],
    [3, 5, 100, 300, 200],
    [8, 8, 128, 128, 128],
    [2, 3, 150, 150, 150],
    [1, 3, 128, 128, 128],
    [8, 16, 64, 64, 64],
    [1, 1, 50, 50, 50],
    [3, 5, 20, 40, 40],
    [3, 5, 10, 20, 20],
    [1, 1, 10, 10, 10],
    [3, 5, 5, 10, 10],
    [3, 5, 2, 5, 5],
]


def gen_inputs():
    for shape in Ss:
        x_shape = shape
        y_shape = [shape[0], shape[1], shape[2] // 2, shape[3] // 2, shape[4] // 2]

        x = torch.randn(x_shape, dtype=torch.float32, device='cuda')
        grad_output = torch.randn(y_shape, dtype=torch.float32, device='cuda')

        yield [x, grad_output]


def benchmark(label, f, x, grad):
    return Timer("f([x, grad])",
                 globals=locals(),
                 label=benchmark_name,
                 description=label,
                 sub_label=f"{tuple(x.shape)}",
                 num_threads=torch.get_num_threads()).blocked_autorange(min_run_time=2)


def compare(args):
    x, grad_output = args
    print(f"{tuple(x.shape)}")

    aten_func = aten.avg_pool3d_backward

    kernel_size = (2, 2, 2)
    stride = (2, 2, 2)
    padding = (0, 0, 0)
    ceil_mode = False
    count_include_pad = False
    divisor_override = None

    def f(args):
        x, grad_output = args
        val = aten_func(
            grad_output, x, kernel_size, stride, padding,
            ceil_mode, count_include_pad, divisor_override
        )
        return (val,)

    decomposed = make_fx(f, decomposition_table=decompositions, tracing_mode="fake")(args)
    compiled_decomposed = compile_fx_inner(decomposed, args, cudagraphs=False)
    yield benchmark(f"Decomposed", compiled_decomposed, *args)

    # Just show the first two generated kernels
    # torch._logging.set_logs(output_code=False)

    cuda_f = cudagraphify_impl(f, args, static_input_idxs=tuple(range(len(args))))
    yield benchmark(f"Eager", cuda_f, *args)


results = []
for x in gen_inputs():
    for res in compare(x):
        results.append(res)

compare = Compare(results)
compare.trim_significant_figures()
compare.print()
