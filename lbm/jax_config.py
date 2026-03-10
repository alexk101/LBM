import os

def configure_jax():
    os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ['XLA_FLAGS'] = (
        '--xla_gpu_triton_gemm_any=True '
        '--xla_gpu_enable_latency_hiding_scheduler=true '
    )