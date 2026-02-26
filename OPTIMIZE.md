Performance Analysis: pocket.rs

1. Per-Token Tensor Allocation (Main Bottleneck)  


The step() function (lines 211-293) is called up to 1000 times per utterance, and every call creates multiple new ONNX tensors:

- Line 220-222: sequence tensor from cache_latent.clone() -- allocates, copies, creates ORT memory info, creates tensor, releases
  memory info
- Line 223-226: text_embeddings empty tensor -- same allocation dance
- Inside the flow step loop (lines 259-289), per LSD step (currently 1, but designed for more):
  - Line 262-264: c_tensor from conditioning (1024 floats)
  - Line 265-266: s_tensor (1 float)
  - Line 267-268: t_tensor (1 float)
  - Line 269-271: x_tensor from latent (32 floats)

Each Value::from_slice call at value.rs:58-122:

1. Allocates a Vec<u8> + copies data into it
2. Calls CreateCpuMemoryInfo (ORT API call)
3. Calls CreateTensorWithDataAsOrtValue (ORT API call)
4. Calls ReleaseMemoryInfo (ORT API call)
5. Then the Value is dropped after session.run(), calling ReleaseValue

Over 1000 tokens, that's ~6000+ tensor create/destroy cycles. This is likely the single largest overhead, especially for small
tensors (the s and t scalars are 1 float each).

Fix: Pre-allocate reusable tensors for sequence, text_embeddings, c_tensor, s_tensor, t_tensor, and x_tensor once before the
generation loop, then write data into them in-place via get_tensor_mutable_data. This eliminates all per-token allocation and ORT API
calls for tensor lifecycle management. You'd need a Value::write_data method or similar.

2. CString Allocation Per Inference Call

In Session::run() (session.rs:274-343), every call converts input/output name &strs to CStrings. The same names ("sequence",
"text_embeddings", "conditioning", "eos_logit", etc.) are converted on every token step.

Fix: Cache CString name pointers alongside the session. Build the name pointer arrays once during initialization.

3. cache_latent.clone() and Redundant Copies

- Line 219: let sequence_data: Vec<f32> = cache_latent.clone(); -- clones 32 floats into a new Vec, then immediately copies them
  again into an ORT tensor. Two copies for 128 bytes.
- Line 290: \*cache_latent = latent.clone(); -- clones the latent back after computation.

Fix: With pre-allocated tensors, you'd write directly into the tensor buffer, eliminating the intermediate clone entirely.

4. extract_tensor().to_vec() Copies

- Line 241-242: conditioning is extracted as a slice then .to_vec()'d (1024 floats = 4KB copy)
- Line 245-246: eos_logit_tensor is extracted then .to_vec()'d (1 float -- negligible but unnecessary)
- Line 315-316 in decode_audio: audio frame extracted and .to_vec()'d

These are needed because extract_tensor returns a reference tied to the Value lifetime, and outputs is consumed on line 248. If the
output Value lifetime were extended (or the conditioning data were written directly into a pre-allocated flow_step input), these
copies could be avoided.

5. deepclone of Flow State on Every Utterance

Line 397-406: At the start of each utterance, the reset flow state (all KV-cache tensors) is deep-cloned. This queries shape/type
metadata for each tensor, allocates new buffers, and copies the data. With a transformer-based flow model having multiple layers,
this could be significant.

Fix: This happens once per utterance, not per token, so it's less critical. But if the KV-cache is large, consider a batch_deepclone
that reuses a single pre-allocated buffer pool.

6. rand::thread_rng() Per Token

Line 251: rand::thread_rng() is called inside step() on every token. While thread_rng() is relatively cheap (thread-local lookup),
moving it outside the generation loop and passing it in would be cleaner and marginally faster.

7. Graph Optimization Level

onnx.rs:7: GRAPH_OPTIMIZATION_LEVEL = EnableBasic. The models are int8-quantized, which means they're already optimized for size, but
ORT's EnableExtended or EnableAll optimization levels can fuse operations and reduce kernel launches. This is particularly relevant
for CPU execution where operator fusion matters more.

Fix: Try GraphOptimizationLevel::EnableAll. This can significantly speed up CPU inference by fusing adjacent operators (e.g.,
MatMul+Add, Conv+BN+Relu).

8. Thread Count

onnx.rs:9: INTRA_OP_NUM_THREADS = 2. This limits parallelism within each operator. For CPU execution, this is conservative -- on a
multi-core machine, allowing more threads (4-8, or matching physical core count) can speed up the larger matrix operations in the
flow model and decoder.

Fix: Make this configurable or auto-detect based on num_cpus::get_physical(). The 73MB flow_lm_main model likely has matrices large
enough to benefit from more parallelism.

9. Why CPU May Outperform CUDA Here

The Pocket TTS architecture is autoregressive -- it generates one latent frame at a time with a KV-cache. Each step processes a [1,
1, 32] input through the transformer, which means the matrix operations are narrow (batch=1, sequence length=1). For CUDA, the
overhead of:

- CPU-to-GPU data transfer per token (the tensors created on CPU then used by CUDA EP)
- Kernel launch overhead for small matrices
- GPU memory allocation/deallocation for the many small tensors

...dominates the actual compute time. The GPU never gets enough work to saturate its ALUs. The CPU, with int8 quantization and
VNNI/AVX-512 instructions, processes these small matmuls with lower overhead per operation.

This is a well-known pattern: autoregressive models with batch=1 and short sequence lengths are CPU-bound, not compute-bound.

Summary: Prioritized Fixes

┌──────────┬─────────────────────────────────────────────────────────────┬──────────────────────────────────────────────────────┐
│ Priority │ Issue │ Impact │
├──────────┼─────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────┤
│ High │ Pre-allocate and reuse tensors in step() and decode_audio() │ Eliminates ~6000+ alloc/dealloc cycles per utterance │
├──────────┼─────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────┤
│ High │ GraphOptimizationLevel::EnableAll │ Free speedup from operator fusion on CPU │
├──────────┼─────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────┤
│ Medium │ Increase INTRA_OP_NUM_THREADS (4-8 or auto-detect) │ Better CPU utilization for larger matmuls │
├──────────┼─────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────┤
│ Medium │ Cache CString name pointers in session wrapper │ Eliminates per-run string allocations │
├──────────┼─────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────┤
│ Low │ Avoid .to_vec() on extracted tensors │ Saves ~4KB copy per token │
├──────────┼─────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────┤
│ Low │ Hoist thread_rng() out of the loop │ Minor but trivial │
└──────────┴─────────────────────────────────────────────────────────────┴──────────────────────────────────────────────────────┘

The tensor reuse fix (#1) would require adding a way to mutate tensor data in-place (e.g., Value::write_in_place(&mut self, data:
&[T]) using get_tensor_mutable_data). This is the most impactful single change since it removes the entire per-token allocation
overhead that dominates the non-inference time.
