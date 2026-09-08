# Third-party notices

## SageAttention-AMD

The optional experimental `H3_VDN_SDPA=sage-i8-bf16` implementation is built
from the `third_party/sageattention-amd` Git submodule, fixed for this revision
at commit `18d949018cec1467ac6d30c12b3494f2f51bb552`. SageAttention-AMD is
licensed under the Apache License, Version 2.0; its license text and detailed
notices are retained in the submodule as `LICENSE` and
`THIRD_PARTY_NOTICES.md`.

Its gfx12 WMMA fragment-packing approach was informed by thu-ml/SageAttention
pull request #368 at commit
`66f5e64c9e36084c863a4480e570069245e58f90`. Copyright notices include the
SageAttention team (2024) and Advanced Micro Devices, Inc. (2026). The
integrated implementation removes Torch/ATen, uses raw HIP pointers with a
caller-owned stream/workspace, implements H3's interval mask, and uses BF16 PV.

## ccv Metal kernels

The rectangular Morton decoder and the dynamic symmetric int8 quantization /
Metal 4 TensorOps scheduling design in `h3_shaders.metal` are adapted from
ccv's Metal FlashAttention `NAMatMulKernel` and `NAInt8MatMulKernel`,
distributed under the following BSD-3-Clause license:

Copyright (c) 2010, Liu Liu
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

- Redistributions of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
- Neither the name of the authors nor the names of its contributors may be
  used to endorse or promote products derived from this software without
  specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
