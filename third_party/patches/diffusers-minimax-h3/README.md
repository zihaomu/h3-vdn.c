# Pinned Diffusers changes for the MiniMax-H3 BF16 oracle

Base repository: <https://github.com/huggingface/diffusers>

Base revision: `3a2f35d4efa4c059c8bfb3bc0d6c906264895c81`
Resulting revision used by the accepted run:
`c41d51befdaef751cbd98f07a8eb4f4279d1578c`

Apply the patches in lexical order to a clean checkout of the base revision:

```sh
git checkout 3a2f35d4efa4c059c8bfb3bc0d6c906264895c81
git am /path/to/h3-vdn.c/third_party/patches/diffusers-minimax-h3/*.patch
```

Patch SHA-256 values:

| Patch | SHA-256 |
|---|---|
| `0001-MiniMaxH3-pin-the-AdaLN-SiLU-to-fp32-explicitly.patch` | `ee22a2504f5a050da0fefb5d6f9242b66f2487fec91dd4b312bc84dea4ccc938` |
| `0002-Fix-MiniMax-H3-inference-steps-to-count-NFEs.patch` | `41279460b277bf00c6474d2a26a664e636c003b6b68ade61ed6f7dda04c015cd` |

These are source-code fixtures. They contain no model weights or generated
media.
