# External components

- The NYUv2 dataset helper was adapted from the [CompletionFormer](https://github.com/youmi-zym/CompletionFormer) data pipeline. It is included for data loading only; the CompletionFormer model is not included. The local source repository carries an MIT license, reproduced in third_party_licenses/CompletionFormer_LICENSE.txt.
- The deformable-convolution operator in ops/DCNv2 follows [DCNv2](https://github.com/CharlesShang/DCNv2). It is required by DNINet's propagation module. Its BSD 3-Clause license is reproduced in third_party_licenses/DCNv2_LICENSE.txt.
- ops/BpOps contains the CUDA source used by the DNINet model's back-projection operator.

These components are identified separately so the repository does not present them as new DNINet contributions.
