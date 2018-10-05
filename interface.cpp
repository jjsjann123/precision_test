#include <torch/torch.h>
#include <ATen/ATen.h>

#include <vector>

at::Tensor element_wise_CUDA(const at::Tensor input,
                                  const at::Tensor f1,
                                  const at::Tensor b);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("elementwise", &element_wise_CUDA, "testing");
}
