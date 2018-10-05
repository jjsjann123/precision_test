#include <iostream>
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>


#include <cuda.h>
#include <cuda_runtime.h>


#include <vector>


__host__ int get_tensor_spatial_size(const at::Tensor& input)
{
  auto space_size = input.size(2);
  for (int i = 3; i < input.ndimension(); i++) {
    space_size *= input.size(i);
  }
  return space_size;
}

__host__ at::ScalarType promote_scalartype(const at::Tensor& input)
{
  return input.type().scalarType() == at::ScalarType::Half ?
           at::ScalarType::Float : input.type().scalarType();
}

template <typename scalar_t>
__global__ void element_wise_kernel(
      const scalar_t* __restrict__ input,
      const scalar_t* __restrict__ f1,
      const scalar_t* __restrict__ b,
      scalar_t* __restrict__ out,
      const int ss) {

  int address_base = blockIdx.x*ss + blockIdx.y*gridDim.x*ss;

  auto f1_c = f1[blockIdx.x];
  auto b_c = b[blockIdx.x];

  for (int offset = threadIdx.x; offset < ss ; offset+= blockDim.x) {
    out[address_base+offset] = f1_c * input[address_base+offset] + b_c;
    //out[address_base+offset] = f1_c * input[address_base+offset];
  }
}
at::Tensor element_wise_CUDA(
    const at::Tensor input,
    const at::Tensor f1,
    const at::Tensor b) {
  const auto batch_size = input.size(0);
  const auto feature_size = input.size(1);
  at::Tensor out = at::empty_like(input);

  auto space_size = get_tensor_spatial_size(input);

  const dim3 block(512);
  const dim3 grid(feature_size, batch_size);
  AT_DISPATCH_FLOATING_TYPES(input.type(), "batchnorm_forward", ([&] {
    element_wise_kernel<scalar_t><<<grid, block>>>(
        input.data<scalar_t>(),
        f1.data<scalar_t>(),
        b.data<scalar_t>(),
        out.data<scalar_t>(),
        space_size);
  }));
  
  return out;
}
