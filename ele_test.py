import torch
import numpy as np
print("using jit")
from torch.utils.cpp_extension import load
muladd = load(name='muladd', sources=['./interface.cpp', './kernel.cu'])

def compare(inp0, inp1, rtol, atol):
    return np.allclose(inp0, inp1, rtol=rtol, atol=atol)

feature_size = 1
space_size = 1
batch_size = 16
dtype = np.float32
f1 = np.random.randn(feature_size).astype(dtype)
b = np.random.randn(feature_size).astype(dtype)
inp = (np.random.randn(batch_size, feature_size, space_size, space_size) * 2.0 + 3.5).astype(dtype)
type_tensor = torch.cuda.FloatTensor

inp_t = type_tensor(inp)
data_t = type_tensor(inp.transpose(1, 0, 2, 3).reshape(feature_size, -1))

f1_t = type_tensor(f1)
b_t = type_tensor(b)
f1_s = f1_t.view(feature_size, 1, 1)
b_s = b_t.view(feature_size, 1, 1)
np_f1 = f1_s.cpu().numpy()
np_b = b_s.cpu().numpy()
np_inp = inp_t.cpu().numpy()

# numpy execution
#np_out = (np_f1 * np_inp + np_b)
np_out = np_b + (np_inp * np_f1)
#np_out = (np_f1 * np_inp)

# pytorch execution
#out_t = f1_s * inp_t + b_s
out_t = b_s + inp_t * f1_s
out = out_t.cpu().numpy()
#out = (f1_s * inp_t + b_s).cpu().numpy()
#out = (f1_s * inp_t).cpu().numpy()

# kernel execution
out_sbn = muladd.elementwise(inp_t, f1_s, b_s)

print(compare(np_out, out, 1e-8, 1e-8)) 
print(np.array_equal(np_out, out))

print(compare(np_out, out_sbn.clone().cpu().detach().numpy(), 1e-3, 1e-3))
print(compare(np_out, out_sbn.clone().cpu().detach().numpy(), 1e-7, 1e-7))
print(np.array_equal(np_out, out_sbn.clone().cpu().detach().numpy()))

a = np_out.flatten()
b = out_sbn.clone().cpu().detach().numpy().flatten()
z = a - b
ind = z.nonzero()
print(ind)
print(z[ind])
print(a[ind])
print(b[ind])
print(out_t[ind].cpu().flatten().numpy())
