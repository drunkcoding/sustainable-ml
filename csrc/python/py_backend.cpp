#include <torch/extension.h>
#include "backend/delegated_mul.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    py::class_<MatMulThreadPool>(m, "matmul_threadpool")
        .def(py::init<int>())
        .def("enqueue", &MatMulThreadPool::Enqueue)
        .def("wait_all", &MatMulThreadPool::WaitAll);
        // .def("wait_gpu", &MatMulThreadPool::WaitGPU);
}
