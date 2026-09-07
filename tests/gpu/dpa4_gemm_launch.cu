// Standalone regression: include the implementation to exercise its private
// GEMM seam without allocating the full descriptor's edge workspace.
#include "../../cpp/cuda/src/dpa4.cu"
#include <iostream>

int main() {
    using namespace mdescriptor::cuda;
    for (const auto shape : {std::pair<std::int64_t, std::int64_t>{65535LL * 16 + 1, 1},
                             {2, 65536}}) {
        const auto rows = shape.first;
        const auto batches = shape.second;
        const auto size = rows * batches;
        std::vector<float> input(size, 2.0F), output(size, -1.0F);
        float *left, *right, *product;
        check_cuda(cudaMalloc(&left, size * sizeof(float)), "left");
        check_cuda(cudaMalloc(&right, batches * sizeof(float)), "right");
        check_cuda(cudaMalloc(&product, size * sizeof(float)), "product");
        std::vector<float> weights(batches, 3.0F);
        check_cuda(cudaMemcpy(left, input.data(), size * sizeof(float), cudaMemcpyHostToDevice), "input");
        check_cuda(cudaMemcpy(right, weights.data(), batches * sizeof(float), cudaMemcpyHostToDevice), "weights");
        launch_row_major_gemm(left, right, product, rows, 1, 1, 1, 1, 1,
                              rows, 1, rows, batches, nullptr, "boundary GEMM");
        check_cuda(cudaMemcpy(output.data(), product, size * sizeof(float), cudaMemcpyDeviceToHost), "output");
        if (!std::all_of(output.begin(), output.end(), [](float x) { return x == 6.0F; })) return 1;
        cudaFree(left); cudaFree(right); cudaFree(product);
        std::cout << "passed rows=" << rows << " batches=" << batches << '\n';
    }
}
