// Standalone regression: include the implementation to exercise its private
// GEMM seam without allocating the full descriptor's edge workspace.
#include "../../cpp/cuda/src/dpa4.cu"
#include <cmath>
#include <iostream>

namespace {

void check_strided_product(int rows, int columns, int inner, int layout) {
    using namespace mdescriptor::cuda;
    constexpr int batches = 3;
    const int lda = layout == 2 ? 640 : inner + 7;
    const int ldb = columns + 5;
    const int ldc = columns + 3;
    const int sa = layout == 1 ? 0 : layout == 2 ? 192 : rows * lda + 11;
    const int sb = layout == 0 ? inner * ldb + 13 : 0;
    const int sc = rows * ldc + 17;
    const std::size_t as = (batches - 1U) * sa + (rows - 1U) * lda + inner;
    const std::size_t bs = (batches - 1U) * sb + (inner - 1U) * ldb + columns;
    const std::size_t cs = batches * static_cast<std::size_t>(sc);
    std::vector<float> a(as), b(bs), expected(cs, -123.0F), actual(cs, -123.0F);
    for (std::size_t i = 0; i < as; ++i) {
        a[i] = std::sin(static_cast<float>(i % 1009) * 0.37F) * 0.17F;
    }
    for (std::size_t i = 0; i < bs; ++i) {
        b[i] = std::cos(static_cast<float>(i % 1013) * 0.41F) * 0.13F;
    }
    for (int batch = 0; batch < batches; ++batch) {
        for (int row = 0; row < rows; ++row) {
            for (int column = 0; column < columns; ++column) {
                float value = 0.0F;
                for (int k = 0; k < inner; ++k) {
                    value = std::fma(a[batch * sa + row * lda + k],
                                     b[batch * sb + k * ldb + column], value);
                }
                expected[batch * sc + row * ldc + column] = value;
            }
        }
    }
    float *da, *db, *dc;
    check_cuda(cudaMalloc(&da, as * sizeof(float)), "strided A");
    check_cuda(cudaMalloc(&db, bs * sizeof(float)), "strided B");
    check_cuda(cudaMalloc(&dc, cs * sizeof(float)), "strided C");
    check_cuda(cudaMemcpy(da, a.data(), as * sizeof(float), cudaMemcpyHostToDevice), "copy A");
    check_cuda(cudaMemcpy(db, b.data(), bs * sizeof(float), cudaMemcpyHostToDevice), "copy B");
    check_cuda(cudaMemcpy(dc, actual.data(), cs * sizeof(float), cudaMemcpyHostToDevice), "copy C");
    launch_row_major_gemm(da, db, dc, rows, columns, inner, lda, ldb, ldc,
                         sa, sb, sc, batches, nullptr, "strided GEMM");
    check_cuda(cudaMemcpy(actual.data(), dc, cs * sizeof(float), cudaMemcpyDeviceToHost), "read C");
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    // Also checks untouched row/batch padding, not only the dense product.
    if (actual != expected) {
        throw std::runtime_error("strided GEMM differs from ordered FP32 FMA reference");
    }
    std::cout << "passed shape=" << rows << 'x' << columns << 'x' << inner
              << " layout=" << layout << '\n';
}

} // namespace

int main() {
    using namespace mdescriptor::cuda;
    for (const auto shape : {
             std::array<int, 3>{1, 25, 1}, {31, 128, 17}, {32, 64, 16},
             {33, 65, 17}, {48, 192, 152}, {63, 384, 128}, {64, 384, 128},
             {65, 385, 129}, {152, 384, 48}, {65, 1152, 64}}) {
        for (int layout : {0, 1}) {
            check_strided_product(shape[0], shape[1], shape[2], layout);
        }
    }
    check_strided_product(65, 384, 192, 2);
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
