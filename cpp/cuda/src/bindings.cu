#include "mdescriptor/cuda/backend.hpp"

#include <pybind11/pybind11.h>

#include <memory>
#include <stdexcept>
#include <string>

namespace py = pybind11;
using mdescriptor::cuda::Backend;
using mdescriptor::cuda::CudaCancelledError;
using mdescriptor::cuda::CudaOutOfMemory;
using mdescriptor::cuda::CudaUnavailable;

PYBIND11_MODULE(_cuda, module) {
    module.doc() = "MDescriptor optional CUDA backend";
    module.attr("MODEL_SNAPSHOT_ABI") = 1;
    py::register_exception<CudaCancelledError>(module, "CudaCancelledError", PyExc_RuntimeError);
    py::register_exception<CudaUnavailable>(module, "CudaUnavailable", PyExc_ImportError);
    py::register_exception<CudaOutOfMemory>(module, "CudaOutOfMemory", PyExc_MemoryError);

    py::class_<Backend, std::shared_ptr<Backend>>(module, "CudaBackend")
        .def(py::init<std::string, py::dict>())
        .def_property_readonly("feature_count", &Backend::feature_count)
        .def("compute", &Backend::compute, py::arg("batch"), py::arg("control") = py::none())
        .def("predict", &Backend::predict, py::arg("batch"), py::arg("control") = py::none())
        .def("metadata", &Backend::metadata)
        .def("close", &Backend::close, py::call_guard<py::gil_scoped_release>());

    module.def("create_backend", [](const std::string& name, const py::dict& options) {
        try {
            return std::make_shared<Backend>(name, options);
        } catch (const CudaUnavailable& error) {
            PyErr_SetString(PyExc_ImportError, error.what());
            throw py::error_already_set();
        } catch (const CudaOutOfMemory& error) {
            PyErr_SetString(PyExc_MemoryError, error.what());
            throw py::error_already_set();
        }
    });
    module.def("create_predictor", [](const std::string& name, const py::dict& options) {
        if (name != "NEP" && name != "DPA4C") {
            throw std::invalid_argument("CUDA prediction supports NEP and DPA4C only");
        }
        py::dict predictor_options(options);
        predictor_options["_prediction"] = true;
        try {
            return std::make_shared<Backend>(name, predictor_options);
        } catch (const CudaUnavailable& error) {
            PyErr_SetString(PyExc_ImportError, error.what());
            throw py::error_already_set();
        } catch (const CudaOutOfMemory& error) {
            PyErr_SetString(PyExc_MemoryError, error.what());
            throw py::error_already_set();
        }
    });
}
