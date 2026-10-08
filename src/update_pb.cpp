#include <pybind11/embed.h>

namespace py = pybind11;

// Called from within the C++ solvers, which run with the GIL released,
// so it must be reacquired before touching the progress bar object.
void update_pb(const py::object& pb, int step_inc) {
    py::gil_scoped_acquire gil;
    pb.attr("update")(step_inc);
}
