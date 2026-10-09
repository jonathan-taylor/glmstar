#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/embed.h>
#include <glmnetpp>
#include "driver.h"

using namespace glmnetpp;
namespace py = pybind11;

// ELNET_POINT for dense X.
py::dict elnet_point_exp(
		 double alm0,
		 double almc,
		 double alpha,
		 int m,
		 int no,
		 int ni,
		 const Eigen::Ref<Eigen::MatrixXd> x,
		 Eigen::Ref<Eigen::VectorXd> r,
		 Eigen::Ref<Eigen::VectorXd> xv,
		 const Eigen::Ref<Eigen::VectorXd> v,
		 int intr,
		 const Eigen::Ref<Eigen::VectorXi> ju,
		 const Eigen::Ref<Eigen::VectorXd> vp,
		 const Eigen::Ref<Eigen::MatrixXd> cl,
		 int nx,
		 double thr,
		 int maxit,
		 Eigen::Ref<Eigen::VectorXd> a,
		 double aint,
		 Eigen::Ref<Eigen::VectorXd> g,
		 Eigen::Ref<Eigen::VectorXi> ia,
		 Eigen::Ref<Eigen::VectorXi> iy,
		 int iz,
		 Eigen::Ref<Eigen::VectorXi> mm,
		 int nino,
		 double rsqc,
		 int nlp,
		 int jerr
		 ) {
  using internal_t = ElnetPointInternal<
    util::glm_type::gaussian,
    util::mode_type<util::glm_type::gaussian>::wls,
    double, int, int>;
  using elnet_point_t = ElnetPoint<
    util::glm_type::gaussian,
    util::mode_type<util::glm_type::gaussian>::wls,
    internal_t>;
  auto f = [&]() {
    elnet_point_t elnet_point(
			      alm0, almc, alpha, x, r, xv, v, intr, ju, vp,
			      cl, nx, thr, maxit, a, aint, g, 
			      ia, iy, iz, mm, nino, rsqc, nlp);
    elnet_point.fit(m, jerr);
  };
  {
    py::gil_scoped_release nogil;
    run(f, jerr);
  }

  py::dict result;
  
  result["almc"] = almc;
  result["r"] = r;
  result["xv"] = xv;
  result["ju"] = ju;
  result["vp"] = vp;
  result["cl"] = cl;
  result["nx"] = nx;
  result["a"] = a;
  result["aint"] = aint;
  result["g"] = g;
  result["ia"] = ia;
  result["iy"] = iy;
  result["iz"] = iz;
  result["mm"] = mm;
  result["nino"] = nino;
  result["rsqc"] = rsqc;
  result["nlp"] = nlp;
  result["jerr"] = jerr;

  return result;
}

// WLS for sparse X.
py::dict spelnet_point_exp(
    double alm0,
    double almc,
    double alpha,
    int m,
    int no, // #obs = number of rows in x
    int ni, // #vars = number of cols in x
    py::array_t<double, py::array::c_style | py::array::forcecast> x_data_array,
    py::array_t<int, py::array::c_style | py::array::forcecast> x_indices_array,
    py::array_t<int, py::array::c_style | py::array::forcecast> x_indptr_array,
    const Eigen::Ref<Eigen::VectorXd> xm,
    const Eigen::Ref<Eigen::VectorXd> xs,
    Eigen::Ref<Eigen::VectorXd> r,
    Eigen::Ref<Eigen::VectorXd> xv,
    const Eigen::Ref<Eigen::VectorXd> v,
    int intr,
    const Eigen::Ref<Eigen::VectorXi> ju,
    const Eigen::Ref<Eigen::VectorXd> vp,
    const Eigen::Ref<Eigen::MatrixXd> cl,
    int nx,
    double thr,
    int maxit,
    Eigen::Ref<Eigen::VectorXd> a,
    double aint,
    Eigen::Ref<Eigen::VectorXd> g,
    Eigen::Ref<Eigen::VectorXi> ia,
    Eigen::Ref<Eigen::VectorXi> iy,
    int iz,
    Eigen::Ref<Eigen::VectorXi> mm,
    int nino,
    double rsqc,
    int nlp,
    int jerr
    )
{

  // Map the scipy csc_matrix x  to Eigen
  // This prevents copying. However, note the lack of 'const' use, but we take care not to change data
  Eigen::Map<Eigen::VectorXd> x_data_map(x_data_array.mutable_data(), x_data_array.size());
  Eigen::Map<Eigen::VectorXi> x_indices_map(x_indices_array.mutable_data(), x_indices_array.size());
  Eigen::Map<Eigen::VectorXi> x_indptr_map(x_indptr_array.mutable_data(), x_indptr_array.size());
  // Create MappedSparseMatrix from the mapped arrays
  Eigen::MappedSparseMatrix<double, Eigen::ColMajor> eigen_x(no, ni, x_data_array.size(), 
							     x_indptr_map.data(), x_indices_map.data(),
							     x_data_map.data());

    using internal_t = SpElnetPointInternal<
        util::glm_type::gaussian,
        util::mode_type<util::glm_type::gaussian>::wls,
        double, int, int>;
    using elnet_point_t = SpElnetPoint<
        util::glm_type::gaussian,
        util::mode_type<util::glm_type::gaussian>::wls,
        internal_t>;
    auto f = [&]() {
        elnet_point_t elnet_point(
                alm0, almc, alpha, eigen_x, r, xm, xs, xv, v, intr, ju, vp,
                cl, nx, thr, maxit, a, aint, g, 
                ia, iy, iz, mm, nino, rsqc, nlp);
        elnet_point.fit(m, jerr);
    };
    {
      py::gil_scoped_release nogil;
      run(f, jerr);
    }

    py::dict result;
    
    result["almc"] = almc;
    result["r"] = r;
    result["xv"] = xv;
    result["ju"] = ju;
    result["vp"] = vp;
    result["cl"] = cl;
    result["nx"] = nx;
    result["a"] = a;
    result["aint"] = aint;
    result["g"] = g;
    result["ia"] = ia;
    result["iy"] = iy;
    result["iz"] = iz;
    result["mm"] = mm;
    result["nino"] = nino;
    result["rsqc"] = rsqc;
    result["nlp"] = nlp;
    result["jerr"] = jerr;

    return result;
}


PYBIND11_MODULE(_elnet_point, m) {
    m.def("elnet_point", &elnet_point_exp,
	  py::arg("alm0"),
	  py::arg("almc"),
	  py::arg("alpha"),
	  py::arg("m"),
	  py::arg("no"),
	  py::arg("ni"),
	  py::arg("x"),	  
	  py::arg("r"),
	  py::arg("xv"),
	  py::arg("v"),
	  py::arg("intr"),
	  py::arg("ju"),
	  py::arg("vp"),
	  py::arg("cl"),
	  py::arg("nx"),
	  py::arg("thr"),
	  py::arg("maxit"),
	  py::arg("a"),
	  py::arg("aint"),
	  py::arg("g"),
	  py::arg("ia"),
	  py::arg("iy"),
	  py::arg("iz"),
	  py::arg("mm"),
	  py::arg("nino"),
	  py::arg("rsqc"),
	  py::arg("nlp"),
	  py::arg("jerr"));
    
    m.def("spelnet_point", &spelnet_point_exp,
	  py::arg("alm0"),
	  py::arg("almc"),
	  py::arg("alpha"),
	  py::arg("m"),
	  py::arg("no"),
	  py::arg("ni"),
	  py::arg("x_data_array"),
	  py::arg("x_indices_array"),
	  py::arg("x_indptr_array"),
	  py::arg("xm"),
	  py::arg("xs"),
	  py::arg("r"),
	  py::arg("xv"),
	  py::arg("v"),
	  py::arg("intr"),
	  py::arg("ju"),
	  py::arg("vp"),
	  py::arg("cl"),
	  py::arg("nx"),
	  py::arg("thr"),
	  py::arg("maxit"),
	  py::arg("a"),
	  py::arg("aint"),
	  py::arg("g"),
	  py::arg("ia"),
	  py::arg("iy"),
	  py::arg("iz"),
	  py::arg("mm"),
	  py::arg("nino"),
	  py::arg("rsqc"),
	  py::arg("nlp"),
	  py::arg("jerr"));

}


