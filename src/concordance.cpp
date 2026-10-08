// Harrell's concordance index in O(n log n), following the algorithm in the
// R survival package's "concordance" vignette: subjects are processed in
// decreasing time order, keeping the weights of the current risk set in a
// binary indexed tree keyed by the rank of the predictor.

#include <algorithm>
#include <numeric>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

namespace {

struct Fenwick {
  std::vector<double> tree;
  explicit Fenwick(std::size_t n) : tree(n + 1, 0.) {}
  void add(std::size_t i, double w) {   // 0-based index i
    for (++i; i < tree.size(); i += i & (~i + 1)) tree[i] += w;
  }
  double prefix(std::size_t i) const {  // sum over indices < i
    double s = 0.;
    for (; i > 0; i -= i & (~i + 1)) s += tree[i];
    return s;
  }
};

// Accumulate, over comparable pairs within the subjects `idx`, the sum of
// w_i w_j (1 + sign(pred_i - pred_j)) / 2 into num and of w_i w_j into den,
// where i is an event and j is at risk at its time but not an event tied with it.
void concordance_stratum(const std::vector<int> &idx,
                         const Eigen::Ref<const Eigen::MatrixXd> &pred,
                         const Eigen::Ref<const Eigen::VectorXd> &time,
                         const Eigen::Ref<const Eigen::VectorXi> &status,
                         const Eigen::Ref<const Eigen::VectorXd> &start,
                         const Eigen::Ref<const Eigen::VectorXd> &weight,
                         Eigen::Ref<Eigen::VectorXd> num,
                         double &den)
{
  const std::size_t m = idx.size();

  std::vector<int> by_stop(idx), by_start(idx);
  std::sort(by_stop.begin(), by_stop.end(),
            [&](int a, int b) { return time(a) > time(b); });
  std::sort(by_start.begin(), by_start.end(),
            [&](int a, int b) { return start(a) > start(b); });

  std::vector<int> order(m);
  std::vector<std::size_t> rank(pred.rows());

  for (Eigen::Index l = 0; l < pred.cols(); ++l) {
    // dense ranks of the predictor within the stratum, ties sharing a rank
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return pred(idx[a], l) < pred(idx[b], l); });
    std::size_t r = 0;
    for (std::size_t k = 0; k < m; ++k) {
      if (k > 0 && pred(idx[order[k]], l) > pred(idx[order[k - 1]], l)) ++r;
      rank[idx[order[k]]] = r;
    }

    Fenwick tree(r + 1);
    double total = 0.;
    double den_l = 0.;
    std::size_t p_stop = 0, p_start = 0;

    while (p_stop < m) {
      // next event time, adding everyone with a later stop time
      while (p_stop < m && !status(by_stop[p_stop])) {
        int j = by_stop[p_stop++];
        tree.add(rank[j], weight(j));
        total += weight(j);
      }
      if (p_stop == m) break;
      const double t = time(by_stop[p_stop]);

      // subjects with stop == t: censored ones are at risk, events are queried
      std::size_t group_end = p_stop;
      while (group_end < m && time(by_stop[group_end]) == t) {
        int j = by_stop[group_end++];
        if (!status(j)) {
          tree.add(rank[j], weight(j));
          total += weight(j);
        }
      }
      // subjects whose interval starts at or after t are not at risk at t
      while (p_start < m && start(by_start[p_start]) >= t) {
        int j = by_start[p_start++];
        tree.add(rank[j], -weight(j));
        total -= weight(j);
      }
      for (std::size_t k = p_stop; k < group_end; ++k) {
        int i = by_stop[k];
        if (!status(i)) continue;
        double less = tree.prefix(rank[i]);
        double equal = tree.prefix(rank[i] + 1) - less;
        num(l) += weight(i) * (less + 0.5 * equal);
        den_l += weight(i) * total;
      }
      // the events at t are at risk at earlier times
      for (std::size_t k = p_stop; k < group_end; ++k) {
        int i = by_stop[k];
        if (status(i)) {
          tree.add(rank[i], weight(i));
          total += weight(i);
        }
      }
      p_stop = group_end;
    }
    if (l == 0) den += den_l;
  }
}

} // namespace

// Returns (num, den): the C index of column l of pred is num[l] / den.
py::tuple concordance(const Eigen::Ref<const Eigen::MatrixXd> &pred,
                      const Eigen::Ref<const Eigen::VectorXd> &time,
                      const Eigen::Ref<const Eigen::VectorXi> &status,
                      const Eigen::Ref<const Eigen::VectorXd> &start,
                      const Eigen::Ref<const Eigen::VectorXi> &strata,
                      const Eigen::Ref<const Eigen::VectorXd> &weight)
{
  const Eigen::Index n = time.size();
  if (pred.rows() != n || status.size() != n || start.size() != n ||
      strata.size() != n || weight.size() != n)
    throw std::invalid_argument("concordance: inputs must have the same number of rows");

  int nstrata = n > 0 ? strata.maxCoeff() + 1 : 0;
  std::vector<std::vector<int>> groups(nstrata);
  for (Eigen::Index i = 0; i < n; ++i) groups[strata(i)].push_back(static_cast<int>(i));

  Eigen::VectorXd num = Eigen::VectorXd::Zero(pred.cols());
  double den = 0.;
  for (const auto &idx : groups)
    if (!idx.empty())
      concordance_stratum(idx, pred, time, status, start, weight, num, den);
  return py::make_tuple(num, den);
}

PYBIND11_MODULE(_concordance, m) {
  m.def("concordance", &concordance,
        py::arg("pred"), py::arg("time"), py::arg("status"), py::arg("start"),
        py::arg("strata"), py::arg("weight"));
}
