/*
<%
setup_pybind11(cfg)
%>
*/
// Time-decayed item popularity used by the TIDE and CausalEPP baselines.
// Derived from the TIDE authors' implementation; made dataset independent
// (interaction file path and catalog size are passed at load time).
//
// For item i with training interaction times t_1 <= ... <= t_n:
//   pop_1 = 0,   pop_j = (pop_{j-1} + 1) * exp(-(t_j - t_{j-1}) / tau_i)
// and the popularity queried at time t is
//   0                                      if t < t_1
//   pop_j + 1                              if t == t_j (last interaction at or before t)
//   (pop_j + 1) * exp(-(t - t_j) / tau_i)  otherwise, t_j = last interaction <= t
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

static std::vector<std::vector<long long>> g_ts;   // per item: interaction timestamps (ascending)
static std::vector<std::vector<double>> g_pop;     // per item: decayed popularity right after each interaction
static std::vector<double> g_tau;

// item_interactions.csv: one row per item id, "idx,count,ts1,ts2,..." (ts ascending)
void load_popularity(py::array_t<double> tau_p, const std::string& path) {
    py::buffer_info buf = tau_p.request();
    const double* tau = static_cast<const double*>(buf.ptr);
    const long n_tau = static_cast<long>(buf.size);
    std::ifstream fin(path);
    if (!fin) throw std::runtime_error("cannot open " + path);
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(fin, line)) {
        if (!line.empty()) lines.push_back(line);
    }
    const long n_items = static_cast<long>(lines.size());
    if (n_items != n_tau)
        throw std::runtime_error("tau length does not match the number of item rows in " + path);
    g_ts.assign(n_items, {});
    g_pop.assign(n_items, {});
    g_tau.assign(n_items, 0.0);
    for (long i = 0; i < n_items; ++i) {
        std::stringstream ss(lines[i]);
        std::string tok;
        std::getline(ss, tok, ',');
        const long idx = std::atol(tok.c_str());
        if (idx != i) throw std::runtime_error("item rows must be ordered by id in " + path);
        std::getline(ss, tok, ',');
        const long cnt = std::atol(tok.c_str());
        g_tau[i] = tau[i];
        std::vector<long long>& ts = g_ts[i];
        ts.reserve(cnt);
        while (std::getline(ss, tok, ',')) ts.push_back(std::atoll(tok.c_str()));
        if (static_cast<long>(ts.size()) != cnt)
            throw std::runtime_error("interaction count mismatch for item " + std::to_string(i));
        std::vector<double>& pop = g_pop[i];
        pop.assign(cnt, 0.0);
        for (long j = 1; j < cnt; ++j)
            pop[j] = (pop[j - 1] + 1) * std::exp(-1 * double(ts[j] - ts[j - 1]) / g_tau[i]);
    }
}

py::array_t<double> popularity(py::array_t<double> item_p, py::array_t<double> timestamp_p) {
    py::buffer_info b1 = item_p.request();
    py::buffer_info b2 = timestamp_p.request();
    auto result = py::array_t<double>(b1.size);
    py::buffer_info b3 = result.request();
    const double* items = static_cast<const double*>(b1.ptr);
    const double* stamps = static_cast<const double*>(b2.ptr);
    double* out = static_cast<double*>(b3.ptr);
    for (long i = 0; i < static_cast<long>(b1.shape[0]); ++i) {
        const long item = static_cast<long>(items[i]);
        const long long t = static_cast<long long>(stamps[i]);
        const std::vector<long long>& ts = g_ts[item];
        const long n = static_cast<long>(ts.size());
        if (n == 0) { out[i] = 0; continue; }
        long j = 0;
        while (j + 100 < n && ts[j + 100] <= t) j += 100;
        while (j + 10 < n && ts[j + 10] <= t) j += 10;
        while (j < n && ts[j] <= t) ++j;
        if (j == 0) out[i] = 0;
        else if (ts[j - 1] == t) out[i] = g_pop[item][j - 1] + 1;
        else out[i] = (g_pop[item][j - 1] + 1) * std::exp(-1 * double(t - ts[j - 1]) / g_tau[item]);
    }
    return result;
}

PYBIND11_MODULE(stfr_popularity, m) {
    m.doc() = "time-decayed item popularity (TIDE / CausalEPP)";
    m.def("load_popularity", &load_popularity, "load item interaction times and precompute decayed popularity");
    m.def("popularity", &popularity, "decayed popularity of items at the given timestamps");
}
