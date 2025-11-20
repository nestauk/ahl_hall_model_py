#include "adult_weight.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Helper to convert std::vector matrix to NumericMatrix
NumericMatrix to_mat(const std::vector<std::vector<double>> &in)
{
    return NumericMatrix(in);
}

// Helper to convert std::vector to NumericVector
// (NumericVector has a constructor from std::vector<double>)
NumericVector to_vec(const std::vector<double> &in)
{
    return NumericVector(in);
}

// Wrapper 1: Default
py::dict py_adult_weight(std::vector<double> bw, std::vector<double> ht, std::vector<double> age,
                         std::vector<double> sex, std::vector<std::vector<double>> EIchange,
                         std::vector<std::vector<double>> NAchange, std::vector<double> PAL,
                         std::vector<double> pcarb_base, std::vector<double> pcarb, double dt,
                         double days, bool checkValues)
{

    Adult Person(to_vec(bw), to_vec(ht), to_vec(age), to_vec(sex),
                 to_mat(EIchange), to_mat(NAchange),
                 to_vec(PAL), to_vec(pcarb), to_vec(pcarb_base), dt, checkValues);
    return Person.rk4(days);
}

// Wrapper 2: With EI or Fat
py::dict py_adult_weight_EI(std::vector<double> bw, std::vector<double> ht, std::vector<double> age,
                            std::vector<double> sex, std::vector<std::vector<double>> EIchange,
                            std::vector<std::vector<double>> NAchange, std::vector<double> PAL,
                            std::vector<double> pcarb_base, std::vector<double> pcarb, double dt,
                            std::vector<double> extradata, double days, bool checkValues, bool isEnergy)
{

    Adult Person(to_vec(bw), to_vec(ht), to_vec(age), to_vec(sex),
                 to_mat(EIchange), to_mat(NAchange),
                 to_vec(PAL), to_vec(pcarb), to_vec(pcarb_base), dt,
                 to_vec(extradata), checkValues, isEnergy);
    return Person.rk4(days);
}

// Wrapper 3: With EI AND Fat
py::dict py_adult_weight_EI_fat(std::vector<double> bw, std::vector<double> ht, std::vector<double> age,
                                std::vector<double> sex, std::vector<std::vector<double>> EIchange,
                                std::vector<std::vector<double>> NAchange, std::vector<double> PAL,
                                std::vector<double> pcarb_base, std::vector<double> pcarb, double dt,
                                std::vector<double> input_EI, std::vector<double> input_fat,
                                double days, bool checkValues)
{

    Adult Person(to_vec(bw), to_vec(ht), to_vec(age), to_vec(sex),
                 to_mat(EIchange), to_mat(NAchange),
                 to_vec(PAL), to_vec(pcarb), to_vec(pcarb_base), dt,
                 to_vec(input_EI), to_vec(input_fat), checkValues);
    return Person.rk4(days);
}

PYBIND11_MODULE(_core, m)
{
    m.def("adult_weight_wrapper", &py_adult_weight);
    m.def("adult_weight_wrapper_EI", &py_adult_weight_EI);
    m.def("adult_weight_wrapper_EI_fat", &py_adult_weight_EI_fat);
}
