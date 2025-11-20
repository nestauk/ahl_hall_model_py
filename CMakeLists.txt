cmake_minimum_required(VERSION 3.15)
project(ahl_hall_model_py)

find_package(pybind11 REQUIRED CONFIG)

pybind11_add_module(_core
    src/bindings.cpp
    src/adult_weight.cpp
    src/energy_build.cpp
)

target_include_directories(_core PRIVATE src)
install(TARGETS _core DESTINATION ahl_hall_model_py)
