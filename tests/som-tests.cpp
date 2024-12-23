//
// Created by nmillerns on 20/12/24.
//
#include <iostream>
#include <sstream>
#include <stdexcept>
#include "som.hpp"

#define _SS std::stringstream()
#define _PASS std::cout << "PASS" << std::endl;

void basicAssert(bool condition, const std::stringstream msg_stream) {
    if (!condition) {
        throw std::runtime_error(msg_stream.str());
    }
}

void testSimple1dMap() {
    std::cout << "Simple 1d Map..." << std::flush;
    GridSelfOrganizingMap<1> grid(3, 2);
    for (auto neuron_i = grid.begin(); neuron_i != grid.end(); ++neuron_i) {
        basicAssert(1 == neuron_i->length(), _SS << "Wrong Neuron Length 1 vs " << neuron_i->length());
        neuron_i->weightVector()[0] = 0.01 * (neuron_i.row() + neuron_i.col());
    }
    basicAssert(3 == grid.width(), _SS << "Wrong width 3 vs " << grid.width());
    basicAssert(2 == grid.height(),  _SS << "Wrong height 2 vs " << grid.height());

    const double actrual_00 = grid.getNeuron(0, 0).weightVector()[0];
    const double actrual_01 = grid.getNeuron(0, 1).weightVector()[0];
    const double actrual_02 = grid.getNeuron(0, 2).weightVector()[0];
    const double actrual_10 = grid.getNeuron(1, 0).weightVector()[0];
    const double actrual_11 = grid.getNeuron(1, 1).weightVector()[0];
    const double actrual_12 = grid.getNeuron(1, 2).weightVector()[0];

    basicAssert(0.00 == actrual_00, _SS << "Neuron wrong at 0, 0: 0.0 vs " << actrual_00);
    basicAssert(0.01 == actrual_01, _SS << "Neuron wrong at 0, 1: 0.0 vs " << actrual_01);
    basicAssert(0.02 == actrual_02, _SS << "Neuron wrong at 0, 2: 0.0 vs " << actrual_02);
    basicAssert(0.01 == actrual_10, _SS << "Neuron wrong at 1, 0: 0.0 vs " << actrual_10);
    basicAssert(0.02 == actrual_11, _SS << "Neuron wrong at 1, 1: 0.0 vs " << actrual_11);
    basicAssert(0.03 == actrual_12, _SS << "Neuron wrong at 1, 2: 0.0 vs " << actrual_12);
    _PASS
}

int main(int argc, char** argv) {
    std::cout << "Running Unit Tests..." << std::endl;
    testSimple1dMap();
    return 0;
}
