//
// Created by nmillerns on 20/12/24.
//
#include <iostream>
#include <sstream>
#include <stdexcept>
#include "som.hpp"
#include "vis.hpp"

#define _SS StringStreamHelper()
#define _PASS std::cout << "PASS" << std::endl;

void basicAssert(bool condition, const StringStreamHelper& msg_stream) {
    if (!condition) {
        throw std::runtime_error(msg_stream.str());
    }
}

template<size_t N, size_t M>
void testConstrictionWidthHeight() {
    std::cout << "Grid Construction " << N << 'x' << M << "..." << std::flush;
    GridSelfOrganizingMap2d grid(N, M);
    basicAssert(N == grid.width(), _SS << " wrong width in grid " << N << " vs " << grid.width());
    basicAssert(M == grid.height(), _SS << " wrong height in grid " << N << " vs " << grid.width());
    _PASS;
}

template<size_t N, size_t M>
void testGridIteration() {
    std::cout << "Grid Iterator " << N << 'x' << M << "..." << std::flush;
    GridSelfOrganizingMap2d grid(N, M);
    size_t expected_row = 0;
    size_t expected_col = 0;
    // Iterate row wise and check that row, col coordiinates are correct
    for (auto rowwise_iterator = grid.begin(); rowwise_iterator != grid.end(); ++rowwise_iterator) {
        basicAssert(expected_row == rowwise_iterator.row(),
                    _SS << " wrong row in iterator " << expected_row << " vs " << rowwise_iterator.row());
        basicAssert(expected_col == rowwise_iterator.col(),
                    _SS << " wrong row in iterator " << expected_col << " vs " << rowwise_iterator.col());
        ++expected_col;
        // When we hit the end of the row, reset and move down a row
        if (expected_col == N) {
            expected_col = 0;
            ++expected_row;
        }
    }
    _PASS
}

void testSimple1dMap() {
    std::cout << "Simple Map on 1D Space..." << std::flush;
    GridSelfOrganizingMap<1> grid(3, 2);

    for (auto neuron_i = grid.begin(); neuron_i != grid.end(); ++neuron_i) {
        basicAssert(1 == neuron_i->length(), _SS << "Wrong Neuron Length 1 vs " << neuron_i->length());
        neuron_i->weightVector()[0] = 0.01 * (neuron_i.row() + neuron_i.col());
    }

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

void testFitSinglePointSingleNode() {
    std::cout << "Fit Single-Node SOM to Single Point..." << std::flush;
    // A single node should snap exactly to a dataset of a single point
    LinearSelfOrganizingMap<11> som(1);
    std::array<double, 11> point;
    for (size_t dim = 0; dim < 11; ++dim) {
        point[dim] = 0.1 * dim;
    }
    som.fitData({point});
    const auto& neuron = som.getNeuron(0);
    for (size_t dim = 0; dim < 11; ++dim) {
        const double actual = neuron.weightVector()[dim];
        basicAssert(point[dim] == actual, _SS <<
                    "Wrong weight value at " << dim << ". " << point[dim] << " vs " << actual);
    }
    _PASS;
}

int main(int argc, char** argv) {
    std::cout << "Running Unit Tests..." << std::endl;
    try {
        testConstrictionWidthHeight<0, 0>();
        testConstrictionWidthHeight<11, 0>();
        testConstrictionWidthHeight<0, 200>();
        testConstrictionWidthHeight<1276, 554>();
        testGridIteration<0, 0>();
        testGridIteration<1, 0>();
        testGridIteration<0, 1>();
        testGridIteration<1, 1>();
        testGridIteration<1, 20>();
        testGridIteration<55, 1>();
        testGridIteration<640, 480>();
        testSimple1dMap();
        testFitSinglePointSingleNode();
    } catch (std::exception& e) {
        std::cout << "FAIL" << std::endl;
        std::cerr << "    Test Failed: " << e.what() << std::endl;
        std::cout << "\nFAILURE" << std::endl;
        return 1;
    }
    std::cout << "\nSUCCESS!" << std::endl;
    return 0;
}
