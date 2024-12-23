//
// Created by nmillerns on 20/12/24.
//
#include <array>
#include <cstddef>
#include <complex>
#include <experimental/random>
#include <vector>

#ifndef SELFORGANIZINGMAPS_MAP_H
#define SELFORGANIZINGMAPS_MAP_H

template <size_t DIMENSIONS>
class Neuron {
public:
    Neuron() {}
    const double* weightVector() const  { return m_weight_vector; }
    double* weightVector() { return m_weight_vector; }
    static inline size_t length() { return DIMENSIONS; }

private:
    double m_weight_vector[DIMENSIONS];  // The position of the node in the input space
    std::vector<Neuron*> m_neighbors;
};


template<size_t DIMENSIONS>
class GridSelfOrganizingMap {
public:
    GridSelfOrganizingMap(size_t width, size_t height)
        : m_width(width)
        , m_height(height)
        , m_neurons(m_width * m_height) {
    }

    inline size_t width() const { return m_width; }
    inline size_t height() const { return m_height; }
    const Neuron<DIMENSIONS>& getNeuron(size_t row, size_t col) const { return m_neurons.at(rowColToIndex(row, col)); }
    Neuron<DIMENSIONS>& getNeuron(size_t row, size_t col) { return m_neurons.at(rowColToIndex(row, col)); }

    class iterator : public std::vector<Neuron<DIMENSIONS>>::iterator {
    public:
        iterator(typename std::vector<Neuron<DIMENSIONS>>::iterator inner,
                 typename std::vector<Neuron<DIMENSIONS>>::iterator begin,
                 size_t grid_width)
            : std::vector<Neuron<DIMENSIONS>>::iterator(inner)
            , m_grid_width(grid_width)
            , m_begin(begin) {}
        size_t row() { return (*this - m_begin) / m_grid_width; };
        size_t col() { return (*this - m_begin) % m_grid_width; }

    private:
        size_t m_grid_width;
        typename std::vector<Neuron<DIMENSIONS>>::iterator m_begin;
    };

    iterator begin() { return iterator(m_neurons.begin(), m_neurons.begin(), m_width); }
    iterator end() { return iterator(m_neurons.end(), m_neurons.begin(), m_width); }

    iterator BMU(const double vec[DIMENSIONS]) {
        iterator best = begin();
        double mind = norm2(best->weightVector(), vec);
        for (iterator i = begin(); i != end(); ++i) {
            const double d = norm2(i->weightVector(), vec);
            if (d < mind) {
                best = i;
                mind = d;
            }
        }
        return best;
    }

    static double neighbourhoodDist(iterator a, iterator b) {
        return std::pow(0.5, std::abs<double>(1. * a.row() - b.row()) + std::abs<double>(1. * a.col() - b.col()));
    }

    class NullVisualizer {
    public:
        void showStep(const size_t s, const size_t t) {}
    };

    template<class VIS=NullVisualizer>
    void fitData(const std::vector<std::array<double, DIMENSIONS>>& input_data, VIS* vis = nullptr) {
        const size_t L = 250;
        for (size_t s = 0; s < L; ++s) {
            // learning rate schedule alpha(s)
            double a_s = (L * 1. - s) / (1. * L);

            // 1. Randomly pick input vector
            size_t t = std::experimental::randint<size_t>(0, input_data.size() - 1);
            auto& Dt = input_data.at(t);
            if (vis) { vis->showStep(s, t); }  // visualize the current step to animate if vis is available

            // 2. Find the node in the map closest to the input vector. This node is the Best Matching Unit (BMU)
            auto u = BMU(Dt.data());

            // 3. For each node v update its vector by pulling closer to the input vector
            for (auto v = begin(); v != end(); ++v) {
                double* Wv = v->weightVector();
                // Neighbourhood function
                double Theta_uvs = neighbourhoodDist(u, v);
                double diff0 = Dt[0] - Wv[0];
                double diff1 = Dt[1] - Wv[1];
                Wv[0] = Wv[0] + Theta_uvs * a_s * diff0;
                Wv[1] = Wv[1] + Theta_uvs * a_s * diff1;
            }
        }
    }

private:
    static double norm2(const double a[DIMENSIONS], const double b[DIMENSIONS]) {
        double n = 0.;
        for (size_t i = 0; i < DIMENSIONS; ++i) {
            n += (a[i] - b[i]) * (a[i] - b[i]);
        }
        return n;
    }

    inline size_t rowColToIndex(size_t row, size_t col) const { return row * m_width + col; }
    size_t m_width;
    size_t m_height;
    std::vector<Neuron<DIMENSIONS>> m_neurons;
};


template<size_t DIMENSIONS>
class LinearSelfOrganizingMap : public GridSelfOrganizingMap<DIMENSIONS> {
public:
    LinearSelfOrganizingMap(size_t length) : GridSelfOrganizingMap<DIMENSIONS>(length, 1) {}
    inline size_t length() const { return GridSelfOrganizingMap<DIMENSIONS>::width(); }
    const Neuron<DIMENSIONS>& getNeuron(size_t index) const { return GridSelfOrganizingMap<DIMENSIONS>::getNeuron(0, index); }
    Neuron<DIMENSIONS>& getNeuron(size_t index) { return GridSelfOrganizingMap<DIMENSIONS>::getNeuron(0, index); }
};


typedef GridSelfOrganizingMap<2> GridSelfOrganizingMap2d;
typedef LinearSelfOrganizingMap<2> LinearSelfOrganizingMap2d;

#endif //SELFORGANIZINGMAPS_MAP_H
