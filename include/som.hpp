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

    /**
     * Row wise iterator of neurons in the grid
     * Has non-const access to the network
     */
    class iterator : public std::vector<Neuron<DIMENSIONS>>::iterator {
    public:
        iterator(typename std::vector<Neuron<DIMENSIONS>>::iterator inner,
                 typename std::vector<Neuron<DIMENSIONS>>::iterator grid_begin,
                 size_t grid_width)
            : std::vector<Neuron<DIMENSIONS>>::iterator(inner)
            , m_grid_width(grid_width)
            , m_grid_begin(grid_begin) {}
        size_t row() const { return (*this - m_grid_begin) / m_grid_width; };
        size_t col() const { return (*this - m_grid_begin) % m_grid_width; }

    private:
        size_t m_grid_width;
        typename std::vector<Neuron<DIMENSIONS>>::iterator m_grid_begin;
    };

    iterator begin() { return iterator(m_neurons.begin(), m_neurons.begin(), m_width); }
    iterator end() { return iterator(m_neurons.end(), m_neurons.begin(), m_width); }

private:
    class NullVisualizer;

public:
    template<class VIS=NullVisualizer>
    void fitData(const std::vector<std::array<double, DIMENSIONS>>& input_data, VIS* vis = nullptr) {
        const size_t L = 250;
        for (size_t s = 0; s < L; ++s) {
            // learning rate schedule alpha(s)
            const double a_s = (L * 1. - s) / (1. * L);

            // 1. Randomly pick input vector
            const size_t t = std::experimental::randint<size_t>(0, input_data.size() - 1);
            const auto& Dt = input_data.at(t);
            if (vis) { vis->showStep(s, t); }  // visualize the current step to animate if vis is available

            // 2. Find the node in the map closest to the input vector. This node is the Best Matching Unit (BMU)
            auto u = BMU(Dt.data());

            // 3. For each node v update its vector by pulling closer to the input vector
            for (auto v = begin(); v != end(); ++v) {
                double* Wv = v->weightVector();
                // Neighbourhood function
                const double Theta_uvs = neighbourhoodDist(u, v);
                for (size_t dim = 0; dim < DIMENSIONS; ++dim) {
                    Wv[dim] = Wv[dim] + Theta_uvs * a_s * (Dt[dim] - Wv[dim]);
                }
            }
        }
    }

    iterator BMU(const double vec[DIMENSIONS]) {
        iterator bmu = begin();
        double min_dist = euclidDistSq(bmu->weightVector(), vec);
        for (iterator i = begin(); i != end(); ++i) {
            const double d = euclidDistSq(i->weightVector(), vec);
            if (d < min_dist) {
                bmu = i;
                min_dist = d;
            }
        }
        return bmu;
    }

    static double neighbourhoodDist(iterator a, iterator b) {
        return std::pow(0.5, std::abs<double>(1. * a.row() - b.row()) + std::abs<double>(1. * a.col() - b.col()));
    }

private:
    struct NullVisualizer {
        void showStep(const size_t s, const size_t t) {}
    };

    static double euclidDistSq(const double v[DIMENSIONS], const double w[DIMENSIONS]) {
        double dist_sq = 0.;
        for (size_t dim = 0; dim < DIMENSIONS; ++dim) {
            dist_sq += (v[dim] - w[dim]) * (v[dim] - w[dim]);
        }
        return dist_sq;
    }

    inline size_t rowColToIndex(size_t row, size_t col) const { return row * m_width + col; }
    size_t m_width;
    size_t m_height;
    std::vector<Neuron<DIMENSIONS>> m_neurons;
};


template<size_t DIMENSIONS>
class LinearSelfOrganizingMap : public GridSelfOrganizingMap<DIMENSIONS> {
public:
    explicit LinearSelfOrganizingMap(size_t length) : GridSelfOrganizingMap<DIMENSIONS>(length, 1) {}
    inline size_t length() const { return GridSelfOrganizingMap<DIMENSIONS>::width(); }
    const Neuron<DIMENSIONS>& getNeuron(size_t index) const { return GridSelfOrganizingMap<DIMENSIONS>::getNeuron(0, index); }
    Neuron<DIMENSIONS>& getNeuron(size_t index) { return GridSelfOrganizingMap<DIMENSIONS>::getNeuron(0, index); }
};


typedef GridSelfOrganizingMap<2> GridSelfOrganizingMap2d;
typedef LinearSelfOrganizingMap<2> LinearSelfOrganizingMap2d;

#endif //SELFORGANIZINGMAPS_MAP_H
