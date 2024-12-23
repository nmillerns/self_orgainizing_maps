//
// Created by nmiller on 23/12/24.
//

#ifndef SELFORGANIZINGMAPS_VIS_HPP
#define SELFORGANIZINGMAPS_VIS_HPP
#include <iomanip>
#include <opencv2/highgui.hpp>
#include <opencv2/shape.hpp>

#include "som.hpp"

typedef cv::Point3_<uint8_t> Pixel;
struct RGBRef {
    RGBRef(Pixel & pt) : r(pt.x), g(pt.y), b(pt.z) {}
    void assign(uint8_t _r, uint8_t _g, uint8_t _b) {
        r = _r; g = _g; b = _b;
    }
    uint8_t& r;
    uint8_t& g;
    uint8_t& b;
    bool isWhite() { return r == 255 || g == 255 || b == 255; }
};


class FileAnimSelfOrganizingMapVisualizer {
public:
    FileAnimSelfOrganizingMapVisualizer(LinearSelfOrganizingMap2d* linear_som, cv::Mat bg, std::vector<std::array<double, 2>>* data)
            : m_linear_som(linear_som)
            , m_width(bg.size().width)
            , m_height(bg.size().height)
            , m_bg(bg)
            , m_output(m_height, m_width, CV_8UC3)
            , m_data(data)
            , m_path_base("anim_") {
        m_bg.copyTo(m_output);
        std::cout << "Animating to " << m_path_base << "*.png" << std::endl;
    }

    void drawSOM() {
        for (size_t i = 0; i + 1 < m_linear_som->length(); ++i) {
            const auto start = neuronWeightVectorToPoint(m_linear_som->getNeuron(i));
            const auto end = neuronWeightVectorToPoint(m_linear_som->getNeuron(i + 1));
            cv::line(m_output, start, end, cv::Scalar(255, 0, 0));
        }
        for (const auto& neuron: *m_linear_som) {
            cv::circle(m_output, neuronWeightVectorToPoint(neuron), 3, cv::Scalar(0, 0, 255));
        }
    }

    void clear() {
        m_bg.copyTo(m_output);
    }

    void showStep(size_t s, size_t t) {
        clear();
        const auto Dt = m_data->at(t);
        cv::circle(m_output, cv::Point(Dt[0] * m_width, Dt[1] * m_height), 2, cv::Scalar(0, 255, 0), 4);
        drawSOM();
        // Update animation frame
        cv::imwrite((std::stringstream() <<
                                         m_path_base << std::setw(9) << std::setfill('0') << s << ".png").str(),
                    m_output);
    }

    const cv::Mat& output() { return m_output; }

private:
    cv::Point neuronWeightVectorToPoint(const Neuron<2>& neuron) { return cv::Point(neuron.weightVector()[0] * m_width,
                                                                                    neuron.weightVector()[1] * m_height); }

    LinearSelfOrganizingMap2d* m_linear_som;
    const size_t m_width;
    const size_t m_height;
    cv::Mat m_bg;
    cv::Mat m_output;
    std::vector<std::array<double, 2>>* m_data;
    std::string m_path_base;
};

#endif //SELFORGANIZINGMAPS_VIS_HPP
