 //
// Created by nmillerns on 20/12/24.
//
#include <iostream>
#include <iomanip>
#include "vis.hpp"
#include "som.hpp"


/**
 * Extract coordinates of points (nonwhite) on a (white) scatterplot image
 */
std::vector<std::array<double, 2>> dataFromScatterPlotImage(cv::Mat* img) {
    const double width = img->size().width;
    const double height = img->size().height;

    std::vector<std::array<double, 2>> data;
    for (size_t row = 0; row < height; ++row) {
        for (size_t col = 0; col < width; ++col) {
            if (!RGBRef(img->at<Pixel>(row, col)).isWhite()) {
                std::array<double, 2> vec = {col / width, row / height};
                data.emplace_back(vec);
            }

        }
    }
    return data;
}

int main(int argc, char** argv) {
    if (argc != 2 || std::string(argv[1]) == "--help") {
        std::cout << "Usage: " << argv[0] << " (scatter_img.png)" << std::endl;
        return 1;
    }
    std::string img_path = argv[1];
    std::cout << "Loading " << img_path << std::endl;
    cv::Mat img = cv::imread(img_path);

    LinearSelfOrganizingMap2d som(20);
    std::cout << "Initializing simple linear SOM of length " << som.length() << " uniformly along [0,1],0.5" << std::endl;
    for (size_t pos = 0; pos < som.length(); ++pos) {
        som.getNeuron(pos).weightVector()[0] = pos / 19.;
        som.getNeuron(pos).weightVector()[1] = 0.5;
    }

    auto data = dataFromScatterPlotImage(&img);
    FileAnimSelfOrganizingMapVisualizer vis(&som, img, &data);
    std::cout << "Fitting to " << data.size() << " data points..." << std::endl;
    som.fitData(data, &vis);

    std::string final_output = "result.png";
    std::cout << "Showing final result at " << final_output << std::endl;
    vis.clear();
    vis.drawSOM();
    cv::imwrite(final_output, vis.output());

    return 0;
}

