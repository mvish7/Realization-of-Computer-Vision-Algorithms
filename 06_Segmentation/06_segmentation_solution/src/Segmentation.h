#ifndef SEGMENTATION_H
#define SEGMENTATION_H

#include <opencv2/core/core.hpp>

class Segmentation
{
public:
    Segmentation();
    ~Segmentation();

    // Template Matching
    void cutAndSave(const cv::Mat &input, cv::Point origin, cv::Size size, const cv::string &filename);
    void crossCorrelate(const cv::Mat &input, const cv::Mat &templ, cv::Mat &output);
    cv::Point findMaximum(const cv::Mat &input);
    void drawRect(const cv::Mat &input, cv::Point origin, cv::Size size, cv::Mat &output);

    // Hough Transformation
    void houghTransform(const cv::Mat &input, float phiStep, cv::Mat &output);
    void scaleHoughImage(const cv::Mat &input, cv::Mat &output);
    cv::Mat findMaxima(const cv::Mat &input, int n);
    void drawLines(const cv::Mat &input, cv::Mat lines, float phiStep, cv::Mat &output);

private:

};

#endif /* SEGMENTATION_H */
