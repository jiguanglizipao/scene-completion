#include <cstdio>
#include <string>
#include <vector>
#include <cstdint>
#include <sys/time.h>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

const int local_context_size = 80;

template <typename T> static T sqr(T x){return x*x;}
static double ssd(const Mat &a, const Mat &b, const vector<Vec2i> &locs, const Vec2i &boff, Vec3b scale)
{
    double sum = 0;
    for(const auto &p : locs)
    {
        auto diff = Vec3d(a.at<Vec3b>(p[0], p[1])) - Vec3d(b.at<Vec3b>(p[0]+boff[0], p[1]+boff[1]));
        sum += scale[0]*sqr(diff[0])+scale[1]*sqr(diff[1])+scale[2]*sqr(diff[2]);
    }
    return sum;
}

static vector<Vec2i> get_local_context(const Mat& small, const Mat& big)
{
    vector<Vec2i> locs;
    int channels = small.channels();
    int nRows = small.rows ;
    int nCols = small.cols* channels;
    for(int i = 0; i < nRows; i++)
    {
        const uchar* s = small.ptr<uchar>(i);
        const uchar* b = big.ptr<uchar>(i);
        for (int j = 0; j < nCols; j++)
        {
            if(!s[j] && b[j])
                locs.emplace_back(Vec2i(i, j));
        }
    }
    return locs;
}

static pair<Vec2i, double> find_scene(const Mat &source, const Mat &match, const vector<Vec2i> locs)
{
    const int step = 1;
    const int mf_ksize = 5;
    int nRows = match.rows;
    int nCols = match.cols;
    Vec2i mt = Vec2i(0, 0);
    uint64_t mi = UINT64_MAX;

    Mat source_lab, match_lab, source_grad, match_grad;
    Laplacian(source, source_grad, source.depth());
    Laplacian(match, match_grad, match.depth());
    medianBlur(source_grad, source_grad, mf_ksize);
    medianBlur(match_grad, match_grad, mf_ksize);
    cvtColor(source, source_lab, cv::COLOR_RGB2Lab);
    cvtColor(match, match_lab, cv::COLOR_RGB2Lab);

    for(int i = 0; i < nRows-locs.back()[0]; i+=step)
    {
        for (int j = 0; j < nCols-locs.back()[1]; j+=step)
        {
            double context = ssd(source, match, locs, Vec2i(i, j), Vec3d(.81, .90, 1));
            double texture = ssd(source_grad, match_grad, locs, Vec2i(i, j), Vec3d(1, 1, 1));
            double t = context+texture;
            if(t < mi) mi=t, mt=Vec2i(i, j);
        }
    }
    return make_pair(mt, mi);
}

int main(int argc, char** argv)
{
    struct timeval start, end;
    gettimeofday(&start, NULL);
    if (argc != 4)
    {
        printf("usage: %s <source Image> <match Image> <mask Image>\n", argv[0]);
        return -1;
    }

    Mat source = imread(argv[1], cv::IMREAD_COLOR);
    Mat match = imread(argv[2], cv::IMREAD_COLOR);
    Mat mask = 255-imread(argv[3], cv::IMREAD_GRAYSCALE);
    threshold(mask, mask, 128, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    Mat dialation_mask;
    dilate(mask, dialation_mask, getStructuringElement(MORPH_RECT, Size(local_context_size, local_context_size), Point(0, 0)));
    auto locs = get_local_context(mask, dialation_mask);
    auto scene_pair = find_scene(source, match, locs);
    auto scene = scene_pair.first;
    for(const auto &p : locs)
    {
        source.at<Vec3b>(p[0], p[1]) = Vec3b(0,0,0);
        match.at<Vec3b>(p[0]+scene[0], p[1]+scene[1]) = Vec3b(0,0,0);
    }
    gettimeofday(&end, NULL);
    printf("time %.6lf\n", double(end.tv_sec-start.tv_sec)+1e-6*double(end.tv_usec-start.tv_usec));
    imshow("source", source);
    imshow("match", match);
    imshow("mask", mask);
    imshow("dialation_mask", dialation_mask);
    waitKey(0);
    return 0;
}
