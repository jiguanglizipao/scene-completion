#include <cstdio>
#include <string>
#include <vector>
#include <cstdint>
#include <fstream>
#include <algorithm>
#include <sys/time.h>
#include <opencv2/opencv.hpp>
#include <omp.h>
using namespace cv;
using namespace std;

static vector<double> gist(const Mat &image)
{
    Mat image_lab, image_float, gaborImage;
    cvtColor(image, image_lab, COLOR_BGR2Lab);
    image_lab.convertTo(image_float, CV_64F);
    vector<double> gist;
    for(int i=1;i<=5;i++)
    {
        for(int j=0;j<6;j++)
        {
            Mat kernel = getGaborKernel(Size(11, 11), 3, (1e-5+1./6*j)*CV_PI, 0.05*i, 0.5);
            const int size = 4;
            Mat gabor, output(size, size, CV_64FC3);
            filter2D(image_float, gabor, CV_64F, kernel);
            int divx = gabor.rows/size, divy = gabor.cols/size;
            for(int x=0;x<gabor.rows;x++)
            {
                const Vec3d* g = gabor.ptr<Vec3d>(x);
                for(int y=0;y<gabor.cols;y++)
                {
                    if(x/divx>=size || y/divy>=size)continue;
                    if(x%divx==0 && y%divy==0) output.at<Vec3d>(x/divx, y/divy) = Vec3d(0,0,0);
                    output.at<Vec3d>(x/divx, y/divy) += 1./(divx*divy)*g[y];
                }
            }
            for(int x=0;x<output.rows;x++)
            {
                const Vec3d* o = output.ptr<Vec3d>(x);
                for(int y=0;y<output.cols;y++)
                {
                    for(int k=0;k<3;k++) gist.push_back(o[y][k]);
                }
            }
//            imshow("gabor", output);
//            waitKey(0);
        }
    }
    return gist;
}

template <typename T> static T sqr(T x){return x*x;}
static double ssd(const vector<double> &a, const vector<double> &b)
{
    double sum = 0;
    for(size_t i=0;i<a.size();i++) sum += sqr(a[i]-b[i]);
    return sum;
}

extern double local_context_matching(const Mat &source_in, const Mat &match_in, const Mat &mask_in, Mat & result_out);

int main(int argc, char** argv)
{
    struct timeval start, end;
    gettimeofday(&start, NULL);
    if (argc != 5)
    {
        printf("usage: %s <images List> <gists File> <source Image> <mask Image>\n", argv[0]);
        return -1;
    }

    vector<string> files;
    ifstream fi(argv[1]); 
    for(string line; getline(fi, line);) files.push_back(line);
    fi.close();

    Mat source = imread(argv[3], cv::IMREAD_COLOR);
    Mat mask = 255-imread(argv[4], cv::IMREAD_GRAYSCALE);
    vector<double> source_gist = gist(source);

    vector<vector<double>> gists(files.size());
    vector<pair<double, string>> gist_ssd;
    FILE *fip = fopen(argv[2], "r");
    for(size_t i=0;i<files.size();i++)
    {
        gists[i].resize(source_gist.size());
        if(fread(gists[i].data(), sizeof(double), gists[i].size(), fip) != source_gist.size()) continue;
        gist_ssd.emplace_back(make_pair(ssd(source_gist, gists[i]), files[i]));
    }
    fclose(fip);
    sort(gist_ssd.begin(), gist_ssd.end());
//    for(size_t i=0;i<gist_ssd.size();i++)
//    {
//        printf("%s %lf\n", gist_ssd[i].second.c_str(), gist_ssd[i].first);
//    }

    size_t gist_num = min(gist_ssd.size(), 200lu), lcm_num = min(gist_num, 20lu);
    vector<pair<pair<double, string>, size_t>> lcm_loss(gist_num);
    vector<Mat> lcm_results(gist_num);
    #pragma omp parallel for schedule(dynamic)
    for(size_t i=0;i<gist_num;i++)
    {
        if(!omp_get_thread_num()) printf("%.2lf%%\n", 100.*i/gist_num);
        Mat match = imread(gist_ssd[i].second, cv::IMREAD_COLOR);
        double loss = local_context_matching(source, match, mask, lcm_results[i]);
        lcm_loss[i] = make_pair(make_pair(loss, gist_ssd[i].second), i);
    }
    printf("100.00%%\n");

    sort(lcm_loss.begin(), lcm_loss.end());
    for(size_t i=0;i<lcm_num;i++)
    {
        char tmp[4096];
        sprintf(tmp, "result_%03lu.jpg", i);
        printf("%s %s %lf\n", lcm_loss[i].first.second.c_str(), tmp, lcm_loss[i].first.first);
        imwrite(tmp, lcm_results[lcm_loss[i].second]);
    }

    gettimeofday(&end, NULL);
    printf("time %.6lf\n", double(end.tv_sec-start.tv_sec)+1e-6*double(end.tv_usec-start.tv_usec));
    return 0;
}
