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

struct maskStruct
{
    Mat img0, img1, res1, final;
    Point point;
    int drag, numpts, var, flag, flag1;
    int minx,miny,maxx,maxy,lenx,leny;
    Point pts[1024];
    maskStruct(const Mat &src)
    {
        drag = 0;
        numpts = 1024;
        var = 0;
        flag = 0;
        flag1 = 0;
        minx = INT_MAX; miny = INT_MAX; maxx = INT_MIN; maxy = INT_MIN;
        img0 = src;
        res1 = Mat::zeros(img0.size(),CV_8UC1);
        final = Mat::zeros(img0.size(),CV_8UC3);
    }
};

static void maskmouseHandler(int event, int x, int y, int, void *void_ms)
{
    maskStruct *ms = (maskStruct*)void_ms;
    if (event == EVENT_LBUTTONDOWN && !ms->drag)
    {
        if(ms->flag1 == 0)
        {
            if(ms->var==0)
                ms->img1 = ms->img0.clone();
            ms->point = Point(x, y);
            circle(ms->img1,ms->point,2,Scalar(0, 0, 255),-1, 8, 0);
            ms->pts[ms->var] = ms->point;
            ms->var++;
            ms->drag  = 1;
            if(ms->var>1)
                line(ms->img1,ms->pts[ms->var-2], ms->point, Scalar(0, 0, 255), 2, 8, 0);

            imshow("Source", ms->img1);
        }
    }

    if (event == EVENT_LBUTTONUP && ms->drag)
    {
        imshow("Source", ms->img1);

        ms->drag = 0;
    }
    if (event == EVENT_RBUTTONDOWN)
    {
        ms->flag1 = 1;
        ms->img1 = ms->img0.clone();
        for(int i = ms->var; i < ms->numpts ; i++)
            ms->pts[i] = ms->point;

        if(ms->var!=0)
        {
            const Point* pts3[1] = {&ms->pts[0]};
            polylines( ms->img1, pts3, &ms->numpts,1, 1, Scalar(0,0,0), 2, 8, 0);
        }

        for(int i=0;i<ms->var;i++)
        {
            ms->minx = min(ms->minx,ms->pts[i].x);
            ms->maxx = max(ms->maxx,ms->pts[i].x);
            ms->miny = min(ms->miny,ms->pts[i].y);
            ms->maxy = max(ms->maxy,ms->pts[i].y);
        }
        ms->lenx = ms->maxx - ms->minx;
        ms->leny = ms->maxy - ms->miny;

        imshow("Source", ms->img1);
    }

    if (event == EVENT_RBUTTONUP)
    {
        ms->flag = ms->var;

        ms->final = Mat::zeros(ms->img0.size(),CV_8UC3);
        ms->res1 = Mat::zeros(ms->img0.size(),CV_8UC1);
        const Point* pts4[1] = {&ms->pts[0]};

        fillPoly(ms->res1, pts4,&ms->numpts, 1, Scalar(255, 255, 255), 8, 0);
        bitwise_and(ms->img0, ms->img0, ms->final,ms->res1);
        imwrite("mask.png",ms->res1);
        imshow("Source", ms->img1);
        destroyWindow("Source");

    }
    if (event == EVENT_MBUTTONDOWN)
    {
        for(int i = 0; i < ms->numpts ; i++)
        {
            ms->pts[i].x=0;
            ms->pts[i].y=0;
        }
        ms->var = 0;
        ms->flag1 = 0;
        ms->minx = INT_MAX; ms->miny = INT_MAX; ms->maxx = INT_MIN; ms->maxy = INT_MIN;
        imshow("Source", ms->img0);
        ms->drag = 0;
    }
}

int main(int argc, char** argv)
{
    struct timeval start, end;
    gettimeofday(&start, NULL);
    Mat source, mask;
    if (argc == 5)
    {
        source = imread(argv[3], cv::IMREAD_COLOR);
        mask = 255-imread(argv[4], cv::IMREAD_GRAYSCALE);
    }
    else if (argc == 4)
    {
        source = imread(argv[3], cv::IMREAD_COLOR);
        maskStruct ms(source);
        namedWindow("Source", WINDOW_AUTOSIZE);
        setMouseCallback("Source", maskmouseHandler, &ms);
        imshow("Source", ms.img0);
        waitKey(0);
        mask = ms.res1;
    }
    else
    {
        printf("usage: %s <images List> <gists File> <source Image> [mask Image]\n", argv[0]);
        return EXIT_FAILURE;
    }

    vector<string> files;
    ifstream fi(argv[1]); 
    for(string line; getline(fi, line);) files.push_back(line);
    fi.close();

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
    return EXIT_SUCCESS;
}
