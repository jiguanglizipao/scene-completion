#include <cstdio>
#include <string>
#include <vector>
#include <cstdint>
#include <fstream>
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
int main(int argc, char** argv)
{
    struct timeval start, end;
    gettimeofday(&start, NULL);
    if (argc != 3)
    {
        printf("usage: %s <images List> <result File>\n", argv[0]);
        return -1;
    }

    vector<string> files;
    ifstream fi(argv[1]); 
    for(string line; getline(fi, line);) files.push_back(line);
    fi.close();
    vector<vector<double>> datas(files.size());
    #pragma omp parallel for
    for(size_t i=0;i<files.size();i++)
    {
        if(!omp_get_thread_num()) printf("%.2lf%%\n", 100.*i/files.size());
        Mat image = imread(files[i], cv::IMREAD_COLOR);
        datas[i] = gist(image);
    }
    printf("100.00%%\n");
    FILE *fo = fopen(argv[2], "w");
    for(size_t i=0;i<files.size();i++)
    {
        fwrite(datas[i].data(), sizeof(double), datas[i].size(), fo);
    }
    fclose(fo);
    gettimeofday(&end, NULL);
    printf("time %.6lf\n", double(end.tv_sec-start.tv_sec)+1e-6*double(end.tv_usec-start.tv_usec));
    return 0;
}
