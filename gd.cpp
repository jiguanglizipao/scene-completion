#include <cstdio>
#include <string>
#include <vector>
#include <cstdint>
#include <fstream>
#include <sys/time.h>
#include <opencv2/opencv.hpp>
#include <omp.h>
#include <mpi.h>
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

    MPI::Init();
    size_t nproc = MPI::COMM_WORLD.Get_size();
    size_t myid = MPI::COMM_WORLD.Get_rank();
    size_t mylen = (files.size()+nproc-1)/nproc;

    #pragma omp parallel for
    for(size_t i=myid*mylen;i<min((myid+1)*mylen, files.size());i++)
    {
        if(!omp_get_thread_num()) printf("myid=%lu\t%.2lf%%\n", myid, 100.*(i-myid*mylen)/(min((myid+1)*mylen, files.size())-myid*mylen));
        Mat image = imread(files[i], cv::IMREAD_COLOR);
        datas[i] = gist(image);
        for(size_t i=myid*mylen;i<min((myid+1)*mylen, files.size());i++)
        {
            MPI::COMM_WORLD.Send(datas[i].data(), datas[i].size(), MPI::DOUBLE, 0, i);
        }
    }
    printf("myid=%lu\t100.00%%\n", myid);
    if(myid == 0)
    {
        FILE *fo = fopen(argv[2], "w");
        for(size_t id=0;id<nproc;id++)
            for(size_t i=id*mylen;i<min((id+1)*mylen, files.size());i++)
            {
                if(id != 0)
                {
                    datas[i].resize(datas[0].size());
                    MPI::COMM_WORLD.Recv(datas[i].data(), datas[i].size(), MPI::DOUBLE, id, i);
                }
                fwrite(datas[i].data(), sizeof(double), datas[i].size(), fo);
            }
        fclose(fo);
        gettimeofday(&end, NULL);
        printf("time %.6lf\n", double(end.tv_sec-start.tv_sec)+1e-6*double(end.tv_usec-start.tv_usec));
    }
    else
    {
        for(size_t i=myid*mylen;i<min((myid+1)*mylen, files.size());i++)
        {
            MPI::COMM_WORLD.Send(datas[i].data(), datas[i].size(), MPI::DOUBLE, 0, i);
        }
    }
    MPI::Finalize();
    return 0;
}
