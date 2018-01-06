#include <cstdio>
#include <string>
#include <vector>
#include <cstdint>
#include <sys/time.h>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

static const int local_context_size = 80, inpaint_size = 3, mf_step = 5, mf_ksize = 5;
const double const_k = .002;

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
    int nRows = small.rows;
    int nCols = small.cols*channels;
    for(int i = 0; i < nRows; i++)
    {
        const uint8_t* s = small.ptr<uint8_t>(i);
        const uint8_t* b = big.ptr<uint8_t>(i);
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

    for(int i = 0; i < nRows-locs.back()[0]; i+=mf_step)
    {
        for (int j = 0; j < nCols-locs.back()[1]; j+=mf_step)
        {
            double context = ssd(source, match, locs, Vec2i(i, j), Vec3d(.81, .90, 1));
            double texture = ssd(source_grad, match_grad, locs, Vec2i(i, j), Vec3d(1, 1, 1));
            double t = context+texture;
            if(t < mi) mi=t, mt=Vec2i(i, j);
        }
    }
    return make_pair(mt, mi);
}

class GraphCut
{
private:
    struct rec1
    {
        int64_t x, y, c, n;
    };
    vector<rec1> l;
    vector<int64_t> a, b, d, s, dis;
    vector<bool> f;
    int n, rows, cols;
    void insert(int x, int y, int c)
    {
        l.emplace_back(rec1({x, y, c, b[x]}));
        b[x] = l.size()-1;
        l.emplace_back(rec1({y, x, 0, b[y]}));
        b[y] = l.size()-1;
    }
    int bfs()
    {
        int h=-1,t=0;d[0] = n;
        memset(a.data(), 0, a.size()*sizeof(int64_t));
        a[n] = 1;
        while(h != t)
        {
            h++, s[d[h]] = b[d[h]];
            for(int k=b[d[h]];k!=-1;k=l[k].n)
            {
                if((!a[l[k].y])&&(l[k].c))
                {
                    a[l[k].y] = a[d[h]]+1;
                    d[++t] = l[k].y;
                }
            }
        }
        return a[n+1] != 0;
    }
    int dfs(int x, int64_t flow)
    {
        if(x == n+1)return flow;
        int64_t p=0, k;
        for(k=s[x];k!=-1;k=l[k].n)
        {
            if((l[k].c)&&(a[l[k].y] == a[x]+1))
            {
                int64_t t = dfs(l[k].y, min(l[k].c, flow));
                flow-=t, p+=t;
                l[k].c-=t, l[k^1].c+=t;
                if(flow == 0)break;
            }
        }
        s[x] = k;
        return p;
    }
    int get(int x, int y)
    {
        if(x < 0 || y < 0 || x >= rows || y >= cols) return -1;
        return x*cols+y;
    }
    rec1 get(int x)
    {
        return rec1({x/cols, x%cols, 0, 0});
    }
public:
    GraphCut(const Mat &source, const Mat &match, const Mat &mask, const vector<Vec2i> &locs, const Vec2i &scene)
    {
        rows = source.rows;
        cols = source.cols;
        n = rows*cols;
        a.resize(n+2);
        b.resize(n+2);
        d.resize(n+2);
        s.resize(n+2);
        f.resize(n+2);
        dis.assign(n+2, INT32_MAX);

        int h=-1, t=0;
        for(int i=0;i<rows;i++)
        {
            const uint8_t* s = mask.ptr<uint8_t>(i);
            for(int j=0; j<cols;j++)if(s[j])
            {
                dis[get(i, j)]=0;
                d[t++] = get(i, j);
            }
        }
        while(h != t){
            h = h%n+1;
            int x = get(d[h]).x, y = get(d[h]).y;
            for(int dx=-1;dx<2;dx++)for(int dy=-1;dy<2;dy++)if(get(x+dx, y+dy)!=-1 && abs(dx)!=abs(dy))
            {
                if(dis[get(x+dx, y+dy)] > dis[get(x, y)]+1)
                {
                    dis[get(x+dx, y+dy)] = dis[get(x, y)]+1;
                    if(!f[get(x+dx, y+dy)])
                    {
                        d[t=t%n+1] = get(x+dx, y+dy);
                        f[get(x+dx, y+dy)] = true;
                    }
                }
            }
            f[d[h]] = false;
        }

        for(const auto &p : locs)
            f[get(p[0], p[1])] = true;
        Mat source_lab, match_lab, source_grad, match_grad;
        cvtColor(source, source_lab, cv::COLOR_RGB2Lab);
        cvtColor(match, match_lab, cv::COLOR_RGB2Lab);
        Mat ssd(rows, cols, CV_64F), ssd_grad;
        for(int i=0;i<min(rows,match.rows-scene[0]);i++)
        {
            const Vec3b* s = source_lab.ptr<Vec3b>(i);
            const Vec3b* m = match_lab.ptr<Vec3b>(i+scene[0]);
            double *d = ssd.ptr<double>(i);
            for(int j=0;j<min(cols,match.cols-scene[1]);j++)
            {
                auto diff = Vec3d(s[j]) - Vec3d(m[j+scene[1]]);
                d[j] = sqr(diff[0])+sqr(diff[1])+sqr(diff[2]);
            }
        }
        Laplacian(ssd, ssd_grad, ssd.depth());

        memset(b.data(), -1, b.size()*sizeof(int64_t));
        for(int x=0;x<rows;x++)
            for(int y=0;y<cols;y++)
            {
                const double* s = ssd_grad.ptr<double>(x);
                if(!dis[get(x, y)])
                    insert(n, get(x, y), INT32_MAX);
                else if(!f[get(x, y)])
                    insert(get(x, y), n+1, INT32_MAX);
                for(int dx=-1;dx<2;dx++)for(int dy=-1;dy<2;dy++)if(get(x+dx, y+dy)!=-1 && abs(dx)!=abs(dy))
                {
                    insert(get(x, y), get(x+dx, y+dy), pow(const_k*dis[get(x,y)], 3)+fabs(s[y]));
                }
            }
//        Mat tmp(rows, cols, CV_16U);
//        for(int i=0;i<rows;i++)
//        {
//            uint16_t* s = tmp.ptr<uint16_t>(i);
//            for(int j=0; j<cols;j++)if(s[j])
//            {
//                s[j]=dis[get(i, j)]*(1<<8);
//            }
//        }
//        imshow("tmp", ssd_grad);
//        waitKey(0);
    }

    int64_t cost()
    {
        int64_t ans = 0;
        while(bfs())
            ans+=dfs(n, INT32_MAX);
        return ans;
    }
    Mat cut()
    {
        Mat ret(rows, cols, CV_8U);
        for(int i=0;i<rows;i++)
        {
            uint8_t* r = ret.ptr<uint8_t>(i);
            for(int j=0;j<cols;j++)
            {
                if(a[get(i,j)]) r[j] = 255; else r[j] = 0;
            }
        }
        return ret;
    }

};

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
    
    Mat dilated_mask;
    dilate(mask, dilated_mask, getStructuringElement(MORPH_RECT, Size(local_context_size, local_context_size), Point(0, 0)));

    auto locs = get_local_context(mask, dilated_mask);
    
    auto scene_pair = find_scene(source, match, locs);
    auto scene = scene_pair.first;

    GraphCut graph_cut(source, match, mask, locs, scene);
    auto cost = graph_cut.cost();
    auto cut = graph_cut.cut();

    int minx = INT_MAX, miny = INT_MAX, maxx = INT_MIN, maxy = INT_MIN;
    for(int i=0;i<cut.rows;i++)
    {
        for(int j=0;j<cut.cols;j++)
        {
            uint8_t* r = cut.ptr<uint8_t>(i);
            if(r[j] == 255)
            {
                minx = min(minx,i);
                maxx = max(maxx,i);
                miny = min(miny,j);
                maxy = max(maxy,j);
            }
        }
    }

    Mat destination = match.clone(), possion, result, dilated_cut, eroded_cut, inpaint_mask = mask.clone();
    dilate(cut, dilated_cut, getStructuringElement(MORPH_RECT, Size(inpaint_size, inpaint_size), Point(0, 0)));
    erode(cut, eroded_cut, getStructuringElement(MORPH_RECT, Size(inpaint_size, inpaint_size), Point(0, 0)));

    for(int i=0;i<mask.rows;i++)
    {
        for(int j=0;j<mask.cols;j++)
        {
            const uint8_t* r = mask.ptr<uint8_t>(i);
            const uint8_t* dc = dilated_cut.ptr<uint8_t>(i);
            const uint8_t* ec = eroded_cut.ptr<uint8_t>(i);
            uint8_t* im = inpaint_mask.ptr<uint8_t>(i);
            const Vec3b* s = source.ptr<Vec3b>(i);
            Vec3b* d = destination.ptr<Vec3b>(i+scene[0]);
            if(!r[j]) d[j+scene[1]] = s[j];
            if(dc[j] && !ec[j]) im[j] = 255; else im[j] = 0;
        }
    }

    seamlessClone(match(Range(scene[0], scene[0]+source.rows), Range(scene[1], scene[1]+source.cols)), destination, cut.clone(), Point(scene[1]+(maxy+miny)/2, scene[0]+(maxx+minx)/2), possion, NORMAL_CLONE);

    inpaint(possion(Range(scene[0], scene[0]+source.rows), Range(scene[1], scene[1]+source.cols)), inpaint_mask, result, 2*inpaint_size+1, INPAINT_NS);

    gettimeofday(&end, NULL);
    printf("time %.6lf\n", double(end.tv_sec-start.tv_sec)+1e-6*double(end.tv_usec-start.tv_usec));
    printf("%lf\n", cost+scene_pair.second);
    imshow("source", source);
    imshow("match", match);
    imshow("mask", mask);
    imshow("dilated_mask", dilated_mask);
    imshow("cut", cut);
    imshow("inpaint_mask", inpaint_mask);
    imshow("destination", destination);
    imshow("possion", possion(Range(scene[0], scene[0]+source.rows), Range(scene[1], scene[1]+source.cols)));
    imshow("result", result);
    imwrite("dilated_mask.jpg", dilated_mask);
    imwrite("cut.jpg", cut);
    imwrite("inpaint_mask.jpg", inpaint_mask);
    imwrite("destination.jpg", destination);
    imwrite("possion.jpg", possion(Range(scene[0], scene[0]+source.rows), Range(scene[1], scene[1]+source.cols)));
    imwrite("result.jpg", result);
    waitKey(0);
    return 0;
}
