#include <cufft.h>
#include <cutil_math.h>

#include <FreeImagePlus.h>

#include <vector>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <fstream>

#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/timer.hpp>

static __global__ void PointwiseMul(cufftComplex*, const cufftComplex*, int);
static __global__ void Add(cufftComplex*, const cufftComplex, int);
static __global__ void Zero(int, float *);

static __global__ void FormatImage(const unsigned char *, float3 *, int, bool);
static __global__ void ComputeHistograms(const float3 *, int, int, int, int, int, int, float *);
static __global__ void ComputeEnergy(const float *, int, int, float *);
static __global__ void ComputeFeatures(const float *, const float *, int, int, int, int, float *);
static __global__ void PadFeatures(const float *, int, int, int, int, cufftReal *);

#define cudaSafeCall(x) _cudaSafeCall((x), __LINE__)
#define cufftSafeCall(x) _cufftSafeCall((x), __LINE__)

void _cudaSafeCall (cudaError_t error, int line)
{
    if (cudaSuccess != error)
    {
        printf("%d: %s\n", line, cudaGetErrorString(cudaGetLastError()));
        exit(error);
    }
}

void _cufftSafeCall (cufftResult error, int line)
{
    if (CUFFT_SUCCESS != error)
    {
        const char * msg;
        switch (error)
        {
        case CUFFT_INVALID_PLAN:
            msg = "CUFFT_INVALID_PLAN";
            break;
        case CUFFT_INVALID_VALUE:
            msg = "CUFFT_INVALID_VALUE";
            break;
        case CUFFT_INTERNAL_ERROR:
            msg = "CUFFT_INTERNAL_ERROR";
            break;
        case CUFFT_EXEC_FAILED:
            msg = "CUFFT_EXEC_FAILED";
            break;
        case CUFFT_SETUP_FAILED:
            msg = "CUFFT_SETUP_FAILED";
            break;
        case CUFFT_UNALIGNED_DATA:
            msg = "CUFFT_UNALIGNED_DATA";
            break;
        default:
            msg = "unknown";
            break;
        }
        printf("%d: %s\n", line, msg);
        exit(error);
    }
}

struct SVM
{
    uint16_t width;
    uint16_t height;
    uint16_t bins;
    std::vector<float> w;
    float b;
};

int main (int argc, char ** argv)
{
    boost::timer timer;

    if (argc < 3)
    {
        std::cout << "Usage: " << argv[0] << " [image] [svms].gz" << std::endl;
        exit (0);
    }

    std::string imageFilename (argv[1]);
    std::string svmFilename (argv[2]);

    /***** Load image *****/
    std::cout << "Load image: " << std::flush;
    timer.restart();
    fipImage originalImage;
    originalImage.load(imageFilename.c_str());
    std::cout << timer.elapsed() << std::endl;
    /**********/

    /***** Load SVMs *****/
    std::cout << "Load SVMs: " << std::flush;
    timer.restart();
    std::ifstream file(svmFilename.c_str(), std::ios_base::in | std::ios_base::binary);
    boost::iostreams::filtering_streambuf<boost::iostreams::input> in;
    in.push(boost::iostreams::gzip_decompressor());
    in.push(file);
    std::istream incoming(&in);
    std::vector<SVM> svms;
    while (true)
    {
        SVM svm;
        incoming.read((char*)&svm.width, sizeof(uint16_t));
        if(!incoming) break;
        incoming.read((char*)&svm.height, sizeof(uint16_t));
        assert(incoming);
        incoming.read((char*)&svm.bins, sizeof(uint16_t));
        assert(incoming);
        assert(svm.bins == 31);
        
        svm.w.resize(svm.width*svm.height*svm.bins);
        incoming.read((char*)&svm.w[0], svm.width*svm.height*svm.bins*sizeof(float));
        assert(incoming);
        incoming.read((char*)&svm.b, sizeof(float));
        assert(incoming);
        svms.push_back (svm);
    }
    file.close();
    std::cout << timer.elapsed() << " seconds" << std::endl;


    /***** FOREACH scale *****/
    for (int i = 0; i < 200; ++i)
    {
        float scaler = 1.f/pow(pow(2.f, 1.f/10.f), i);
        if (scaler < 0.01f) break;
        fipImage image(originalImage);
        std::cout << "Scale: " << scaler << std::endl;
        timer.restart();
        std::cout << "Rescale image: " << std::flush;
        image.rescale(originalImage.getWidth()*scaler, originalImage.getHeight()*scaler, FILTER_BILINEAR);
        std::cout << timer.elapsed() << std::endl;

        /***** Convert to floating point color *****/
        timer.restart();
        std::cout << "Convert image: " << std::flush;
        unsigned char * d_byte_image;
        float3 * d_color_float_image;
        int imageSize = image.getWidth()*image.getHeight();
        int srcImageSize = image.isGrayscale() ? imageSize : 3*imageSize;
        cudaSafeCall(cudaMalloc((void**)&d_byte_image, srcImageSize));
        int dstImageSize = 3*imageSize;
        cudaSafeCall(cudaMalloc((void**)&d_color_float_image, dstImageSize*sizeof(float)));
        cudaSafeCall(cudaMemcpy(d_byte_image, image.accessPixels(), srcImageSize, cudaMemcpyHostToDevice));
        FormatImage<<<32, 256>>>(d_byte_image, d_color_float_image, imageSize, image.isGrayscale());
        cudaSafeCall(cudaThreadSynchronize());
        cudaSafeCall(cudaGetLastError());
        cudaSafeCall(cudaFree(d_byte_image));
        std::cout << timer.elapsed() << std::endl;

        #if 0
        std::vector<float> h_color_float_image (dstImageSize);
        cudaSafeCall(cudaMemcpy(&h_color_float_image[0], d_color_float_image, h_color_float_image.size()*sizeof(float), cudaMemcpyDeviceToHost));
        std::ofstream outImage ("color_float_dump");
        std::cout << h_color_float_image.size() << std::endl;
        outImage.write((const char *)&h_color_float_image[0], h_color_float_image.size()*sizeof(float));
        exit(0);
        #endif
        /**********/

        /***** Compute Pedro features *****/
        timer.restart();
        std::cout << "Compute features: " << std::flush;
        const int sbin = 8;

        // memory for caching orientation histograms & their norms
        int blocks_x = (int)round((float)image.getWidth()/sbin);
        int blocks_y = (int)round((float)image.getHeight()/sbin);
        float *d_hist, *d_norm;
        cudaSafeCall(cudaMalloc((void**)&d_hist, blocks_x*blocks_y*18*sizeof(float)));

        Zero<<<32, 256>>>(blocks_x*blocks_y*18, d_hist);
        cudaSafeCall(cudaThreadSynchronize());
        cudaSafeCall(cudaGetLastError());

        cudaSafeCall(cudaMalloc((void**)&d_norm, blocks_x*blocks_y*sizeof(float)));

        Zero<<<32, 256>>>(blocks_x*blocks_y, d_norm);
        cudaSafeCall(cudaThreadSynchronize());
        cudaSafeCall(cudaGetLastError());

        // memory for HOG features
        int feat_x = std::max(blocks_x-2, 0);
        int feat_y = std::max(blocks_y-2, 0);
        int feat_bins = 27+4;
        float *d_feat;
        cudaSafeCall(cudaMalloc((void**)&d_feat, feat_x*feat_y*feat_bins*sizeof(float)));

        Zero<<<32, 256>>>(feat_x*feat_y*feat_bins, d_feat);
        cudaSafeCall(cudaThreadSynchronize());
        cudaSafeCall(cudaGetLastError());


        int visible_x = blocks_x*sbin;
        int visible_y = blocks_y*sbin;

        ComputeHistograms<<<32, 256>>>(
            d_color_float_image,
            image.getWidth(),
            image.getHeight(), 
            visible_x,
            visible_y,
            blocks_x,
            blocks_y,
            d_hist);
        cudaSafeCall(cudaThreadSynchronize());
        cudaSafeCall(cudaGetLastError());
        cudaSafeCall(cudaFree(d_color_float_image));

        #if 0
        std::vector<float> h_hist (blocks_x*blocks_y*18);
        cudaSafeCall(cudaMemcpy(&h_hist[0], d_hist, h_hist.size()*sizeof(float), cudaMemcpyDeviceToHost));
        std::ofstream histDump ("hist_dump");
        std::cout << blocks_x << " " << blocks_y << " 18" << std::endl;
        histDump.write((const char *)&h_hist[0], h_hist.size()*sizeof(float));
        exit(0);
        #endif

        ComputeEnergy<<<32, 256>>>(
            d_hist,
            blocks_x,
            blocks_y,
            d_norm);
        cudaSafeCall(cudaThreadSynchronize());
        cudaSafeCall(cudaGetLastError());

        #if 0
        std::vector<float> h_norm (blocks_x*blocks_y);
        cudaSafeCall(cudaMemcpy(&h_norm[0], d_norm, h_norm.size()*sizeof(float), cudaMemcpyDeviceToHost));
        std::ofstream normDump ("norm_dump");
        std::cout << blocks_x << " " << blocks_y << std::endl;
        normDump.write((const char *)&h_norm[0], h_norm.size()*sizeof(float));
        exit(0);
        #endif

        ComputeFeatures<<<32, 256>>>(
            d_hist,
            d_norm,
            blocks_x,
            blocks_y,
            feat_x,
            feat_y,
            d_feat);
        cudaSafeCall(cudaThreadSynchronize());
        cudaSafeCall(cudaGetLastError());
        cudaSafeCall(cudaFree(d_hist));
        cudaSafeCall(cudaFree(d_norm));
        std::cout << timer.elapsed() << std::endl;

        #if 0
        std::vector<float> h_feat (feat_x*feat_y*feat_bins);
        cudaSafeCall(cudaMemcpy(&h_feat[0], d_feat, h_feat.size()*sizeof(float), cudaMemcpyDeviceToHost));
        std::ofstream featDump ("feat_dump");
        std::cout << feat_x << " " << feat_y << " " << feat_bins << std::endl;
        featDump.write((const char *)&h_feat[0], h_feat.size()*sizeof(float));
        exit(0);
        #endif
        /**********/

        std::cout << "Features: (" << feat_x << ", " << feat_y << ")" << std::endl;

        int pad_x = 1;
        while (feat_x > pad_x) pad_x <<= 1;
        int pad_y = 1;
        while (feat_y > pad_y) pad_y <<= 1;

        std::cout << "Padded features: (" << pad_x << ", " << pad_y << ")" << std::endl;
        
        cufftReal * d_feat_pad;
        cudaSafeCall(cudaMalloc((void**)&d_feat_pad, pad_x*pad_y*feat_bins*sizeof(cufftReal)));

        PadFeatures<<<32, 256>>>(
            d_feat,
            feat_x,
            feat_y,
            pad_x,
            pad_y,
            d_feat_pad);
        cudaSafeCall(cudaThreadSynchronize());
        cudaSafeCall(cudaGetLastError());
        cudaSafeCall(cudaFree(d_feat));

        #if 0
        std::vector<cufftReal> h_feat_pad (pad_x*pad_y*feat_bins);
        cudaSafeCall(cudaMemcpy(&h_feat_pad[0], d_feat_pad, h_feat_pad.size()*sizeof(cufftReal), cudaMemcpyDeviceToHost));
        std::ofstream featPadDump ("feat_pad_dump");
        std::cout << pad_x << " " << pad_y << " " << feat_bins << std::endl;
        featPadDump.write((const char *)&h_feat_pad[0], h_feat_pad.size()*sizeof(cufftReal));
        exit(0);
        #endif

        /***** Apply FFT to input image *****/
        cufftHandle planImage;
        int n[2] = {pad_x, pad_y};
        timer.restart();
        std::cout << "FFT on input image features: " << std::flush;
        //cufftSafeCall(cufftPlanMany(&planImage, 2, n, NULL, 1, 0, NULL, 1, 0, CUFFT_R2C, feat_bins));
        cufftSafeCall(cufftPlan2d(&planImage, pad_x, pad_y, CUFFT_R2C));
        cufftComplex * d_feat_freq;
        // Note: for R2C CUFFT only stores non-redundant complex coefficients
        cudaSafeCall(cudaMalloc((void**)&d_feat_freq, sizeof(cufftComplex)*pad_x*(pad_y/2+1)*feat_bins));
        for (int j = 0; j < feat_bins; ++j)
        {
            cufftSafeCall(cufftExecR2C(planImage, d_feat_pad + pad_x*pad_y*j, d_feat_freq + pad_x*(pad_y/2+1)*j));
            cudaSafeCall(cudaThreadSynchronize());
            cudaSafeCall(cudaGetLastError());
        }
        cufftSafeCall(cufftDestroy(planImage));
        std::cout << timer.elapsed() << " seconds" << std::endl;

        #if 1
        std::vector<cufftComplex> h_feat_freq (pad_x*(pad_y/2+1)*feat_bins);
        cudaSafeCall(cudaMemcpy(&h_feat_freq[0], d_feat_freq, h_feat_freq.size()*sizeof(cufftComplex), cudaMemcpyDeviceToHost));
        std::ofstream featFreqDump ("feat_freq_dump");
        std::cout << pad_x << " " << pad_y << " " << feat_bins << std::endl;
        featFreqDump.write((const char *)&h_feat_freq[0], h_feat_freq.size()*sizeof(cufftComplex));
        exit(0);
        #endif
        /**********/

        #if 1
        exit(0);
        #endif

        /***** FOREACH SVM *****/
        for (std::vector<SVM>::const_iterator j = svms.begin(); j != svms.end(); ++j)
        {
            // Image too small for filter
            if (j->width > feat_x || j->height > feat_y)
            {
                continue;
            }
            std::cout << j - svms.begin() << std::endl;
            float * d_filter, * d_filter_padded;
            cudaSafeCall(cudaMalloc((void**)&d_filter, sizeof(float)*j->w.size()));
            cudaSafeCall(cudaMemcpy(d_filter, &j->w[0], sizeof(float)*j->w.size(), cudaMemcpyHostToDevice));
            cudaSafeCall(cudaMalloc((void**)&d_filter_padded, sizeof(float)*feat_bins*feat_x*feat_y));

/*
            PadFilter<<<32, 256>>>(
                d_filter,
                j->width,
                j->height,
                feat_x,
                feat_y,
                feat_bins,
                d_filter_padded);
            cudaSafeCall(cudaThreadSynchronize());
            cudaSafeCall(cudaGetLastError());
*/
        }
       
        /***** Free memory for image features and freq transform *****/ 
        cudaSafeCall(cudaFree(d_feat_pad));
        cudaSafeCall(cudaFree(d_feat_freq));
    }

    #if 1
    exit(0);
    #endif

    int NX = 128;
    int NY = 128;
    
    cufftHandle plan;
    cufftComplex *h_image, *d_image;
    cufftComplex *h_svm, *d_svm;
    cufftComplex *d_image_f, *d_svm_f;
    cufftComplex bias;
    bias.x = 0.f;
    bias.y = 0.f;

    h_image = (cufftComplex*)malloc(sizeof(cufftComplex)*NX*NY);
    h_svm = (cufftComplex*)malloc(sizeof(cufftComplex)*NX*NY);

    cudaSafeCall(cudaMalloc((void**)&d_image, sizeof(cufftComplex)*NX*NY));
    cudaSafeCall(cudaMalloc((void**)&d_image_f, sizeof(cufftComplex)*NX*NY));
    cudaSafeCall(cudaMalloc((void**)&d_svm, sizeof(cufftComplex)*NX*NY));
    cudaSafeCall(cudaMalloc((void**)&d_svm_f, sizeof(cufftComplex)*NX*NY));

    cudaSafeCall(cudaMemcpy(d_image, h_image, sizeof(cufftComplex)*NX*NY, cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(d_svm, h_svm, sizeof(cufftComplex)*NX*NY, cudaMemcpyHostToDevice));

    cufftSafeCall(cufftPlan2d(&plan, NX, NY, CUFFT_C2C));

    for (int i = 0; i < 9360*31; ++i)
    {
    cufftSafeCall(cufftExecC2C(plan, d_image, d_image_f, CUFFT_FORWARD));
    cufftSafeCall(cufftExecC2C(plan, d_svm, d_svm_f, CUFFT_FORWARD));

    PointwiseMul<<<32, 256>>>(d_svm_f, d_image_f, NX*NY);
    cudaSafeCall(cudaGetLastError());

    cufftSafeCall(cufftExecC2C(plan, d_svm_f, d_svm, CUFFT_INVERSE));

    Add<<<32, 256>>>(d_svm, bias, NX*NY);
    cudaSafeCall(cudaGetLastError());
    }
    printf("%f seconds\n", timer.elapsed());

    cudaSafeCall(cudaMemcpy(h_svm, d_svm, sizeof(cufftComplex)*NX*NY, cudaMemcpyDeviceToHost));

    cufftSafeCall(cufftDestroy(plan));
    cudaSafeCall(cudaFree(d_image));
    cudaSafeCall(cudaFree(d_image_f));
    cudaSafeCall(cudaFree(d_svm));
    cudaSafeCall(cudaFree(d_svm_f));

    free(h_image);
    free(h_svm);

    return 0;
}

static __global__ void PointwiseMul(cufftComplex* a, const cufftComplex* b, int size)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < size; i += numThreads)
    {
        cufftComplex c;
        const cufftComplex * a_local = a + i;
        const cufftComplex * b_local = b + i;
        c.x = a_local->x * b_local->x - a_local->y * b_local->y;
        c.y = a_local->x * b_local->y + a_local->y * b_local->x;
        a[i] = c;
    }
}

static __global__ void Add(cufftComplex* a, const cufftComplex b, int size)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < size; i += numThreads)
    {
        a[i].x += b.x;     
        a[i].y += b.y;
    }
}

static __global__ void Zero(int size, float * buf)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < size; i += numThreads)
    {
        buf[i] = 0.f;
    }
}

static __global__ void FormatImage(const unsigned char * byte_image, float3 * color_float_image, int size, bool grayscale)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = threadID; i < size; i += numThreads)
    {
        color_float_image[i].x = (grayscale ? byte_image[i] : byte_image[3*i])/255.f;
        color_float_image[i].y = (grayscale ? byte_image[i] : byte_image[3*i+1])/255.f;
        color_float_image[i].z = (grayscale ? byte_image[i] : byte_image[3*i+2])/255.f;
    }
}

static __global__ void ComputeHistograms(const float3 * color_float_image, int width, int height, int visible_x, int visible_y, int blocks_x, int blocks_y, float * hist)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    const float uu[9] = {1.0000f, 0.9397f, 0.7660f, 0.500f, 0.1736f, -0.1736f, -0.5000f, -0.7660f, -0.9397f};
    const float vv[9] = {0.0000f, 0.3420f, 0.6428f, 0.8660f, 0.9848f, 0.9848f, 0.8660f, 0.6428f, 0.3420f};

    const int sbin = 8;

    const int size = visible_x*visible_y;
    for (int i = threadID; i < size; i += numThreads)
    {
        int x = i / visible_y;
        int y = i % visible_y;

        if (x == 0 || y == 0 || x >= visible_x-1 || y >= visible_y-1)
            continue;

        const float3 * s = color_float_image + (height-1 - min(y,height-2))*width + min(x,width-2);
        float3 dx = *(s+1) - *(s-1);
        float3 dy = *(s-width) - *(s+width);
        float3 v = dx*dx + dy*dy;

        // pick channel with strongest gradient
        float v_max = v.x;
        float dx_max = dx.x;
        float dy_max = dy.x;
        if (v.y > v_max)
        {
            v_max = v.y;
            dx_max = dx.y;
            dy_max = dy.y;
        }
        if (v.z > v_max)
        {
            v_max = v.z;
            dx_max = dx.z;
            dy_max = dy.z;
        }

        // snap to one of 18 orientations
        float best_dot = 0.f;
        int best_o = 0;
        for (int o = 0; o < 9; ++o)
        {
            float dot = uu[o]*dx_max + vv[o]*dy_max;
            if (dot > best_dot)
            {
                best_dot = dot;
                best_o = o;
            }
            else if (-dot > best_dot)
            {
                best_dot = -dot;
                best_o = o+9;
            }
        }

        // snap to one of 18 orientations
        float xp = ((float)x+0.5f)/sbin - 0.5f;
        float yp = ((float)y+0.5f)/sbin - 0.5f;
        int ixp = (int)floor(xp);
        int iyp = (int)floor(yp);
        float vx0 = xp-ixp;
        float vy0 = yp-iyp;
        float vx1 = 1.f - vx0;
        float vy1 = 1.f - vy0;
        v_max = sqrt(v_max);

        if (ixp >= 0 && iyp >= 0)
            atomicAdd(hist + ixp*blocks_y + iyp + best_o*blocks_x*blocks_y, vx1*vy1*v_max);

        if (ixp+1 < blocks_x && iyp >= 0)
            atomicAdd(hist + (ixp+1)*blocks_y + iyp + best_o*blocks_x*blocks_y, vx0*vy1*v_max);

        if (ixp >= 0 && iyp+1 < blocks_y)
            atomicAdd(hist + ixp*blocks_y + (iyp+1) + best_o*blocks_x*blocks_y, vx1*vy0*v_max);

        if (ixp+1 < blocks_x && iyp+1 < blocks_y)
            atomicAdd(hist + (ixp+1)*blocks_y + (iyp+1) + best_o*blocks_x*blocks_y, vx0*vy0*v_max); 
    }
}

static __global__ void ComputeEnergy(const float * hist, int blocks_x, int blocks_y, float * norm)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    // compute energy in each block by summing over orientations
    for (int o = threadID; o < 9; o += numThreads)
    {
        const float * src1 = hist + o*blocks_x*blocks_y;
        const float * src2 = hist + (o+9)*blocks_x*blocks_y;
        float * dst = norm;
        float * end = norm + blocks_x*blocks_y;
        while (dst < end)
        {
            atomicAdd(dst, (*src1 + *src2) * (*src1 + *src2));
            ++dst;
            ++src1;
            ++src2;
        }
    }
}

static __global__ void ComputeFeatures(const float * hist, const float * norm, int blocks_x, int blocks_y, int feat_x, int feat_y, float * feat)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    const float eps = 0.0001f;

    const int feats = feat_x*feat_y;
    for (int i = threadID; i < feats; i += numThreads)
    {
        int x = i / feat_y;
        int y = i % feat_y;

        float * dst = feat + x*feat_y + y;
        const float *src, *p;
        float n1, n2, n3, n4;

        p = norm + (x+1)*blocks_y + y+1;
        n1 = 1.f / sqrt(*p + *(p+1) + *(p+blocks_y) + *(p + blocks_y+1) + eps);
        p = norm + (x+1)*blocks_y + y;
        n2 = 1.f / sqrt(*p + *(p+1) + *(p+blocks_y) + *(p + blocks_y+1) + eps);
        p = norm + x*blocks_y + y+1;
        n3 = 1.f / sqrt(*p + *(p+1) + *(p+blocks_y) + *(p + blocks_y+1) + eps);
        p = norm + x*blocks_y + y;
        n4 = 1.f / sqrt(*p + *(p+1) + *(p+blocks_y) + *(p + blocks_y+1) + eps);

        float t1 = 0.f;
        float t2 = 0.f;
        float t3 = 0.f;
        float t4 = 0.f;

        // contrast-sensitive features
        src = hist + (x+1)*blocks_y + (y+1);
        for (int o = 0; o < 18; ++o)
        {
            float h1 = min(*src * n1, 0.2f);
            float h2 = min(*src * n2, 0.2f);
            float h3 = min(*src * n3, 0.2f);
            float h4 = min(*src * n4, 0.2f);
            *dst = 0.5f * (h1 + h2 + h3 + h4);
            t1 += h1;
            t2 += h2;
            t3 += h3;
            t4 += h4;
            dst += feat_x*feat_y;
            src += blocks_x*blocks_y;
        }

        // contrast-insensitive features
        src = hist + (x+1)*blocks_y + (y+1);
        for (int o = 0; o < 9; ++o)
        {
            float sum = *src + *(src + 9*blocks_x*blocks_y);
            float h1 = min(sum * n1, 0.2f);
            float h2 = min(sum * n2, 0.2f);
            float h3 = min(sum * n3, 0.2f);
            float h4 = min(sum * n4, 0.2f);
            *dst = 0.5f * (h1 + h2 + h3 + h4);
            dst += feat_x*feat_y;
            src += blocks_x*blocks_y;
        }

        // texture features
        *dst = 0.2357f * t1;
        dst += feat_x*feat_y;
        *dst = 0.2357f * t2;
        dst += feat_x*feat_y;
        *dst = 0.2357f * t3;
        dst += feat_x*feat_y;
        *dst = 0.2357f * t4;
    }
}

static __global__ void PadFeatures(const float * feat, int feat_x, int feat_y, int pad_x, int pad_y, cufftReal * feat_pad)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    const int size = pad_x*pad_y*31;
    for (int i = threadID; i < size; i += numThreads)
    {
        int bin = i / (pad_x*pad_y);
        int rem = i % (pad_x*pad_y);
        int x = rem / pad_y;
        int y = rem % pad_y;

        int j = bin*feat_x*feat_y + x*feat_y + y;
        if (x >= feat_x || y >= feat_y) feat_pad[i] = 0.f;
        else feat_pad[i] = feat[j];
    }
}
