#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <chrono>
// #include "readDCNN.h"
// #include "ppmIO.h"
// #include <cuda.h>

#define GENERIC_ERR cudaError()
typedef float mapel;
typedef int size; 
#define EPS 1e-4
#define mapel_cv CV_32F
#define TILE_SIZE 16 

#define MAX_NO_WEIGHTS 16800
#define MAX_NO_WIN2DET 1024
#define MAX_NO_IMGS 64
#define MAX_NO_CHANNELS 4

#define SOBEL_KSIZE 3 
cv::Mat SOBEL_KERNEL_H;
cv::Mat SOBEL_KERNEL_W;

#define CELL_SIZE 8
#define CELL_ADJ 2
#define BINS_NO 9
#define LOWE_NORMF .2

#define WIN_SLIDE 2
#define PERSON_CROP_H 128
#define PERSON_CROP_W 64
// #define PERSON_CROP_H 16 //dg
// #define PERSON_CROP_W 16

std::chrono::steady_clock::time_point t0;
std::chrono::steady_clock::time_point t1;
std::chrono::duration<float> td;

//dg
// cv::Mat h;
// cv::Mat h2[10];
// double hmin, hmax;
// float hd;
// using namespace cv;
// using namespace std;

// cv::Mat get_sobel_kernels
inline void pdca(FILE* stream, char* src) 
    {for (int i=0; src[i]!=0; ++i) fprintf(stream, "%c", src[i]);}

inline __device__ size xm(  size lNM, size lMC, size lY, size lX, 
                            size inm, size imc, size iy, size ix)
{
    if (inm < lNM && imc < lMC && iy < lY && ix < lX
        && inm >= 0 && imc >= 0 && iy >= 0 && ix >=0) 
        return inm *lMC*lY*lX + imc *lY*lX + iy *lX + ix; 
    else {printf("(%d, %d, %d, %d)>(%d, %d, %d, %d)\n", 
        lNM, lMC, lY, lX, inm, imc, iy, ix); return -1;}
}

inline __device__ int log2round(int n)
{
    // REMARK: TO BE CHANGED
    // printf("i%d", n); //dg
    if (n < 0) return -1;
    int sh = 0;
    for (; n > 1; n>>=1) ++sh;
    // printf("o%d", 1<<sh); //dg
    return 1<<sh;
}


void cpu_hog(cv::Mat* OP, cv::Mat* IN, size N)
{
    cv::Mat img; 
    cv::Mat imc[MAX_NO_CHANNELS]; 
    size    C = IN[0].channels();
    size    H = IN[0].rows;
    size    W = IN[0].cols;

    cv::Mat kH = SOBEL_KERNEL_H(cv::Range(0, SOBEL_KSIZE),
                                cv::Range(0, SOBEL_KSIZE));
    cv::Mat kW = SOBEL_KERNEL_W(cv::Range(0, SOBEL_KSIZE),
                                cv::Range(0, SOBEL_KSIZE));

    cv::Mat sbcy;   cv::Mat sbcx;
    cv::Mat amp;    cv::Mat ampc[MAX_NO_CHANNELS]; cv::Mat ampceq;
    cv::Mat arc;    cv::Mat arcc[MAX_NO_CHANNELS];

    size Hp     = floor((float)H/CELL_SIZE); size Hpm = max(Hp, 2);
    size Wp     = floor((float)W/CELL_SIZE); size Wpm = max(Wp, 2); // discard partials
    size soCHc  = BINS_NO * sizeof(mapel);
    size lCS    = Hpm * Wpm;
    size soCH   = lCS * soCHc; 
    size soCS   = lCS * sizeof(mapel);
    mapel* CH   = (mapel*)malloc(soCH); // patch hist
    mapel* CS   = (mapel*)malloc(soCS); // cached sums

    cv::Mat arcn; cv::Mat ampp; cv::Mat arcp; cv::Mat arcb;
    cv::Mat b0md; cv::Mat b1md; // bin +X ifmad

    size lEHc   = (CELL_ADJ*CELL_ADJ) * BINS_NO;
    // size soEHc  = lEHc * sizeof(mapel);
    size lEH    = (Hpm-1) * (Wpm-1) * lEHc;
    size soEH   = lEH * sizeof(mapel);
    mapel* EH   = (mapel*)malloc(soEH); // equalized hists
    cv::Mat op;
    
    for (size nx = 0; nx < N; ++nx)
{
    t0 = std::chrono::high_resolution_clock::now();
        
    // POLAR GRAD
    img = IN[nx];
    cv::split(img, imc);
    amp = cv::Mat(H, W, mapel_cv, cv::Scalar(0));
    arc = cv::Mat(H, W, mapel_cv, cv::Scalar(0));
    
    for (size cx = 0; cx < C; ++cx)
    {
        cv::extractChannel(img, imc[cx], cx);
        cv::filter2D(imc[cx], sbcy, mapel_cv, kH, cv::Point(-1, -1), 0., 0);
        cv::filter2D(imc[cx], sbcx, mapel_cv, kW, cv::Point(-1, -1), 0., 0);
        cv::cartToPolar(sbcx, sbcy, ampc[cx], arcc[cx]);
        cv::add(ampc[cx], -cx * EPS, ampc[cx]); // eps to differentiate channels
        amp = cv::max(amp, ampc[cx]);
    }
    
    for (size cx = 0; cx < C; ++cx)
    {
        cv::compare(amp, ampc[cx], ampceq, cv::CmpTypes::CMP_EQ);
        cv::add(arc, arcc[cx], arc, ampceq); 
    }    

    t1 = std::chrono::high_resolution_clock::now();
    td= t1 - t0;
    fprintf(stderr, "Polar grad execution time = %f\n", (float)td.count());

    // HIST APPEND
    memset(CH, 0, soCH);
    memset(CS, 0, soCS);
    // cv::add(arc, 1, arc, arc > CV_PI); // (arc % pi) // wtf ???
    cv::add(arc, -CV_PI, arc, arc > CV_PI); // (arc % pi)

    t0 = std::chrono::high_resolution_clock::now();
    
    cv::multiply(arc, BINS_NO/CV_PI, arcn);

    // printf("amp "); for (int i = 0; i < 8; ++i) { for (int j=0;j<8;++j) printf("%.2f ", *(float*)amp.ptr(i,j)); printf("\n    ");} printf("\n"); //dg
    // printf("arc "); for (int i = 0; i < 8; ++i) { for (int j=0;j<8;++j) printf("%.2f ", *(float*)arcn.ptr(i,j)); printf("\n    ");} printf("\n"); //dg

    for     (size py = 0; py < Hp; ++py)
        for (size px = 0; px < Wp; ++px) // leave Xpm cell vals 0 if out of img
    {
        ampp = amp( cv::Range(py * CELL_SIZE, (py+1) * CELL_SIZE),
                    cv::Range(px * CELL_SIZE, (px+1) * CELL_SIZE));
        arcp = arcn(cv::Range(py * CELL_SIZE, (py+1) * CELL_SIZE),
                    cv::Range(px * CELL_SIZE, (px+1) * CELL_SIZE));
        
        size ploc = py*Wp+px;
        mapel* CHc = &CH[ploc*BINS_NO]; 
        mapel* CSc = &CS[ploc];
        
        for (size bx = 0; bx < BINS_NO+1; ++bx)
        {
            arcb = (bx <= arcp) & (arcp < (bx+1));
            arcb.convertTo(arcb, mapel_cv);

            b0md = (bx+1 - arcp);
            b1md = (arcp - bx);
            cv::multiply(arcb, b0md, b0md, 1./255); cv::multiply(ampp, b0md, b0md);
            cv::multiply(arcb, b1md, b1md, 1./255); cv::multiply(ampp, b1md, b1md);
            
            CHc[bx%BINS_NO]     += cv::sum(b0md)[0];
            CHc[(bx+1)%BINS_NO] += cv::sum(b1md)[0];
        }
        for (size bx = 0; bx < BINS_NO; ++bx)  CSc[0] += CHc[bx];
        // printf("ch  "); for (int j=0;j<BINS_NO;++j) printf("%.2f ", CHc[j]); printf("\n    "); printf("\n"); //dg
    }

    t1 = std::chrono::high_resolution_clock::now();
    td= t1 - t0;
    fprintf(stderr, "Hist append get execution time = %f\n", (float)td.count());

    // HIST EQUALIZE

    t1 = std::chrono::high_resolution_clock::now();

    for     (size py = 0; py < (Hpm-1); ++py)
        for (size px = 0; px < (Wpm-1); ++px)
    {
        size eloc = py*(Wpm-1)+px;
        mapel* EHc = &EH[eloc*lEHc];
        mapel eqel; mapel esc1 = 0; mapel esc2 = 0;

        for     (size ay = 0; ay < CELL_ADJ; ++ay) 
            for (size ax = 0; ax < CELL_ADJ; ++ax)
        {
            size ploc = (py+ay)*Wp+(px+ax);
            esc1 += CS[ploc];
        }
        for     (size ay = 0; ay < CELL_ADJ; ++ay) 
            for (size ax = 0; ax < CELL_ADJ; ++ax)
        {
            size ploc = (py+ay)*Wp+(px+ax);
            size bloc = ay*CELL_ADJ+ax;
            for (size bx = 0; bx < BINS_NO;  ++bx)
            {
                eqel = min(CH[ploc*BINS_NO+bx]/esc1, .2);
                EHc[bloc*BINS_NO+bx] = eqel;
                esc2 += eqel;
            }
        }
        for     (size ay = 0; ay < CELL_ADJ; ++ay) 
            for (size ax = 0; ax < CELL_ADJ; ++ax)
        {
            size bloc = ay*CELL_ADJ+ax;
            for (size bx = 0; bx < BINS_NO;  ++bx)
                EHc[bloc*BINS_NO+bx] /= esc2;
        }
    }

    t1 = std::chrono::high_resolution_clock::now();
    td= t1 - t0;
    fprintf(stderr, "Hist norm execution time = %f\n", (float)td.count());

    OP[nx] = cv::Mat(lEH, 1, mapel_cv, (void*)EH).clone();
    // OP[nx] = cv::Mat(lEH, 1, mapel_cv);
    // for(size i = 0; i < lEH; ++i) *(float*)OP[nx].ptr(0) = EH[i];
    // if (1) {printf("eh  "); for (int j=0;j<36;++j) printf("%.2f ", EH[j]); printf("\n    "); printf("\n");} //dg

    // printf("cpu_hog[img=%d]_finished_succesfully\n", nx); //dg
}
    free(CH); free(CS); free(EH);
    return;

    /* DEGUG SNIPPETS
    printf("a1"); //dg
    printf("arc "); for (int i = 0; i < 8; ++i) { for (int j=0;j<8;++j) printf("%.2f ", *(float*)arc.ptr(i,j)); printf("\n    ");} printf("\n"); //dg
    printf("e2[%d %d] ", py, px); for (int i = 0; i < 36; ++i) printf("%.2f ", EHc[i]); printf("\n"); //dg
    printf("hog "); for (int i = 0; i < 15*7; ++i) { for (int j=0;j<36;++j) printf("%.2f ", EH[i*36+j]); printf("\n    ");} printf("\n"); //dg
    printf("%d, %d", OP[0].rows, OP[0].cols);
    */
}

void cpu_svm(mapel* OP, cv::Mat* HO, size N, mapel* IQ, size M, mapel IQ_bias)
{
    t0 = std::chrono::high_resolution_clock::now();
        
    for (size nx = 0; nx < N; ++nx)
    {
        mapel* HOn = (mapel*)(HO[nx].ptr(0));
        OP[nx] = IQ_bias; 
        for(size qx = 0; qx < M; ++qx) OP[nx] += HOn[qx] * IQ[qx];
        // for(size qx = 0; qx < M; ++qx) printf("%f ", HOn[qx] * IQ[qx]); printf("\n");//dg
    }

    t1 = std::chrono::high_resolution_clock::now();
    td= t1 - t0;
    fprintf(stderr, "SVM execution time = %f\n", (float)td.count());

    return;
}


__global__ void kern_polar_grad(mapel* OP, mapel* IN,
    mapel* IK, size C, size H, size W)
{
    extern __shared__ mapel exdbm[];
    
    // CONST > KERNEL
    const size KS = SOBEL_KSIZE;
    const size KB2 = SOBEL_KSIZE-1;
    const size KB = KB2/2;
    // IMG_RES > META
    size nx = blockIdx.x; size N    = gridDim.x;

    size ty = blockIdx.y; size t2ly = blockDim.x;
    size tx = blockIdx.z; size t2lx = blockDim.y;
    // IMG_RES > LOCAL (TILE)
    size ly = threadIdx.x; 
    size lx = threadIdx.y;
    // IMG_RES > GLOBAL
    size gy = ly + (ty * t2ly);
    size gx = lx + (tx * t2lx);

    // LOCALIZE > IN
    size t2lyb = t2ly + KB2; // bordered
    size t2lxb = t2lx + KB2;  
    // printf("%d %d %d\n", t2lyb, t2lxb, KB); //dg
    size lsINn = C * t2lyb * t2lxb;    
    mapel* sINn = &exdbm[0];

    for         (size cx = 0; cx < C; ++cx) // channel
        for     (size sly = -KB + ly; sly < t2ly + KB; sly += t2ly) // tile shape step
            for (size slx = -KB + lx; slx < t2lx + KB; slx += t2lx) // ... max 3x/corner
    {
        // if (cx==0)printf("[%d %d] %d %d\n", ly, lx, sly, slx);
        size sgy = sly + (t2ly * ty);
        size sgx = slx + (t2lx * tx);

        if (0 <= sgy && sgy < H && 0 <= sgx && sgx < W)
            sINn[       xm(1,C,t2lyb,t2lxb,    0,  cx, sly+KB, slx+KB)]
                = IN[   xm(N,C,H,    W,        nx, cx, sgy,    sgx)];
        else sINn[      xm(1,C,t2lyb,t2lxb,    0,  cx, sly+KB, slx+KB)] = 0.;
    }
    
    // if (gy==0&&gx==0) //dg
    // {
    // //     // printf("in  "); for (int i = 0; i < 8; ++i) { for (int j=0;j<8;++j) printf("%.2f ", IN[xm(N,C,H,W,0,0,i,j)]); printf("\n    ");} printf("\n"); //dg
    //     printf("in  "); for (int i = 0; i < 18; ++i) { for (int j=0;j<18;++j) printf("%.2f ", sINn[xm(N,C,t2lyb,t2lxb,0,0,i,j)]); printf("\n    ");} printf("\n"); //dg
    // } // it has to be here bc does not work without it

    // LOCALIZE > IK (linearized)
    size lIK  = 2 * SOBEL_KSIZE * SOBEL_KSIZE;
    mapel* sIK = &exdbm[lsINn];
    for (size lizd = ly*t2lx+lx; lizd < lIK; lizd+=t2ly*t2lx)
        sIK[lizd] = IK[lizd];
    __syncthreads();

    // CALC
    mapel sby;      mapel sbcy;
    mapel sbx;      mapel sbcx; 
    mapel amp = 0;  mapel ampc;
    mapel arc = 0;

    for (size cx = 0; cx < C; ++cx)
    {
        // CALC > SOBEL 
        sbcy = 0; sbcx = 0;
        for     (size ky = -KB; ky <= KB; ++ky)
            for (size kx = -KB; kx <= KB; ++kx)
        {
            sbcy += sINn[   xm(1,C,t2lyb,t2lxb, 0, cx, ly+KB+ky, lx+KB+kx)] 
                * sIK[      xm(1,2,KS,   KS,    0, 0,  ky+KB,    kx+KB)];
            sbcx += sINn[   xm(1,C,t2lyb,t2lxb, 0, cx, ly+KB+ky, lx+KB+kx)]
                * sIK[      xm(1,2,KS,   KS,    0, 1,  ky+KB,    kx+KB)];
        }
        // sbcy /= 255; sbcx /= 255;
        // CALC > MAGNITUDE**2
        ampc = (sbcy * sbcy) + (sbcx * sbcx); //dg
        // ampc = (sbcy * sbcy) + (sbcx * sbcx); //dg
        if (ampc > amp) {amp = ampc; sby = sbcy; sbx = sbcx;}
    }
    // CALC > POLAR
    amp = sqrtf(amp);  //todo: "0 when compiled with -prec-sqrt=true"
    if (amp > 0) arc = atan2f(sby, sbx); else arc = 0;
    if (arc < 0) arc += CV_PI; // +-pi -> [0, 2pi]

    // GLOBALIZE > OP
    if (gy<H && gx<W)
    {
        OP[xm(N,2,H,W, nx, 0, gy, gx)] = amp;
        OP[xm(N,2,H,W, nx, 1, gy, gx)] = arc;
    } 
    /**/
    // __syncthreads();
    // if (gy==0&&gx==0)
    // {
    // //     if (nx==0) for (int n=0; n<N; ++n)
    // //     {
    // //         printf("amp "); for (int i = 0; i < 8; ++i) { for (int j=0;j<8;++j) printf("%.1f ", OP[xm(N,2,H,W,0,0,i,j)]); printf("\n    ");} printf("\n"); //dg
    // // //     }
    // //     printf("arc "); for (int i = 0; i < H; ++i) { for (int j=0;j<W;++j) printf("%.2f ", OP[xm(N,2,H,W,0,1,i,j)]); printf("\n    ");} printf("\n"); //dg
    // //     // printf("ik  "); for (int i = 0; i < 2; ++i) { for (int j=0;j<9;++j) printf("%.2f ", sIK[i*9+j]); printf("\n    ");} printf("\n"); //dg
    // }
}

__global__ void kern_hist_append(mapel* OP, mapel* PG, size Hpc, size H, size W)
{
    extern __shared__ mapel exdbm[];
    // CONST
    mapel bin2arc = BINS_NO / CV_PI;
    // IMG_RES > META
    size Hb2p = blockDim.z;
    size nx =   blockIdx.x;                 size N =    gridDim.x;
    size Hp =   gridDim.y * Hb2p;           size Wp =   gridDim.z;
    // IMG_RES > TILING 
    size b2ty = threadIdx.z;
    size ty =   blockIdx.y * Hb2p + b2ty;   size t2ly = blockDim.x;
    size tx =   blockIdx.z;                 size t2lx = blockDim.y;
    // IMG_RES > LOCAL (PATCH)
    size ly =   threadIdx.x;
    size lx =   threadIdx.y;
    size ll =   ly * t2lx + lx;
    // IMG_RES > GLOBAL
    size gy = ly + (ty * t2ly);
    size gx = lx + (tx * t2lx);

    // LOCALIZE
    size lCHpb = t2ly * t2lx;
    mapel* sCHp = &exdbm[b2ty * lCHpb * BINS_NO]; 

    for (size bx=0; bx<BINS_NO; ++bx) 
        sCHp[xm(1,t2ly,t2lx,BINS_NO, 0, ly, lx, bx)] = 0;

    if (gy < H && gx < W)
    {
        mapel amp = PG[xm(N,2,H,W, nx, 0, gy, gx)];
        mapel arc = PG[xm(N,2,H,W, nx, 1, gy, gx)];
        if (arc > CV_PI) arc -= CV_PI; // (arc % pi)
        arc *= bin2arc;
        // if (!(0<=amp&&amp<500 && 0<=arc&&arc<9)) fprintf(stderr, "%f, %f\n", amp, arc); //dg

    // CALC
        size bin0 = (size)arc;  mapel b0sh = (bin0+1) - arc;
        size bin1 = (bin0+1);   mapel b1sh = arc - bin0;

    // APPEND
        sCHp[xm(1,t2ly,t2lx,BINS_NO, 0, ly, lx, bin0 % BINS_NO)] = b0sh * amp;
        sCHp[xm(1,t2ly,t2lx,BINS_NO, 0, ly, lx, bin1 % BINS_NO)] = b1sh * amp;
        __syncthreads();

    // REDUCE
        for (size bx=0; bx<BINS_NO; ++bx)
            for (size ar=log2round(lCHpb); ar>0; ar/=2) 
        {
            if (ll < ar && ll + ar < lCHpb) 
                sCHp[       xm(1,1,t2ly*t2lx,BINS_NO, 0, 0, ll,     bx)] 
                    += sCHp[xm(1,1,t2ly*t2lx,BINS_NO, 0, 0, ll+ar,  bx)];
            __syncthreads();
        }
        // PG[xm(N,2,H,W, nx, 1, gy, gx)] = arc; //dg
    }
    // if(ly==0&&lx==0) if(ty==0&&tx==0) //dg
    // {
    //     printf("arc "); for (int i = 0; i < 8; ++i) { for (int j=0;j<8;++j) printf("%.2f ", PG[xm(N,2,H,W, nx, 1, i, j)]); printf("\n    ");} printf("\n");
    //     printf("[%d %d] ", ty, tx); for (size bx = 0; bx < BINS_NO; ++bx) {/*printf("|");*/ printf("%.2f ", sCHp[bx]);} printf("\n");
    // }

    // GLOBALIZE
    if (ty < Hpc && tx < Wp)
        for (size bx=ll; bx < BINS_NO; ++bx)
            OP[xm(N,Hpc,Wp,BINS_NO, nx, ty, tx, bx)] = sCHp[bx];
            
    /**/
    // if(ly==0&&lx==0) if(ty==0&&tx==0) //dg
    // {
    //     printf("[%d %d] ", ty, tx); for (size bx = 0; bx < BINS_NO; ++bx) {printf("|"); printf("%f, ", OP[xm(N,Hp,Wp,BINS_NO, nx, ty, tx, bx)]);} printf("\n");
    // }
}

__global__ void kern_hist_equalize(mapel* OP, mapel* CH, size Hpe)
{
    extern __shared__ mapel exdbm[];
    // IMG_RES > META
    size Hb2p = blockDim.z;
    size nx =   blockIdx.x;                 size N =    gridDim.x;
    // size Hp =   gridDim.y * Hb2p;           
    size Wp =   gridDim.z;
    // IMG_RES > TILING
    size b2ty = threadIdx.z;
    size ty =   blockIdx.y * Hb2p + b2ty;   
    size tx =   blockIdx.z;             
    // IMG_RES > LOCAL (PATCH)
    size al =   threadIdx.x;
    size ay =   al / CELL_ADJ;
    size ax =   al % CELL_ADJ;
    size bx =   threadIdx.y;
    size ll =   al * BINS_NO + bx;
    // IMG_RES > GLOBAL > OUTPUT
    // size gol =   (ty*Wp + tx) * (t2ly*t2lx) + ll;

    if (ty < Hpe)
    {   
        // LOCALIZE & MERGE
        size lEHp = CELL_ADJ*CELL_ADJ*BINS_NO;
        mapel* sEHp = &exdbm[b2ty * lEHp];
        
        sEHp[xm(1,CELL_ADJ,CELL_ADJ,BINS_NO, 0, ay, ax, bx)]
            = CH[xm(N,Hpe+CELL_ADJ-1,Wp+CELL_ADJ-1,BINS_NO, nx, ty+ay, tx+ax, bx)];
        
        // REDUCE > SUMUP 1
        mapel* sumup = &exdbm[(Hb2p + b2ty) * lEHp];
        sumup[ll] = sEHp[ll];
        __syncthreads();

        for (size ar=log2round(lEHp); ar>0; ar/=2) 
        {
            if (ll < ar && ll + ar < lEHp) 
            {
                sumup[ll] += sumup[ll+ar];
            }
            __syncthreads();
        }        
        // EQUALIZE
        // if (ty==0&&tx==0&&ll==0) printf("%f s0\n", sumup[0]);//dg

        sEHp[ll] = min(sEHp[ll]/sumup[0], LOWE_NORMF);

        // __syncthreads(); if (ty==0&&tx==0&&ll==0) {float e=0; for (int i=0; i<lEHp; ++i) {e+=sEHp[i], printf("%f ", sEHp[i]);}; printf("| %f \n", e);} // dg
        
        // REDUCE > SUMUP 2 (duplicate to sumup 1)
        sumup[ll] = sEHp[ll];
        __syncthreads();

        for (size ar=log2round(lEHp); ar>0; ar/=2) 
        {
            if (ll < ar && ll + ar < lEHp) 
            {
                sumup[ll] += sumup[ll+ar];
                // if (ty==0&&tx==0&&ll==0) printf(">> %f\n", sEHp[0]);
            }
            __syncthreads();
        }
        // if (ty==0&&tx==0&&ll==0) printf("%f s0\n", sumup[0]);//dg
        // if (ty==0&&tx==0&&ll==0) printf("%f %f\n", sEHp[ll], sumup[0]);//dg

        // EQUALIZE
        sEHp[ll] = sEHp[ll]/sumup[0];
        
        // GLOBALIZE
        // printf("[%d, %d]\n", ty, tx);
        OP[xm(N,Hpe,Wp,lEHp, nx,ty,tx,ll)] = sEHp[ll];
        /**/
        // __syncthreads(); if (ty==0&&tx==0&&ll==0) {float e=0; for (int i=0; i<lEHp; ++i) {e+=sEHp[i], printf("%f ", sEHp[i]);}; printf("| %f \n", e);} // dg
        // if (ty==0&&tx==0&&ll==0) {printf("eh  "); for (int j=0;j<36;++j) printf("%.2f ", sEHp[j]); printf("\n    "); printf("\n");} //dg
    }
}

void gpu_hog(cv::Mat* OP, cv::Mat* IN, size N)
{
    // PARAMETERS
    // cv::Mat img; 
    cv::Mat imc[MAX_NO_CHANNELS]; 
    size    C = IN[0].channels();
    size    H = IN[0].rows;
    size    W = IN[0].cols;

    // INPUT
    size    lINnc = H * W ;
    size    soINnc = lINnc * sizeof(mapel);
    size    soIN = N * C * lINnc * sizeof(mapel);
    mapel*  dIN; cudaMalloc((void**)&dIN, soIN); 

    for (size nx=0; nx < N; ++nx)
    {
        cv::split(IN[nx], imc);
        for (size cx=0; cx < C; ++cx)
        {
            // cv::extractChannel(IN[nx], imc, cx);
            // printf("%f\n", ((float*)imc[cx].ptr())[0]); //dg
            // printf("%fv\n", ((float*)IN[nx].ptr())[cx]);
            // printf("%d\n", /**/ soIN);
            cudaMemcpy(&dIN[(nx*C+cx)*lINnc], imc[cx].ptr(), soINnc, cudaMemcpyHostToDevice);
        }        
    } 
    // POLAR_GRAD > OP
    size    soPG = N * 2 * H * W * sizeof(mapel);
    mapel*  dPG; cudaMalloc((void**)&dPG, soPG);

    // POLAR_GRAD > KERNELS
    size lIKx =     SOBEL_KSIZE * SOBEL_KSIZE;
    size soIKx =    lIKx * sizeof(mapel);
    size soIK  =    2 * soIKx;
    mapel* dIK; cudaMalloc((void**)&dIK, soIK); 
    // for (size i=0; i < 9; ++i) printf("%f, ", ((float*)SOBEL_KERNEL_H.ptr(0))[i]); printf("\n"); //dg
    cudaMemcpy(&dIK[0*lIKx], SOBEL_KERNEL_H.clone().ptr(0), soIKx, cudaMemcpyHostToDevice);
    cudaMemcpy(&dIK[1*lIKx], SOBEL_KERNEL_W.clone().ptr(0), soIKx, cudaMemcpyHostToDevice);

    // POLAR_GRAD > TILING
    size    Ht_pg = ceil((float)H/TILE_SIZE);
    size    Wt_pg = ceil((float)W/TILE_SIZE);

    size KB2 = SOBEL_KSIZE-1;
    size sosINn = C * (TILE_SIZE+KB2) * (TILE_SIZE+KB2) * sizeof(mapel);

    // POLAR_GRAD > RUN

     t0 = std::chrono::high_resolution_clock::now();
        
    kern_polar_grad<<<
        dim3(N, Ht_pg, Wt_pg), 
        dim3(TILE_SIZE, TILE_SIZE), 
        sosINn + soIK
    >>>(dPG, dIN, dIK, C, H, W);

    t1 = std::chrono::high_resolution_clock::now();
        td= t1 - t0;
        fprintf(stderr, "Polar grad execution time = %f\n", (float)td.count());

    // HIST_APPEND > PATCHING (duplicate to cpu_hog)
    size Hp =   floor((float)H/CELL_SIZE); size Hpm = max(Hp, 2);
    size Wp =   floor((float)W/CELL_SIZE); size Wpm = max(Wp, 2); 
    
    // HIST_APPEND > TILING (several full patches per block)
    size lCHredo = CELL_SIZE * CELL_SIZE;
    size Hb2p = max((int)floor((float)TILE_SIZE*TILE_SIZE/lCHredo), 1);
    size Hb = ceil((float)Hp/Hb2p);

    size sosCH = Hb2p  * lCHredo * BINS_NO * sizeof(mapel);

    // HIST_APPEND > OP
    size soCH = N * Hpm * Wpm * BINS_NO * sizeof(mapel);
    mapel* dCH; cudaMalloc((void**)&dCH, soCH); //cudaMemset(dCH, 0, soCH);

    // HIST_APPEND > RUN
    t1 = std::chrono::high_resolution_clock::now(); 

    kern_hist_append<<<
        dim3(N, Hb, Wp),
        dim3(CELL_SIZE, CELL_SIZE, Hb2p),
        sosCH
    >>>(dCH, dPG, Hp, H, W);

    td= t1 - t0;
    fprintf(stderr, "Hist Append execution time = %f\n", (float)td.count());

    // HIST_APPEND > PATCHING
    size Hpe = Hpm+1-CELL_ADJ;
    size Wpe = Wpm+1-CELL_ADJ;

    // HIST_EQALIZE > TILING
    size lEHp = (CELL_ADJ * CELL_ADJ * BINS_NO);
    size Hb2te = max((int)floor((float)TILE_SIZE*TILE_SIZE/lEHp), 1);
    size Hbe = ceil((float)Hpe/Hb2te);

    size sosEH = Hb2te * lEHp * sizeof(mapel);

    // HIST_EQALIZE > OP
    size lEHn = Hpe * Wpe * lEHp; 
    size soEHn = lEHn * sizeof(mapel);
    size soEH = N * lEHn * sizeof(mapel);
    // printf("%d", N); //dg
    mapel* dEH; cudaMalloc((void**)&dEH, soEH); 
    // return; //dg

    // HIST_EQALIZE > RUN

    t1 = std::chrono::high_resolution_clock::now();
    

    kern_hist_equalize<<<
        dim3(N, Hbe, Wpe),
        dim3(CELL_ADJ*CELL_ADJ, BINS_NO, Hb2te),
        sosEH + sosEH
    >>>(dEH, dCH, Hpe);

    td= t1 - t0;
    fprintf(stderr, "Total execution time = %f\n", (float)td.count());

    // printf("%d, %d, %d, %d", Hpe, Wpe, lEHp, lEHp);
    // return; //dg

    // OUTPUT
    mapel* hEHn = (mapel*)malloc(soEHn);
    for (size nx = 0; nx < N; ++nx)
    {
        cudaMemcpy(hEHn, &dEH[nx*lEHn], soEHn, cudaMemcpyDeviceToHost);
        OP[nx] = cv::Mat(lEHn, 1, mapel_cv, (void*)hEHn).clone();
    }

    // : > DEALLOC
    cudaFree(dIN);
    cudaFree(dIK);
    cudaFree(dPG);
    cudaFree(dCH);
    cudaFree(dEH);
    free(hEHn);
    /**/
}

__global__ void kern_svm(mapel* OP, mapel* HO, mapel* IQ, size M)
{
    size nx = blockIdx.x; 
    size N = gridDim.x;
    size Dt = gridDim.y;
    size bx = blockIdx.y;
    size lx = threadIdx.x;
    size b2lx = blockDim.x;
    size mx = bx * b2lx + lx;

    if (mx < M)
        // printf("%f ", IQ[mx]);
        HO[nx*M+mx] *= IQ[mx];
    __syncthreads();
    // if (lx==0) {for (int i=0; i<b2lx; ++i) printf("%f ", HO[nx*M+i]); printf("\n");} //dg

    for (size ar = log2round(b2lx); ar > 0; ar /= 2)
    {
        if (lx < ar && lx + ar < b2lx && mx + ar < M && mx < M) 
        {
            HO[nx*M+mx] += HO[nx*M+(mx+ar)];
            // printf("%d %d %d : %d %d %d; %f\n", nx, mx, lx, ar, mx + ar, lx+ar, HO[1]);
        }
        __syncthreads();
    }

    if(lx==0 && mx < M) 
    {
        OP[nx*Dt+bx] = HO[nx*M+mx];
        // printf("%d %f\n", nx*M+mx, HO[nx*M+mx]);//dg
    }
    // __syncthreads();
}

void gpu_svm(mapel* OP, cv::Mat* HO, size N, mapel* IQ, size M, mapel IQ_bias)
{
    // INPUT > HOG
    size lHOn = M;
    size soHOn = M * sizeof(mapel);
    size soHO = N * soHOn;
    mapel* dHO; cudaMalloc((void**)&dHO, soHO);

    cv::Mat imc;
    for (size nx = 0; nx < N; ++nx)
    {
        imc = HO[nx];
        imc.convertTo(imc, mapel_cv);
        cudaMemcpy(&dHO[nx*lHOn], imc.ptr(), soHOn, cudaMemcpyHostToDevice);
    }
    
    // INPUT > WEIGHTS
    size soIQ = M * sizeof(mapel);
    mapel* dIQ; cudaMalloc((void**)&dIQ, soIQ);
    cudaMemcpy(dIQ, IQ, soIQ, cudaMemcpyHostToDevice);

    // TILING
    size Dl = (TILE_SIZE*TILE_SIZE);
    size Dt = ceil((float)M/Dl);

    // DETECTIONS
    size soOP = N * Dt * sizeof(mapel);
    mapel* dOP; cudaMalloc((void**)&dOP, soOP);
    cudaMemset(dOP, 0, soOP);

    mapel* hOP = (mapel*)malloc(soOP); // partials

    // RUN
    t1 = std::chrono::high_resolution_clock::now();

    kern_svm<<<dim3(N, Dt), dim3(Dl)>>>(dOP, dHO, dIQ, M);

    td= t1 - t0;
    fprintf(stderr, "SVM execution time = %f\n", (float)td.count());

    // OUTPUT 
    cudaMemcpy(hOP, dOP, soOP, cudaMemcpyDeviceToHost);

    for (size nx = 0; nx < N; ++nx)
    {
        OP[nx] = IQ_bias;
        for (size bx = 0; bx < Dt; ++bx)
        {
            OP[nx] += hOP[nx*Dt+bx];
            // printf("%f ", hOP[nx*Dt+bx]);//dg
        } //printf("\n");
    }

    // DEALLOC
    cudaFree(dHO);
    cudaFree(dIQ);
    cudaFree(dOP);
    free(hOP);
}

/* 
$env:Path += ";C:\Users\krism\Desktop\semestr9\ORwCUDA\opencv\build\x64\vc16\bin"
nvcc -o test .\zszc.cu -I ..\opencv\build\include\ -L ..\opencv\build\x64\vc16\lib\ -nvcc -o test .\zszc.cu -I ..\opencv\build\include\ -L ..\opencv\build\x64\vc16\lib\ -l opencv_world470 --run --run-args "gpu","d",".\classi_",".\db2\sample_false\1.png"
s = lambda v: sum([float(x) for x in v.split(' ')])
for fi in $(ls -1); do echo $fi; ../../test.exe gpu - $fi >> ../../hog_true; done
*/
namespace INFO {
    // REMARK: assert all(i,j in 0..N): C,H,W_i == C,H,W_j
    // REMARK - border type
    // REMARK - const NHW
    // REMARK - fcnptr
    int nina = 4; // no of non img_path arguments
    char usage[] = "Usage: ${exec} (cpu|gpu) (d ${weights_path})|(h *) {0..%d}${image_path}\n";
    char max_n[] = "Maximal batch size (%d) exceeded." 
        "Recompile program with \"MAX_NO_IMGS\" parameter modified.\n";
    char infound[] = " - Image not found, aborting.\n";
    char devndef[] = " - Device not defined, aborting.\n";
    char wnfound[] = " - Weights file not found, aborting.\n";
    char dimmmatch[] = "Hog processed data has different length (%d) than weights (%d).\n";
}

int main(int argc, char *argv[])
{
    // PARSE ARGS > NO
    if      (argc < INFO::nina)
        {fprintf(stderr, INFO::usage, MAX_NO_IMGS); return 1;}
    else if (argc > INFO::nina + MAX_NO_IMGS)
        {fprintf(stderr, INFO::max_n, MAX_NO_IMGS); return 1;}
    char* arg_device = argv[1];
    char* arg_mode = argv[2];
    char* arg_weights = argv[3];
    char** arg_imgs = &argv[INFO::nina];
    size N = argc - INFO::nina;

    // PARSE ARGS > DEVICE
    char enum_device;
    if      (!strcmp(arg_device, "cpu")) enum_device = 'c';
    else if (!strcmp(arg_device, "gpu")) enum_device = 'g';
    else {pdca(stderr, arg_device); fprintf(stderr, INFO::devndef); return 1;}

    // PARSE ARGS > MODE, WEIGHTS
    char enum_mode;
    if      (arg_mode[0] == 'h') enum_mode = 'h';
    else if (arg_mode[0] == 'p') enum_mode = 'p';
    else if (arg_mode[0] == 'd') enum_mode = 'd';
    // else if (arg_mode[0] == 'w') enum_mode = 'w';
    if (arg_mode[1] != 0) 
        {fprintf(stderr, INFO::usage, MAX_NO_IMGS); return 1;}

    // INIT > IMAGES
    cv::Mat IN[MAX_NO_IMGS];
    for (size nx = 0; nx < N; ++nx)
    {
        char* img_path = argv[INFO::nina + nx];
        cv::Mat img = cv::imread(img_path);
        if (img.empty()) {pdca(stderr, img_path); printf(INFO::infound); return 1;}

        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        img.convertTo(IN[nx], mapel_cv);
    }

    // INIT > SOBEL_KERNELS
    cv::Mat _tmp;
    _tmp = cv::Mat(SOBEL_KSIZE+1 /*CVHF*/, SOBEL_KSIZE, mapel_cv, cv::Scalar(0));
        *(mapel*)_tmp.ptr(0, 1) = -1; *(mapel*)_tmp.ptr(2, 1) = 1; 
        *(mapel*)_tmp.ptr(2, 2) = 0; // CVHF
        // *(mapel*)_tmp.ptr(1, 1) = 1; //dg
    SOBEL_KERNEL_H = _tmp.clone();
    _tmp = cv::Mat(SOBEL_KSIZE+1 /*CVHF*/, SOBEL_KSIZE, mapel_cv), cv::Scalar(0);
        *(mapel*)_tmp.ptr(1, 0) = -1; *(mapel*)_tmp.ptr(1, 2) = 1; 
            // *(mapel*)_tmp.ptr(1, 1) = 1; //dg
            *(mapel*)_tmp.ptr(2, 2) = 0; // CVHF
    SOBEL_KERNEL_W = _tmp.clone();

    // MODE > HOG
    if (enum_mode == 'h')
    {
        cv::Mat HOOP[MAX_NO_IMGS];

        for (size nx = 0; nx < N; ++nx)
            cv::resize(IN[nx], IN[nx], cv::Size(PERSON_CROP_W, PERSON_CROP_H));
        
        if      (enum_device == 'c') cpu_hog(HOOP, IN, N);
        else if (enum_device == 'g') gpu_hog(HOOP, IN, N);

        for (size nx = 0; nx < N; ++nx)
        {   
            for (size fx = 0; fx < HOOP[nx].rows; ++fx) 
                printf("%f,", *(float*)HOOP[nx].ptr(fx));
            printf("\n");
        }
        return 0;
    }

    // MODE > DETECT
    if (enum_mode == 'd')
    {
        // INIT > WEIGHTS
        size M = 0;
        mapel IQ[MAX_NO_WEIGHTS];
        mapel IQ_bias;

        // REMARK: MAX_NO_WEIGHTS
        FILE* weights_handler = fopen(arg_weights, "r");
        if (!weights_handler) 
            {pdca(stderr, arg_weights); fprintf(stderr, INFO::wnfound);; return 1;}
        fscanf(weights_handler, "%f,\n", &IQ_bias);
        while (fscanf(weights_handler, "%f,", &IQ[M++]) != EOF) {}
        --M;
        fclose(weights_handler);

        cv::Mat HOOP[MAX_NO_IMGS];
        mapel OP[MAX_NO_IMGS];

        for (size nx = 0; nx < N; ++nx)
            cv::resize(IN[nx], IN[nx], cv::Size(PERSON_CROP_W, PERSON_CROP_H));

        t0 = std::chrono::high_resolution_clock::now();

        if      (enum_device == 'c') cpu_hog(HOOP, IN, N); 
        else if (enum_device == 'g') gpu_hog(HOOP, IN, N);
            
        if (N > 0) if (HOOP[0].rows != M) 
            {fprintf(stderr, INFO::dimmmatch, HOOP[0].rows, M); return 1;}
        // M = 36*((PERSON_CROP_H-8)/8)*((PERSON_CROP_W-8)/8); //dg

        // printf("--------------------------%d %d %d\n", HOOP[0].rows, HOOP[0].cols, M);//dg
        

        if      (enum_device == 'c') cpu_svm(OP, HOOP, N, IQ, M, IQ_bias);
        else if (enum_device == 'g') gpu_svm(OP, HOOP, N, IQ, M, IQ_bias);

        // t0 = std::chrono::high_resolution_clock::now();
        t1 = std::chrono::high_resolution_clock::now();
        td= t1 - t0;
        fprintf(stderr, "Total execution time = %f\n", (float)td.count());

        for (size nx = 0; nx < N; ++nx)
        {
            pdca(stdout, arg_imgs[nx]);
            printf(": %.2f\n", OP[nx]);
        }
        return 0;
    }

    // MODE > PLAYGROUND
    // if (enum_mode == 'p')
    // {
        // MID > HOG
        // cv::Mat HOOP[MAX_NO_IMGS];
        // N = 1;
        // size M = 0;
        // mapel IQ[MAX_NO_WEIGHTS];
        // mapel IQ_bias;

        // // REMARK: MAX_NO_WEIGHTS
        // FILE* weights_handler = fopen(arg_weights, "r");
        // if (!weights_handler) 
        //     {pdca(stderr, arg_weights); fprintf(stderr, INFO::wnfound);; return 1;}
        // fscanf(weights_handler, "%f,\n", &IQ_bias);
        // while (fscanf(weights_handler, "%f,", &IQ[M++]) != EOF) {}
        // --M;
        // fclose(weights_handler);

        // cv::Mat imc; cv::extractChannel(IN[0], imc, 0);
        // // printf("arc "); for (int i = 0; i < 8; ++i) { for (int j=0;j<8;++j) printf("%.2f ", *(float*)imc.ptr(i,j)); printf("\n    ");} printf("\n"); //dg
        // gpu_hog(HOOP, IN, N);
        // M = HOOP[0].rows;
        // M = 1024;
        // mapel OP[10];


        // // RUN > CPU > HOG
        // gpu_hog(HOOP, IN, N);
        // N = 1;
        // HOOP[0] = cv::Mat(10000, 1, mapel_cv, cv::Scalar(1));
        // HOOP[1] = cv::Mat(10000, 1, mapel_cv, cv::Scalar(1));
        // mapel* IQ = (float*)HOOP[1].ptr();
        // mapel IQ_bias = 0;
        // size M = 22; 

        
        // printf("-------------------------------------------\n");
        // cpu_svm(OP, HOOP, N, IQ, M, IQ_bias);
        // printf("%f\n", OP[0]);
        // printf("-------------------------------------------\n");
        // gpu_svm(OP, HOOP, N, IQ, M, IQ_bias);
        // printf("%f\n", OP[0]);


        // printf("a5 "); //dg

        // printf("%d, %d\n", HOOP[0].rows, HOOP[0].cols);
        // printf("hog "); for (int i = 0; i < HOOP[0].size[0]/36; ++i) 
        //     {for (int j=0;j<36;++j) printf("%.2f ", *(float*)HOOP[0].ptr(i*36+j)); 
        //     printf("\n    ");} printf("\n"); //dg
    // }

    // MODE > WINDOW
    // if (enum_mode == 'w')
    // {
    //     // todo
    // }

    return 1;
}

/*
// printf("eeeeeeeeeeeeeeeee %d %d %d \n", N, Dt, Dl);//dg
    kern_svm<<<dim3(N, Dt, 1), dim3(Dl,1,1)>>>(dOP, dHO, dIQ, IQ_bias, M, 0);
    // for (int nx=0; nx<2; ++nx)
    //     kern_svm<<<dim3(1, Dt, 1), dim3(Dl,1,1)>>>(dOP, dHO, dIQ, IQ_bias, M, nx);


    // for (size nx = 0; nx < N; ++nx)
    //     for (size mx=0; mx < 8; ++mx)
    // {
    //         HO[nx].convertTo(HO[nx], mapel_cv);
    //         cudaMemcpy(&dHO[nx*M+mx], (float*)HO[nx].ptr(0,0), sizeof(mapel), cudaMemcpyHostToDevice);
    // }
    // cv::Mat imc;
    // for (size nx = 0; nx < N; ++nx) //dg
    // {
    //     imc = HO[nx].clone();
    //     *(float*)imc.ptr(0, 1) = 20;
    //     cudaMemcpy(&dHO[nx * lHOn], imc.ptr(), 8, 
    //         cudaMemcpyHostToDevice);
    // }
    // printf("M=%d %d\n", M, soHO);
    // for (size nx = 0; nx < N; ++nx) //dg
    //     {
    //         for (size i=0; i <36; ++i) printf("%.2f ", ((float*)HO[nx].ptr())[i]);
    //         printf("\n");
    //     }
    // mapel* aaa = (mapel*)malloc(N*M*sizeof(mapel));
    // cudaMemcpy(aaa, dHO, N*M*sizeof(mapel), cudaMemcpyDeviceToHost);
    // for (size nx = 0; nx < N; ++nx) //dg
    //     {
    //         for (size i=0; i <36; ++i) printf("%.2f ", aaa[M*nx+i]);
    //         printf("\n");
    //     }

__global__ void kern_svm(mapel* OP, mapel* HO, mapel* IQ, mapel IQ_bias, size M, size nsh)
{
    size nx = blockIdx.x + nsh; 
    size N = gridDim.x;
    size bx = blockIdx.y;
    size lx = threadIdx.x;
    size mx = bx * blockDim.y + lx;

    // if (mx==0&&nx==0) for (size nx = 0; nx < N; ++nx) //dg
    //     {
    //         for (size i=0; i <36; ++i) printf("%.2f ", HO[nx*M+i]);
    //         printf("a%d\n", nx);
    //     }__syncthreads();

    // HO[nx*M+mx] *= IQ[mx];
    __syncthreads();

    // if (mx==0&&nx==0) for (size nx = 0; nx < N; ++nx) //dg
    //     {
    //         for (size i=0; i <36; ++i) printf("%.2f ", HO[nx*M+i]);
    //         printf("b%d\n", nx);
    //     }__syncthreads();
    // mapel aaa = 0;
    // if (mx==0) {for(int r=0; r<M; ++r) aaa += HO[nx*M+r]; printf("|||%f|||", aaa); nsh=aaa;} 
    // __syncthreads();
    
    for (size ar = log2round(M); ar > 0; ar /= 2)
    {
        if (mx < ar && mx + ar < M) 
        {
            HO[nx*M+mx] += HO[nx*M+(mx+ar)];
            // if(mx < 16) printf("%d %d %d | %d %d %d\n", ar, mx, mx+ar, nx, nx*M+mx, nx*M+(mx+ar)); //dg
        }
        __syncthreads();
    }

    // if (mx==0&&nx==0) for (size nx = 0; nx < N; ++nx) //dg
    //     {
    //         for (size i=0; i <36; ++i) printf("%.2f ", HO[nx*M+i]);
    //         // printf("c%d\n", nx);
    //     }__syncthreads();

    if(mx==0) OP[nx] = HO[nx*M+0] + 0; //IQ_bias;
    __syncthreads();
}

    /*

    // launchKernelConvLayer(outputMaps, inputMaps, convFilters, N, M, C, H, W, K);
            
    // save outputs for the first image
    // int n = 0;
    // float *outputImage = (float *)malloc(H * W * sizeof(float));
    // for (int m = 0; m < M; ++m)
    // {
    //     for (int p = 0; p < Hout * Wout; ++p)
    //     {
    //         outputImage[p] = outputMaps[n * M * Hout * Wout + m * Hout * Wout + p] / 3;
    //     }

    //     char strInpuNoExt[256];
    //     strcpy(strInpuNoExt, argv[2]);
    //     strInpuNoExt[strlen(argv[2]) - 4] = '\0';
    //     char strMapIdx[8];
    //     sprintf(strMapIdx, "_0%d.ppm", m);
    //     writePPM(strcat(strInpuNoExt, strMapIdx), outputImage, Wout, Hout, 1);
    // }

    // free(outputImage);
    // free(outputMaps);
    // free(inputMaps);
    // free(inputImage);
    // free(convFilters);


    // size C = img.channels();
    // size H = img.rows;
    // size W = img.cols;
    // size soIN_c = H * W * sizeof(mapel); size so_IN = C * soIN_c;
    // mapel* INa = (mapel*)INm.data;  //todo: check

    // size Ht2p = floor((float)TILE_SIZE/CELL_SIZE); size Ht_ha = ceil((float)Hp/Ht2p);
    // size Wt2p = floor((float)TILE_SIZE/CELL_SIZE); size Wt_ha = ceil((float)Wp/Wt2p);


    // mapel ampc[MAX_NO_CHANNELS]; cudaMemset(ampc, 0, MAX_NO_CHANNELS * sizeof(mapel)); 
    // mapel arcc[MAX_NO_CHANNELS]; cudaMemset(arcc, 0, MAX_NO_CHANNELS * sizeof(mapel));


    if (!(  SOBEL_KERNEL_H.rows == SOBEL_KSIZE 
        ||  SOBEL_KERNEL_H.rows == SOBEL_KSIZE
        ||  SOBEL_KERNEL_H.rows == SOBEL_KSIZE
        ||  SOBEL_KERNEL_H.rows == SOBEL_KSIZE
        ))
    {return 1;} //todo: print

if      (wei_path[0] == '-' && wei_path[1] != 0)
    {
        FILE* hogout = fopen(&wei_path[1], "a");
        cv::Mat HOOP[MAX_NO_IMGS];

        // printf("a4 "); //dg

        for (size nx = 0; nx < N; ++nx)
            cv::resize(IN[nx], IN[nx], cv::Size(PERSON_CROP_W, PERSON_CROP_H));
        cpu_hog(HOOP, IN, N);
        size hlen = HOOP[0].rows;

        for (size nx = 0; nx < N; ++nx)
        {
            for (size fx = 0; fx < hlen; ++fx) 
                fprintf(hogout, "%f,", *(float*)HOOP[nx].ptr(fx));
            fprintf(hogout, "\n");
        }
        fclose(hogout);
        return 0;
    }
    else if (wei_path[0] == '-' && wei_path[1] == 0)
    {
        printf(INFO::usage);
        return 1;
    }

size ey = py-CELL_ADJ+1; 
        size ex = px-CELL_ADJ+1;
        size eloc = ey*Wp+ex; // map current patch loc to eh loc

        // mapel* EHc = &EH[eloc*lEHc];
        mapel eqel;
        mapel esc2 = 0;

        for     (size ay = 0; ay < min(CELL_ADJ, Hp-py); ++ay) 
            for (size ax = 0; ax < min(CELL_ADJ, Wp-px); ++ax) // (..., 2, 2, 2, 1)
        {
            // HEQ > FORWARD UPDATE EHs FOR CH
            size eloc_forward = (py+ay)*Wp + (px+ax);
            size bloc_forward = (CELL_ADJ-1-ay)*CELL_ADJ+(CELL_ADJ-1-ax);

            ES[eloc_forward] += CSc;
            // HEQ > EQUALIZE CURRENT EQPATCH (after (ay, ax)=(0,0) is summed up)
            for (size bx = 0; bx < BINS_NO; ++bx)
            {
                memcpy(&EH[(eloc_forward*lEHc) + (bloc_forward*BINS_NO) + bx], CHc)
                eqel = max(CHc[bx]/ES[eloc], LOWE_NORMF);
                EH[(eloc_forward*lEHc) + (bloc_forward*BINS_NO) + bx] = eqel;
                esc2 += eqel;
            }
            
            for (size bx = 0; bx < BINS_NO; ++bx)
            {
                EHc[(ay*CELL_ADJ+ax)*BINS_NO + bx] /= esc2;
                printf("%f ", EHc[(ay*CELL_ADJ+ax)*BINS_NO + bx]); 
            }
            printf("a8\n");

            // printf("%d\n", bx);
            // for (int i = 0; i < 20; ++i) printf("%f ", *(float*)arcb.ptr(i/8, i%8)); printf("\n");
            // for (int i = 0; i < 20; ++i) printf("%f ", *(float*)arcp.ptr(i/8, i%8)); printf("\n");
            // for (int i = 0; i < 20; ++i) printf("%f ", *(float*)b0mlt.ptr(i/8, i%8)); printf("\n");
            // for (int i = 0; i < 20; ++i) printf("%f ", *(float*)b1mlt.ptr(i/8, i%8)); printf("\n");


        // }
        // for     (size ay = 0; ay < min(CELL_ADJ, Hp-py); ++ay)
        //     for (size ax = 0; ax < min(CELL_ADJ, Wp-px); ++ax)

            // size eloc_backward = (((py-ay)*Wp+(px-ax)))
            // size eloc_forward =  (((py+ay)*Wp+(px+ax))*(CELL_ADJ*CELL_ADJ*BINS_NO)) //for top-left
                // + ((ay*CELL_ADJ+ax)*BINS_NO)


        // for (int i = 0; i < 16; ++i) printf("%lf ", *ampc[cx].ptr(i)); printf("\n");
        // for (int i = 0; i < 16; ++i) printf("%d ", *h_gtc.ptr(i)); printf("\n");
        // for (int i = 0; i < 20; ++i) printf("%f ", *(float*)sbcy[cx].ptr(i/8, i%8)); printf("\n");
        // for (int i = 0; i < 20; ++i) printf("%f ", *(float*)sbcx[cx].ptr(i/8, i%8)); printf("\n");


    // size soCS = Hp * Wp * sizeof(mapel); size soCH = soCS * BINS_NO;
    // mapel* CH = (mapel*)malloc(soCH); // cell hists
    // mapel* CS = (mapel*)malloc(soCS); // cell sums


            // for (size bx = 0; bx < BINS_NO; ++bx)
            // {
            //     hist[bx] = min(hist[bx]/cums, .2);
            //     cumc += hist[bx];
            // }
            // for (size bx = 0; bx < BINS_NO; ++bx)
            //     hist[bx] /= cumc;

// HIST EQUALIZE 2/2
    for     (size py = 0; py < (Hp-1); ++py)
        for (size px = 0; px < (Wp-1); ++px)
    {
        ploc = py * Wp + px;
        mapel* EH_loc = &EH[ploc * lEHc];
        mapel  ES_loc = ES[ploc];
        for         (size ay = 0; ay < CELL_ADJ; ++ay)
            for     (size ax = 0; ay < CELL_ADJ; ++ax)
                for (size bx = 0; bx < BINS_NO;  ++bx)
            EQ_loc[(ay * CELL_ADJ + ax) * BINS_NO + bins] 
                =   CH[]

ploc = py * Wp + px;
        mapel* CH_loc = &CH[ploc * BINS_NO];
        for (size bx = 0; bx < BINS_NO; ++bx) CH_loc[bx] = 0;
        CS[ploc] = 0;

        for (size bx = 0; bx < BINS_NO; ++bx)
        {
            arcb = (bx <= arcp) & (arcp <= (bx+1));
            cv::multiply(arcb * (bx+1 - arcp),  ampp, b0mlt); b0sum = 
            cv::multiply(arcb * (arcp - bx),    ampp, b1mlt);

            CH_loc[bx]              += cv::sum(b0mlt)[0];
            CH_loc[(bx+1)%BINS_NO]  += cv::sum(b1mlt)[0];
        }
        for (size bx = 0; bx < BINS_NO; ++bx) CS[ploc] += CH_loc[bx];


    cv::Mat ampp; cv::Mat arcp; 
    cv::Mat arcb; cv::Mat b0mlt; cv::Mat b1mlt;


            ampp = amp( cv::Range(CELL_SIZE * py, CELL_SIZE * (py+1)),
                        cv::Range(CELL_SIZE * px, CELL_SIZE * (px+1)));
            arcp = arc( cv::Range(CELL_SIZE * py, CELL_SIZE * (py+1)),
                        cv::Range(CELL_SIZE * px, CELL_SIZE * (px+1)));
            arcp = (arcp + CV_PI) / (CV_2PI / BINS_NO); 

            for (size bx = 0; bx < BINS_NO; ++bx)
            {
                arcb = (bx <= arcp) & (arcp <= (bx+1));
                cv::multiply(arcb * (bx+1 - arcp),  ampp, b0mlt);
                cv::multiply(arcb * (arcp - bx),    ampp, b1mlt);

                hist[bx]                += cv::sum(b0mlt)[0];
                hist[(bx+1)%BINS_NO]    += cv::sum(b1mlt)[0];
                cums                    += cv::sum(b0mlt)[0];
                cums                    += cv::sum(b1mlt)[0];
            }

*/