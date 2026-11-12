#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cmath>
#include <string>
#include <thread>

#define CUDA_CHECK(x) do { cudaError_t err=(x); if(err!=cudaSuccess){std::cerr<<"CUDA error: "<<cudaGetErrorString(err)<<" @ "<<__FILE__<<":"<<__LINE__<<std::endl; std::exit(EXIT_FAILURE);} } while(0)

constexpr int Q=9, ITERATIONS_DEFAULT=12000;
constexpr float OMEGA_DEFAULT=1.8f, U_IN_DEFAULT=0.04f;
constexpr float CS2 = 1.0f / 3.0f;

constexpr float HW[Q]={4.f/9.f,1.f/9.f,1.f/9.f,1.f/9.f,1.f/9.f,1.f/36.f,1.f/36.f,1.f/36.f,1.f/36.f};
__constant__ float W[Q]={4.f/9.f,1.f/9.f,1.f/9.f,1.f/9.f,1.f/9.f,1.f/36.f,1.f/36.f,1.f/36.f,1.f/36.f};
__constant__ int CX[Q]={0,1,0,-1,0,1,-1,-1,1}, CY[Q]={0,0,1,0,-1,1,1,-1,-1}, OPP[Q]={0,3,4,1,2,7,8,5,6};

__device__ __forceinline__ float feq(int i,float rho,float ux,float uy){float cu=3.f*(CX[i]*ux+CY[i]*uy), uu=ux*ux+uy*uy; return W[i]*rho*(1.f+cu+0.5f*cu*cu-1.5f*uu);} 

__global__ void lbm_step_pull(const float* fIn,float* fOut,const unsigned char* obstacle,int nx,int ny,float omega,float uIn,int inflowDir){
    int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y; 
    if(x>=nx||y>=ny) return; 
    
    int idx = y*nx+x;
    int cells = nx * ny;

    if(obstacle[idx]){for(int i=0;i<Q;++i) fOut[i*cells+idx]=fIn[OPP[i]*cells+idx]; return;}

    bool isTop = (y==ny-1), isBottom=(y==0), isLeft=(x==0), isRight=(x==nx-1);
    int outDir = (inflowDir==0?1:(inflowDir==1?0:(inflowDir==2?3:2)));

    bool isInflow = (inflowDir==0 && isLeft) || (inflowDir==1 && isRight) || (inflowDir==2 && isTop) || (inflowDir==3 && isBottom);
    bool isOutflow= (outDir==0 && isLeft)   || (outDir==1 && isRight)   || (outDir==2 && isTop)   || (outDir==3 && isBottom);

    bool isWall = (isTop || isBottom || isLeft || isRight) && !isInflow && !isOutflow;
    if(isWall){ for(int i=0;i<Q;++i) fOut[i*cells+idx]=fIn[OPP[i]*cells+idx]; return; }

    float fLoc[Q];

    if(isInflow){
        float ux=0.f, uy=0.f;
        if(inflowDir==0) ux = uIn;       
        if(inflowDir==1) ux = -uIn;      
        if(inflowDir==2) uy = -uIn;      
        if(inflowDir==3) uy = uIn;       
        for(int i=0;i<Q;++i) fLoc[i]=feq(i,1.0f,ux,uy);
    }
    else if(isOutflow){
        int sx=x, sy=y;
        if(outDir==0) sx = x+1;
        if(outDir==1) sx = x-1;
        if(outDir==2) sy = y-1;
        if(outDir==3) sy = y+1;
        int sidx = sy*nx + sx;
        for(int i=0;i<Q;++i) fLoc[i]=fIn[i*cells+sidx];
    }
    else {
        for(int i=0;i<Q;++i){
            int sx=x-CX[i], sy=y-CY[i];
            int sidx=sy*nx+sx;
            fLoc[i]=(sx<0||sx>=nx||sy<0||sy>=ny||obstacle[sidx]) ? fIn[OPP[i]*cells+idx] : fIn[i*cells+sidx];
        }
    }

    float rho=0,ux=0,uy=0; 
    for(int i=0;i<Q;++i){float fi=fLoc[i]; rho+=fi; ux+=fi*CX[i]; uy+=fi*CY[i];}
    if(!isfinite(rho)||rho<1e-8f){rho=1;ux=uy=0;} else {ux/=rho; uy/=rho;}
    
    for(int i=0;i<Q;++i){
        float fe=feq(i,rho,ux,uy); 
        float out=fLoc[i]-omega*(fLoc[i]-fe); 
        fOut[i*cells+idx]=isfinite(out)?out:fe;
    }
}

__global__ void compute_velocity_mag(const float* f,const unsigned char* obstacle,float* vel,int nx,int ny){
    int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y; 
    if(x>=nx||y>=ny) return; 
    
    int idx=y*nx+x; 
    int cells=nx*ny;
    
    if(obstacle[idx]){vel[idx]=0; return;}
    
    float rho=0,ux=0,uy=0; 
    for(int i=0;i<Q;++i){float fi=f[i*cells+idx]; rho+=fi; ux+=fi*CX[i]; uy+=fi*CY[i];}
    if(!isfinite(rho)||rho<1e-8f){vel[idx]=0; return;} ux/=rho; uy/=rho; float v=sqrtf(ux*ux+uy*uy); vel[idx]=isfinite(v)?v:0;
}

static bool load_obstacle_csv(const std::string& path,std::vector<unsigned char>& obstacle,int nx,int ny){
    std::ifstream in(path); if(!in.is_open()){std::cerr<<"Cannot open obstacle CSV: "<<path<<std::endl; return false;} std::fill(obstacle.begin(),obstacle.end(),0);
    std::string line; std::getline(in,line); if(line.find("x")==std::string::npos) in.seekg(0); int count=0;
    while(std::getline(in,line)){ if(line.empty()) continue; std::stringstream ss(line); std::string xs,ys,os; if(!std::getline(ss,xs,',')) continue; if(!std::getline(ss,ys,',')) continue; if(!std::getline(ss,os,',')) continue;
        int x=std::stoi(xs), y=std::stoi(ys), occ=std::stoi(os); if(x>=0&&x<nx&&y>=0&&y<ny){obstacle[y*nx+x]=(occ!=0)?1:0; count+=(occ!=0);} }
    std::cout<<"Loaded obstacle mask: "<<path<<" | occupied="<<count<<"/"<<(nx*ny)<<std::endl; return true;
}

// Oude CSV functie voor de eind-plot
static void write_velocity_csv(const std::string& path,const std::vector<float>& vel,int nx,int ny){ 
    FILE *f = fopen(path.c_str(), "w");
    if (!f) { std::cerr << "Kon output bestand niet openen: " << path << std::endl; return; }
    fprintf(f, "x,y,vel\n");
    for(int y=0; y<ny; ++y) {
        for(int x=0; x<nx; ++x) {
            fprintf(f, "%d,%d,%.5f\n", x, y, vel[y*nx+x]);
        }
    }
    fclose(f);
}

// Nieuwe binaire functie voor de animatie-frames
static void write_velocity_bin(const std::string& path, const std::vector<float>& vel){ 
    FILE *f = fopen(path.c_str(), "wb");
    if (!f) return;
    fwrite(vel.data(), sizeof(float), vel.size(), f);
    fclose(f);
}

int main(int argc, char** argv) {
    std::string maskPath = "mask.csv", outPath = "vel.csv";
    int nx = 1024, ny = 512;
    float uIn = U_IN_DEFAULT;
    float omega = OMEGA_DEFAULT;
    int iters = ITERATIONS_DEFAULT;
    int inflowDir = 1;
    int saveInterval = iters;

    if(argc >= 3) { maskPath = argv[1]; outPath = argv[2]; }
    if(argc >= 5) { nx = std::max(1, std::stoi(argv[3])); ny = std::max(1, std::stoi(argv[4])); }
    if(argc >= 6) uIn = std::stof(argv[5]);
    if(argc >= 7) iters = std::max(1, std::stoi(argv[6]));
    if(argc >= 8) inflowDir = std::max(0, std::min(3, std::stoi(argv[7])));
    if(argc >= 9) saveInterval = std::max(1, std::stoi(argv[8]));
    if(argc >= 10) omega = std::max(0.51f, std::min(1.99f, std::stof(argv[9])));

    float tau = 1.0f / omega;
    float nu_lbm = CS2 * (tau - 0.5f);

    int cells = nx * ny;
    size_t distBytes = size_t(cells) * Q * sizeof(float);
    size_t obsBytes  = size_t(cells) * sizeof(unsigned char);
    size_t velBytes  = size_t(cells) * sizeof(float);

    std::vector<float> h_f0(cells * Q), h_f1(cells * Q), h_vel(cells); 
    std::vector<unsigned char> h_obstacle(cells, 0);
    
    if(!load_obstacle_csv(maskPath, h_obstacle, nx, ny)) return 1;
    
    for(int idx = 0; idx < cells; ++idx) {
        for(int i = 0; i < Q; ++i) { 
            h_f0[i * cells + idx] = HW[i]; 
            h_f1[i * cells + idx] = HW[i]; 
        }
    }

    float *d_fA=nullptr,*d_fB=nullptr,*d_vel=nullptr; unsigned char* d_obstacle=nullptr;
    CUDA_CHECK(cudaMalloc(&d_fA,distBytes)); CUDA_CHECK(cudaMalloc(&d_fB,distBytes)); CUDA_CHECK(cudaMalloc(&d_obstacle,obsBytes)); CUDA_CHECK(cudaMalloc(&d_vel,velBytes));
    CUDA_CHECK(cudaMemcpy(d_fA,h_f0.data(),distBytes,cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(d_fB,h_f1.data(),distBytes,cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(d_obstacle,h_obstacle.data(),obsBytes,cudaMemcpyHostToDevice));

    dim3 block(16,16), grid((nx+15)/16,(ny+15)/16);
    auto t0=std::chrono::high_resolution_clock::now();
    
    std::thread writer_thread;
    
    std::string baseOut = outPath.substr(0, outPath.rfind('.'));

    for(int it=1; it<=iters; ++it){
        lbm_step_pull<<<grid,block>>>(d_fA,d_fB,d_obstacle,nx,ny,omega,uIn,inflowDir);
        std::swap(d_fA,d_fB);

        if (it % saveInterval == 0 || it == iters) {
            compute_velocity_mag<<<grid,block>>>(d_fA,d_obstacle,d_vel,nx,ny);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            // B2 FIX: cudaMemcpy BEFORE join() so the memcpy overlaps with previous disk I/O
            CUDA_CHECK(cudaMemcpy(h_vel.data(),d_vel,velBytes,cudaMemcpyDeviceToHost));

            char framePath[256];
            snprintf(framePath, sizeof(framePath), "%s_iter%06d.bin", baseOut.c_str(), it);

            if (writer_thread.joinable()) writer_thread.join(); // join after memcpy, not before

            std::vector<float> h_vel_copy = h_vel;
            std::string path_copy = framePath;
            writer_thread = std::thread([path_copy, h_vel_copy]() {
                write_velocity_bin(path_copy, h_vel_copy);
            });

            int progress = (int)((float)it / iters * 100.0f);
            int barWidth = 40;
            std::cout << "\r   [";
            int pos = barWidth * progress / 100;
            for (int i = 0; i < barWidth; ++i) {
                if (i < pos) std::cout << "=";
                else if (i == pos) std::cout << ">";
                else std::cout << " ";
            }
            std::cout << "] " << progress << "% (" << it << "/" << iters << " iters)" << std::flush;
        }
    }
    std::cout << "\n";

    if (writer_thread.joinable()) writer_thread.join();

    auto t1=std::chrono::high_resolution_clock::now();

    // B1 FIX: h_vel already holds the final frame from the loop above (it==iters case).
    // No need to re-launch compute_velocity_mag; just write the CSV from existing h_vel.
    write_velocity_csv(outPath,h_vel,nx,ny);

    std::chrono::duration<double> dt=t1-t0;
    std::cout<<"STL-mask GPU simulatie klaar. Tijd: "<<dt.count()<<" s\n";
    std::cout<<"u_in="<<uIn<<" | iterations="<<iters<<" | inflowDir="<<inflowDir<<" (0:left,1:right,2:top,3:bottom)\n";
    std::cout<<"LBM-parameters: omega="<<omega<<" | tau="<<tau<<" | nu_lbm="<<nu_lbm<<"\n";
    std::cout<<"Output geschreven: "<<outPath<<std::endl;

    CUDA_CHECK(cudaFree(d_fA)); CUDA_CHECK(cudaFree(d_fB)); CUDA_CHECK(cudaFree(d_obstacle)); CUDA_CHECK(cudaFree(d_vel));
    return 0;
}