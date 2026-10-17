#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define HIP_OK(x) do { hipError_t e=(x); if(e!=hipSuccess){std::fprintf(stderr,"HIP error %s:%d: %s\n",__FILE__,__LINE__,hipGetErrorString(e));std::exit(2);} } while(0)

struct Input { double tau,sigma,psi,a,b,f0,v0,L,eta,prev,tol; int maxit,actual,converged; };
struct Output { double v,mu,g; int iterations,converged; };

__device__ __forceinline__ double asinh_equivalent(double x) { return log(x+sqrt(x*x+1.0)); }

__global__ void original_kernel(const Input* in,double* v,double* mu,double* g,int* iterations,int* converged,int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return; Input x=in[i];
  double z=fmax(x.prev,1e-45),C=.5/x.v0*exp((x.f0+x.b*log(x.v0*x.psi/x.L))/x.a),inv_eta=1.0/x.eta;
  int count=0; bool done=false;
  for(int k=0;k<60;k++){
    bool evaluating=!done&&k<x.maxit; if(!evaluating)continue;
    double q=z*C,s=sqrt(q*q+1.0),m=x.a*log(q+s),d=x.a*C/s,r=(x.tau-x.sigma*m)*inv_eta-z;
    count++; if(fabs(r)<x.tol){done=true;continue;}
    z=fmax(1e-45,z-r/(-(x.sigma*inv_eta)*d-1.0));
  }
  double q=z*C,m=x.a*asinh_equivalent(q);
  v[i]=z;mu[i]=m;g[i]=(x.tau-x.sigma*m)*inv_eta-z;iterations[i]=count;converged[i]=done?1:0;
}

__global__ void fixed_replay_kernel(const Input* in,double* v,double* mu,double* g,int* iterations,int* converged,int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return; Input x=in[i];
  double z=fmax(x.prev,1e-45),C=.5/x.v0*exp((x.f0+x.b*log(x.v0*x.psi/x.L))/x.a),inv_eta=1.0/x.eta;
  for(int k=0;k<60;k++){
    bool evaluating=k<x.actual; if(!evaluating)continue;
    double q=z*C,s=sqrt(q*q+1.0),m=x.a*log(q+s),d=x.a*C/s,r=(x.tau-x.sigma*m)*inv_eta-z;
    bool last=k+1==x.actual,update=!last||!x.converged;
    if(update)z=fmax(1e-45,z-r/(-(x.sigma*inv_eta)*d-1.0));
  }
  double q=z*C,m=x.a*asinh_equivalent(q);
  v[i]=z;mu[i]=m;g[i]=(x.tau-x.sigma*m)*inv_eta-z;iterations[i]=x.actual;converged[i]=x.converged;
}

Output cpu(Input x){
  double z=std::max(x.prev,1e-45),C=.5/x.v0*std::exp((x.f0+x.b*std::log(x.v0*x.psi/x.L))/x.a),inv_eta=1.0/x.eta;
  int count=0;bool done=false;
  for(int k=0;k<60;k++){
    if(done||k>=x.maxit)continue;
    double q=z*C,s=std::sqrt(q*q+1.0),m=x.a*std::log(q+s),d=x.a*C/s,r=(x.tau-x.sigma*m)*inv_eta-z;
    count++;if(std::fabs(r)<x.tol){done=true;continue;}z=std::max(1e-45,z-r/(-(x.sigma*inv_eta)*d-1.0));
  }
  double q=z*C,m=x.a*std::asinh(q);return{z,m,(x.tau-x.sigma*m)*inv_eta-z,count,done?1:0};
}

double percentile(std::vector<double> x,double q){std::sort(x.begin(),x.end());double p=(x.size()-1)*q;size_t lo=(size_t)p,hi=std::min(lo+1,x.size()-1);return x[lo]*(hi-p)+x[hi]*(p-lo);}
template<class Launch> std::vector<double> measure(Launch launch,hipStream_t stream,int warmup,int repeat){
  for(int i=0;i<warmup;i++)launch();HIP_OK(hipStreamSynchronize(stream));std::vector<double> values;
  for(int i=0;i<repeat;i++){hipEvent_t a,b;HIP_OK(hipEventCreate(&a));HIP_OK(hipEventCreate(&b));HIP_OK(hipEventRecord(a,stream));launch();HIP_OK(hipEventRecord(b,stream));HIP_OK(hipEventSynchronize(b));float ms;HIP_OK(hipEventElapsedTime(&ms,a,b));values.push_back(ms*1000.);HIP_OK(hipEventDestroy(a));HIP_OK(hipEventDestroy(b));}return values;
}
void emit(const char*variant,const std::vector<double>& values){
  std::printf("summary variant=%s median_us=%.9f p5_us=%.9f p95_us=%.9f\n",variant,percentile(values,.5),percentile(values,.05),percentile(values,.95));
  for(size_t i=0;i<values.size();i++)std::printf("sample variant=%s index=%zu us=%.9f\n",variant,i,values[i]);
  return;
}

int main(int argc,char**argv){
  int n=argc>1?std::atoi(argv[1]):65537,warmup=argc>2?std::atoi(argv[2]):10,repeat=argc>3?std::atoi(argv[3]):30;
  const Input base[]={
    {8e6,1e7,.2,.01,.015,.6,1e-6,.02,4e6,1e-6,1e-8,60,8,1},
    {5.8e6,1e7,2e4,.012,.018,.6,1e-6,.02,4e6,0,1e-8,60,22,1},
    {1e5,1e7,2e4,.012,.018,.6,1e-6,.02,4e6,1e-12,1e-8,60,6,1},
    {1.2e7,1e7,.02,.008,.012,.58,1e-6,.01,2e6,1,1e-8,60,4,1},
    {2e4,1e3,2e4,.01,.015,.6,1e-6,.02,4e6,1e-6,1e-8,60,4,1},
    {7e6,1e7,.2,.004,.018,.6,1e-6,.01,3e6,1e-5,1e-8,60,6,1},
    {8e6,1e7,.2,.01,.015,.6,1e-6,.02,4e6,1e-4,1e-8,60,6,1},
    {8e6,1e7,.2,.01,.015,.6,1e-6,.02,4e6,1e-45,1e-8,1,1,0},
    {6.5e6,9e6,.05,.009,.014,.59,1e-6,1e-4,3.5e6,1e-3,1e-8,60,6,1}};
  std::vector<Input> host(n);for(int i=0;i<n;i++)host[i]=base[i%9];Input* input;double *v,*mu,*g;int *iterations,*converged;
  HIP_OK(hipMalloc(&input,(size_t)n*sizeof(Input)));HIP_OK(hipMalloc(&v,(size_t)n*sizeof(double)));HIP_OK(hipMalloc(&mu,(size_t)n*sizeof(double)));HIP_OK(hipMalloc(&g,(size_t)n*sizeof(double)));HIP_OK(hipMalloc(&iterations,(size_t)n*sizeof(int)));HIP_OK(hipMalloc(&converged,(size_t)n*sizeof(int)));HIP_OK(hipMemcpy(input,host.data(),(size_t)n*sizeof(Input),hipMemcpyHostToDevice));
  hipStream_t stream;HIP_OK(hipStreamCreate(&stream));dim3 block(256),grid((n+255)/256);
  auto original=[&](){original_kernel<<<grid,block,0,stream>>>(input,v,mu,g,iterations,converged,n);};
  auto replay=[&](){fixed_replay_kernel<<<grid,block,0,stream>>>(input,v,mu,g,iterations,converged,n);};
  original();HIP_OK(hipStreamSynchronize(stream));std::vector<double>rv(n),rm(n),rg(n),xv(n),xm(n),xg(n);std::vector<int>ri(n),rc(n),xi(n),xc(n);HIP_OK(hipMemcpy(rv.data(),v,(size_t)n*8,hipMemcpyDeviceToHost));HIP_OK(hipMemcpy(rm.data(),mu,(size_t)n*8,hipMemcpyDeviceToHost));HIP_OK(hipMemcpy(rg.data(),g,(size_t)n*8,hipMemcpyDeviceToHost));HIP_OK(hipMemcpy(ri.data(),iterations,(size_t)n*4,hipMemcpyDeviceToHost));HIP_OK(hipMemcpy(rc.data(),converged,(size_t)n*4,hipMemcpyDeviceToHost));
  replay();HIP_OK(hipStreamSynchronize(stream));HIP_OK(hipMemcpy(xv.data(),v,(size_t)n*8,hipMemcpyDeviceToHost));HIP_OK(hipMemcpy(xm.data(),mu,(size_t)n*8,hipMemcpyDeviceToHost));HIP_OK(hipMemcpy(xg.data(),g,(size_t)n*8,hipMemcpyDeviceToHost));HIP_OK(hipMemcpy(xi.data(),iterations,(size_t)n*4,hipMemcpyDeviceToHost));HIP_OK(hipMemcpy(xc.data(),converged,(size_t)n*4,hipMemcpyDeviceToHost));
  double max_error=0.;int failures=0;for(int i=0;i<n;i++){Output ref=cpu(host[i]);max_error=std::max(max_error,std::max({std::fabs(rv[i]-ref.v),std::fabs(rm[i]-ref.mu),std::fabs(rg[i]-ref.g),std::fabs(xv[i]-ref.v),std::fabs(xm[i]-ref.mu),std::fabs(xg[i]-ref.g)}));failures+=(ri[i]!=ref.iterations||rc[i]!=ref.converged||xi[i]!=ref.iterations||xc[i]!=ref.converged||std::fabs(rv[i]-ref.v)>1e-10||std::fabs(xv[i]-ref.v)>1e-10);}
  std::printf("correctness max_abs_error=%.17g failures=%d status=%s\n",max_error,failures,failures?"FAIL":"PASS");if(failures)return 3;
  auto pre=measure(original,stream,warmup,repeat);auto fixed=measure(replay,stream,warmup,repeat);auto restore=measure(original,stream,warmup,repeat);emit("original_pre",pre);emit("fixed_iteration_replay",fixed);emit("original_restore",restore);double ratio=percentile(restore,.5)/percentile(pre,.5);std::printf("reversibility restore_over_pre=%.9f status=%s\n",ratio,(ratio>=.9&&ratio<=1.1)?"PASS":"FAIL");if(ratio<.9||ratio>1.1)return 4;std::printf("RSF_C3_FIXED_REPLAY_NATIVE_V2_PASS\n");return 0;
}
