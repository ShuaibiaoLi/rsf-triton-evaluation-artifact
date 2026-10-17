#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#define HIP_OK(x) do { auto e=(x); if(e!=hipSuccess){std::fprintf(stderr,"HIP:%s\n",hipGetErrorString(e));std::exit(2);} } while(0)

struct Matrix { int rows=0, cols=0, declared=0; bool symmetric=false; std::vector<int> rp, ci; std::vector<double> val; };

Matrix read_mtx(const std::string& path) {
  std::ifstream input(path); if (!input) { std::fprintf(stderr,"cannot open %s\n",path.c_str()); std::exit(2); }
  Matrix matrix; std::string line; std::getline(input,line); matrix.symmetric=line.find("symmetric")!=std::string::npos||line.find("hermitian")!=std::string::npos;
  do { std::getline(input,line); } while (!line.empty() && line[0]=='%');
  std::istringstream dimensions(line); dimensions>>matrix.rows>>matrix.cols>>matrix.declared;
  std::vector<std::tuple<int,int,double>> entries; int row,column,parsed=0; double value;
  while(std::getline(input,line)){if(line.empty()||line[0]=='%')continue;std::istringstream item(line);item>>row>>column;if(!(item>>value))value=1.0;--row;--column;++parsed;entries.emplace_back(row,column,value);if(matrix.symmetric&&row!=column)entries.emplace_back(column,row,value);}
  if(parsed!=matrix.declared){std::fprintf(stderr,"declared entries mismatch\n");std::exit(2);}
  std::sort(entries.begin(),entries.end(),[](const auto&a,const auto&b){return std::get<0>(a)==std::get<0>(b)?std::get<1>(a)<std::get<1>(b):std::get<0>(a)<std::get<0>(b);});
  matrix.rp.assign(matrix.rows+1,0);for(auto [r,c,v]:entries){matrix.rp[r+1]++;matrix.ci.push_back(c);matrix.val.push_back(v);}for(int r=0;r<matrix.rows;r++)matrix.rp[r+1]+=matrix.rp[r];return matrix;
}

template<int BLOCK_SIZE,int MIN_BLOCKS>
__global__ __launch_bounds__(BLOCK_SIZE,MIN_BLOCKS) void one_thread_per_row(const int* rp,const int* ci,const double* val,const double* x,double* y,int rows){int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=rows)return;double sum=0.0;for(int p=rp[row];p<rp[row+1];p++)sum+=val[p]*x[ci[p]];y[row]=sum;}

__global__ void cooperative_row(const int* rp,const int* ci,const double* val,const double* x,double* y,int rows,int groups_per_wave){
  int width=warpSize/groups_per_wave;int lane=threadIdx.x%width;int row=(blockIdx.x*blockDim.x+threadIdx.x)/width;if(row>=rows)return;double sum=0.0;
  for(int p=rp[row]+lane;p<rp[row+1];p+=width)sum+=val[p]*x[ci[p]];
  for(int offset=width/2;offset>0;offset/=2)sum+=__shfl_down(sum,offset,width);
  if(lane==0)y[row]=sum;
}

double quantile(std::vector<double> values,double fraction){std::sort(values.begin(),values.end());double position=(values.size()-1)*fraction;size_t low=std::floor(position),high=std::ceil(position);return low==high?values[low]:values[low]*(high-position)+values[high]*(position-low);}

template<int BLOCK_SIZE> void launch_one_thread_min_blocks(int min_blocks,const int*rp,const int*ci,const double*val,const double*x,double*y,int rows){
  int grid=(rows+BLOCK_SIZE-1)/BLOCK_SIZE;
  if(min_blocks==1)one_thread_per_row<BLOCK_SIZE,1><<<grid,BLOCK_SIZE>>>(rp,ci,val,x,y,rows);
  else if(min_blocks==2)one_thread_per_row<BLOCK_SIZE,2><<<grid,BLOCK_SIZE>>>(rp,ci,val,x,y,rows);
  else if(min_blocks==4)one_thread_per_row<BLOCK_SIZE,4><<<grid,BLOCK_SIZE>>>(rp,ci,val,x,y,rows);
  else one_thread_per_row<BLOCK_SIZE,8><<<grid,BLOCK_SIZE>>>(rp,ci,val,x,y,rows);
  return;
}

void launch_one_thread(int block,int min_blocks,const int*rp,const int*ci,const double*val,const double*x,double*y,int rows){
  if(block==64)launch_one_thread_min_blocks<64>(min_blocks,rp,ci,val,x,y,rows);
  else if(block==128)launch_one_thread_min_blocks<128>(min_blocks,rp,ci,val,x,y,rows);
  else launch_one_thread_min_blocks<256>(min_blocks,rp,ci,val,x,y,rows);
  return;
}

int main(int argc,char** argv){
  if(argc!=7){std::fprintf(stderr,"usage: %s matrix mapping block_size wave_groups warmup repeat\n",argv[0]);return 2;}
  std::string path=argv[1],mapping=argv[2];int block=std::atoi(argv[3]),wave_groups=std::atoi(argv[4]),warmup=std::atoi(argv[5]),repeat=std::atoi(argv[6]);
  if((mapping!="one_thread_per_row"&&mapping!="cooperative_row")||(block!=64&&block!=128&&block!=256)||(wave_groups!=1&&wave_groups!=2&&wave_groups!=4&&wave_groups!=8)||warmup<0||repeat<1){std::fprintf(stderr,"invalid registered configuration\n");return 2;}
  int device_id=0;hipDeviceProp_t properties;HIP_OK(hipGetDevice(&device_id));HIP_OK(hipGetDeviceProperties(&properties,device_id));int host_wave_size=properties.warpSize;if(host_wave_size%wave_groups!=0){std::fprintf(stderr,"wave size is not divisible by wave_groups\n");return 2;}
  Matrix matrix=read_mtx(path);std::vector<double>x(matrix.cols),reference(matrix.rows),observed(matrix.rows);for(int i=0;i<matrix.cols;i++)x[i]=std::sin(.001*i)+std::cos(.0003*i);for(int r=0;r<matrix.rows;r++)for(int p=matrix.rp[r];p<matrix.rp[r+1];p++)reference[r]+=matrix.val[p]*x[matrix.ci[p]];
  int *dr,*dc;double *dv,*dx,*dy;HIP_OK(hipMalloc(&dr,matrix.rp.size()*sizeof(int)));HIP_OK(hipMalloc(&dc,matrix.ci.size()*sizeof(int)));HIP_OK(hipMalloc(&dv,matrix.val.size()*sizeof(double)));HIP_OK(hipMalloc(&dx,x.size()*sizeof(double)));HIP_OK(hipMalloc(&dy,observed.size()*sizeof(double)));HIP_OK(hipMemcpy(dr,matrix.rp.data(),matrix.rp.size()*sizeof(int),hipMemcpyHostToDevice));HIP_OK(hipMemcpy(dc,matrix.ci.data(),matrix.ci.size()*sizeof(int),hipMemcpyHostToDevice));HIP_OK(hipMemcpy(dv,matrix.val.data(),matrix.val.size()*sizeof(double),hipMemcpyHostToDevice));HIP_OK(hipMemcpy(dx,x.data(),x.size()*sizeof(double),hipMemcpyHostToDevice));
  auto launch=[&](){if(mapping=="one_thread_per_row")launch_one_thread(block,wave_groups,dr,dc,dv,dx,dy,matrix.rows);else{int width=host_wave_size/wave_groups;int rows_per_block=block/width;cooperative_row<<<(matrix.rows+rows_per_block-1)/rows_per_block,block>>>(dr,dc,dv,dx,dy,matrix.rows,wave_groups);}};
  launch();HIP_OK(hipDeviceSynchronize());HIP_OK(hipMemcpy(observed.data(),dy,observed.size()*sizeof(double),hipMemcpyDeviceToHost));double error=0.0,scale=1.0;for(int i=0;i<matrix.rows;i++){error=std::max(error,std::abs(observed[i]-reference[i]));scale=std::max(scale,std::abs(reference[i]));}double tolerance=1e-12+1e-10*scale;bool pass=std::isfinite(error)&&error<=tolerance;
  std::vector<double> samples;if(pass){for(int i=0;i<warmup;i++)launch();HIP_OK(hipDeviceSynchronize());hipEvent_t begin,end;HIP_OK(hipEventCreate(&begin));HIP_OK(hipEventCreate(&end));for(int i=0;i<repeat;i++){float milliseconds;HIP_OK(hipEventRecord(begin));launch();HIP_OK(hipEventRecord(end));HIP_OK(hipEventSynchronize(end));HIP_OK(hipEventElapsedTime(&milliseconds,begin,end));samples.push_back(milliseconds*1000.0);}HIP_OK(hipEventDestroy(begin));HIP_OK(hipEventDestroy(end));}
  std::printf("schema=spmv-four-cell-result/1.0 backend=hip mapping=%s matrix=%s rows=%d cols=%d nnz=%zu block_size=%d num_warps_or_wave_groups=%d warmup=%d repeat=%d max_abs_error=%.17g tolerance=%.17g correctness=%s",mapping.c_str(),path.c_str(),matrix.rows,matrix.cols,matrix.val.size(),block,wave_groups,warmup,repeat,error,tolerance,pass?"PASS":"FAIL");if(pass)std::printf(" median_us=%.9f p5_us=%.9f p95_us=%.9f",quantile(samples,.5),quantile(samples,.05),quantile(samples,.95));std::printf("\n");for(size_t i=0;i<samples.size();i++)std::printf("sample backend=hip mapping=%s index=%zu event_us=%.9f\n",mapping.c_str(),i,samples[i]);
  HIP_OK(hipFree(dr));HIP_OK(hipFree(dc));HIP_OK(hipFree(dv));HIP_OK(hipFree(dx));HIP_OK(hipFree(dy));return pass?0:3;
}
