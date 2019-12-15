
#include "student.h"

namespace {

/*

    struct HistoFunctor1 : public thrust::unary_function<, int>{

                __device__ int operator() (const ColoredObject::Color&C) const {
                        return (C == ColoredObject::Color::BLUE);
                }

        };

*/

    template<typename T>
    __device__ inline T max(const T&a, const T&b) { 
        return a<b ? b : a; 
    }

	struct RGB2VFunctor : public thrust::unary_function<uchar3,uchar>
    {
        __device__
        uchar operator() (const uchar3& RGB) {
            return max(RGB.x, max(RGB.y, RGB.z)); // return the Value, i.e. the max
        }
    };

    struct FilterFunctor : public thrust::binary_function<const uchar3,const uchar,uchar3>
    {
        __device__
        uchar3 operator() (const uchar3&u_rgb, const uchar V) 
        {
            const float3 RGB = make_float3( float(u_rgb.x), float(u_rgb.y), float(u_rgb.z));
            const float d = fmaxf(RGB.x, fmaxf(RGB.y, RGB.z)); // old value
            const float N = d > 0.f ? float(V) / d : 0.f; // ratio
            const float R = fminf(RGB.x * N, 255.f);
            const float G = fminf(RGB.y * N, 255.f);
            const float B = fminf(RGB.z * N, 255.f);
            return make_uchar3(R, G, B); // modify the value of a given pixel
        }
	};

	// first pixel, fill all the shared memory with its neighbours


 __device__ void fill_shared_memory(
        const uchar*const d_V,
        const int py,
        const unsigned width, const unsigned height,
        const unsigned filter_size
    ) {
        extern __shared__ int s_Histo[];
        const int px = blockIdx.x;

        // we have exactly 256 threads
        s_Histo[threadIdx.x] = 0u;
        __syncthreads();


        const int startX = px - (filter_size>>1);
        const int startY = py - (filter_size>>1);
        const int endX =  px+(filter_size>>1);
        const int endY =  py+(filter_size>>1);
        
        if ((startX > 0)&&(startY > 0)&&(endX < width)&&(endY < height)){

                for(unsigned tid=threadIdx.x; tid < filter_size*filter_size; tid+=blockDim.x )
		{
		
			 //TODO: histogram with all neighbours.
                                
                                int x = startX + tid%filter_size;
                                int y = startY + tid/filter_size;
				if( startX<=0 ){x = startX - threadIdx.x%filter_size; }
                                if( endX>=width){x = startX - threadIdx.x%filter_size; }
                                if( startY<=0 ){y = startY - threadIdx.y%filter_size; }
                                if( endY >=height){y = startY - threadIdx.y%filter_size; }
                                atomicAdd( &s_Histo[(unsigned int)d_V[x + y*width]], 1 );
                }
        }

        __syncthreads();

        }

    __device__ void update_histo(
        const uchar*const d_V, 
        const int py,
        const unsigned width, const unsigned height, 
        const unsigned filter_size
    ) {
        // need to remove the top line, and to add the bottom one
        extern __shared__ int s_Histo[];
        const int px = blockIdx.x;

	const int startX = px - (filter_size>>1);
        const int startY = py - (filter_size>>1);
        const int endX = px + (filter_size>>1);
        const int endY = py + (filter_size>>1);
        int x = startX + threadIdx.x%filter_size;
        int y = startY + threadIdx.x/filter_size;
        


            // TODO: modify histogram, remove old top line, add new bottom one
                for(unsigned tid=threadIdx.x; tid < filter_size*filter_size; tid+=blockDim.x )
                {

		 if((startX> 0) && (endX< width)){
                        if(startY - 1 >= 0){
                                atomicSub(&s_Histo[(int)d_V[x + (startY - 1 )*width]], 1);
                        }

                        if(startY + filter_size - 1 < height){
                                 atomicAdd(&s_Histo[(int)d_V[x + (startY + filter_size -1)*width]], 1 );
			}
		}
		}


    }



    __device__ void scan(const int py) 
    {
        extern __shared__ int s_mem[];
        const int *const s_Histo = &s_mem[0];
        volatile int *const s_scan = &s_mem[256];

        // 256 threads ...
        s_scan[threadIdx.x] = s_Histo[threadIdx.x];
        __syncthreads();


	for (int offset=1; offset<8; offset*=2){

		unsigned int a = s_scan[threadIdx.x];
		unsigned int b = s_scan[threadIdx.x - offset];
	
		__syncthreads();
		
		if (threadIdx.x - offset >= 0){	
			b += a;
		}
			__syncthreads();
		if (threadIdx.x - offset >= 0){
			s_scan[threadIdx.x] = b;
		}
			__syncthreads();
		
	}

    }

    __device__ void apply_filter(
        const uchar*const d_V,
        uchar*const d_V_median,
        const int py,
        const unsigned width,
        const unsigned limit
    ) {
        extern __shared__ int s_mem[];
        const int *const s_cdf = &s_mem[256];
        // after scan, the histo is a CDF (cumulative distribution function)
        // then only only thread will succeed the following test ;-)
		// TODO
		//On veut appliquer le median filter sur notre résultat du scan et le ranger dans dV_median.
		//Notre "windows" est de taille filter-size, donc on ne sort qu'un seul thread, le thread median.

		//trouver la valeur mediane dans s_cdf :

		if ( ((!(threadIdx.x >0)) ||  (s_cdf[threadIdx.x - 1] < limit) ) && (s_cdf[threadIdx.x]>=limit) ){

			const int px = blockIdx.x;

			d_V_median[px + py*width] = threadIdx.x;

		}

	}


#define CHECK
#ifdef CHECK
    __device__ void check_scan() 
    {
        extern __shared__ int s_mem[];
        const int *const s_scan = &s_mem[256];
        if( threadIdx.x>0 && s_scan[threadIdx.x-1]>s_scan[threadIdx.x] )
            printf("[%d/%d] bad values: %d\n", blockIdx.x, threadIdx.x, s_scan[threadIdx.x]);
    }
#endif

    __global__ void filter(
        const uchar*const d_V, 
        uchar*const d_V_median, 
		const unsigned width, 
		const unsigned height, 
        const unsigned filter_size
    ) {
        ::fill_shared_memory(d_V, 0, width, height, filter_size);
        // first pixel is specific (no maj): just scan and then apply filter
	
		::scan(0); 
		
#ifdef CHECK
        ::check_scan();
#endif
        ::apply_filter(d_V, d_V_median, 0, width, filter_size*filter_size/2);
        // others came after the first one, only updating the histo
        for(int py=1; py<height; ++py) 
        {
            // maj histo
            ::update_histo(d_V, py, width, height, filter_size);
            // scan
            ::scan(py); 
#ifdef CHECK
            ::check_scan();
#endif
            // apply
			::apply_filter(d_V, d_V_median, py, width, filter_size*filter_size/2);
        }
    }
}

bool StudentWork1::isImplemented() const {
	return true;
}


void StudentWork1::rgb2h(
	const thrust::device_vector<uchar3>&rgb,
	thrust::device_vector<uchar>&V
)
{
    thrust::transform(
        rgb.begin(),
        rgb.end(),
		V.begin(),
		::RGB2VFunctor()
    );
}

void StudentWork1::median(
	const thrust::device_vector<uchar> &d_V,
	thrust::device_vector<uchar> &d_V_median,
	const unsigned width,
	const unsigned height,
	const unsigned filter_size
) {    
	dim3 threads(256);    
	if( d_V.size() != width * height ) 
		std::cout << "Problem with the size of d_V" << std::endl;
	if( d_V_median.size() != width * height ) 
		std::cout << "Problem with the size of d_V_median" << std::endl;
    uchar const*const V = d_V.begin().base().get();
    uchar *const F = d_V_median.begin().base().get();

/*
//test (pas écrit pas le prof) ::

	int numBlocky = height / 256 ;
	if (height % 256 > 0){
        	numBlocky++ ;
    	}

/////////////////////////////////
*/

	dim3 blocks(width); 
	::filter<<<blocks, threads, sizeof(int)*512>>>(V, F, width, height, filter_size);
	std::cout << "do the copy" << std::endl;
}


void StudentWork1::apply_filter(
	const thrust::device_vector<uchar3>&RGB_old,
	const thrust::device_vector<uchar>&V_new,
	thrust::device_vector<uchar3>&RGB_new
) 
{
    thrust::transform(
        RGB_old.begin(), RGB_old.end(),
        V_new.begin(),
        RGB_new.begin(),
        ::FilterFunctor()
    );
}
