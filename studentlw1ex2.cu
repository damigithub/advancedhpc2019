#include <thrust/scatter.h>
#include <thrust/gather.h>
#include "Exercise.hpp"
#include "include/chronoGPU.hpp"


class OddEvenFunctor1 : public thrust::unary_function<const long long, long long>{
	const long long m_half_size;
public:
	__host__ __device__ OddEvenFunctor1() = delete;
	__host__ __device__ OddEvenFunctor1(const long long size)
		: m_half_size(size/2)
	{}
	OddEvenFunctor1(const OddEvenFunctor1&) = default;
	__host__ __device__ long long operator()(const long long &idx){
		return ( idx < m_half_size ) ? idx*2  : 1+(idx-m_half_size)*2;
	}
};

class OddEvenFunctor2 : public thrust::unary_function<const long long, long long>{
        const long long m_half_size;
public:
        __host__ __device__ OddEvenFunctor2() = delete;
        __host__ __device__ OddEvenFunctor2(const long long size)
                : m_half_size(size/2)
        {}
        OddEvenFunctor2(const OddEvenFunctor2&) = default;
        __host__ __device__ long long operator()(const long long &idx){
                return ( idx%2 == 0 ) ? idx/2  : m_half_size+idx/2;
        }
};

void Exercise::Question1(const thrust::host_vector<int>& A, 
					         thrust::host_vector<int>& OE ) const
{

	thrust::device_vector<int> U = A;
	thrust::device_vector<int>resultIndex(U.size());

	auto iterator =
		thrust::make_transform_iterator(
			thrust::make_counting_iterator<int>(0),
			OddEvenFunctor1(A.size())
		);

	thrust::gather(
		iterator,
		iterator + A.size(),
		U.begin(),
		resultIndex.begin()
	);

	OE = resultIndex;

}



void Exercise::Question2(const thrust::host_vector<int>&A, 
						thrust::host_vector<int>&OE) const 
{

thrust::device_vector<int> U = A;
        thrust::device_vector<int>resultIndex(U.size());

        auto iterator =
                thrust::make_transform_iterator(
                        thrust::make_counting_iterator<int>(0),
                        OddEvenFunctor2(A.size())
                );

        thrust::scatter(
                U.begin(),
                U.end(),
                iterator,
                resultIndex.begin()
        );

        OE = resultIndex;

        for(int i=0; i<128; ++i) {
                std::cout<< "ResultQ2[" << i << "] = " << OE[i] << std::endl; 
        }

}




template <typename T>
void Exercise::Question3(const thrust::host_vector<T>& A,
						thrust::host_vector<T>&OE) const 
{

	thrust::device_vector<T> U = A;
        thrust::device_vector<T>resultIndex(U.size());

        auto iterator =
                thrust::make_transform_iterator(
                        thrust::make_counting_iterator<int>(0),
                        OddEvenFunctor2(A.size())
                );

        thrust::scatter(
                U.begin(),
                U.end(),
                iterator,
                resultIndex.begin()
        );

        OE = resultIndex;

}


struct MyDataType {

	MyDataType(int i) : m_i(i) {}
	MyDataType(float j) : m_j(j) {}
	MyDataType() = default;
	~MyDataType() = default;
	int m_i;
	float m_j;
	operator int() const { return m_i; }
	operator float() const { return m_j; }


};

// Warning: do not modify the following function ...
void Exercise::checkQuestion3() const {
	const size_t size = sizeof(MyDataType)*m_size;
	std::cout<<"Check exercice 3 with arrays of size "<<(size>>20)<<" Mb"<<std::endl;
	checkQuestion3withDataType(MyDataType(0));
}
