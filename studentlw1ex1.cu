#include "Exercise.hpp"
#include <chronoGPU.hpp>

struct AdditionFunctor1 : public thrust::binary_function<float,float,float>{
	__device__ float operator() (const float&x, const float&y) const {return x+y;}
};

typedef thrust :: tuple<float , float , float> myFloat3 ;

struct AdditionFunctor3 : public thrust::unary_function<myFloat3, float>{
	__device__ float operator() (const myFloat3& tuple) const {return thrust:: get<0>(tuple) + thrust::get<1>(tuple) + thrust::get<2>(tuple);
	}
};

void Exercise::Question1(const thrust::host_vector<int>& A,
						const thrust::host_vector<int>& B, 
						thrust::host_vector<int>&C) const
{

thrust::device_vector<float> U = A;
thrust::device_vector<float> V = B;
thrust::device_vector<float> result(C.size());
thrust::transform(U.begin(), U.end(), V.begin(), result.begin(), AdditionFunctor1());
C = result;


for(int i=0; i<128; ++i) {
	std::cout<< "ResultQ1[" << i << "] = " << C[i] << std::endl; 
}

}

void Exercise::Question2(thrust::host_vector<int>&A) const 
{

thrust::counting_iterator<float>U(1.f);
thrust::constant_iterator<float>V(4.f);
thrust::device_vector<float>result(A.size());
thrust::transform(U, U+result.size(), V, result.begin(), AdditionFunctor1());

A = result;


for(int i=0; i<128; ++i) {
        std::cout<< "ResultQ2[" << i << "] = " << A[i] << std::endl; 
}

}



void Exercise::Question3(const thrust::host_vector<int>& A,
						const thrust::host_vector<int>& B, 
						const thrust::host_vector<int>& C, 
						thrust::host_vector<int>&D) const 
{

thrust::device_vector<float>Adev(A.size());
thrust::device_vector<float>Bdev(B.size());
thrust::device_vector<float>Cdev(C.size());
thrust::device_vector<float>Ddev(D.size());

Adev = A;
Bdev = B;
Cdev = C;

thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(Adev.begin(), Bdev.begin(), Cdev.begin())),
		  thrust::make_zip_iterator(thrust::make_tuple(Adev.end(), Bdev.end(), Cdev.end())),
		  Ddev.begin(),
		  AdditionFunctor3());

D = Ddev;

for(int i=0; i<128; ++i) {
        std::cout<< "ResultQ3[" << i << "] = " << D[i] << std::endl;
}

}
