#include "D_Matrix.cuh"
#include "H_Matrix.cuh"

#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/scatter.h>
#include <thrust/gather.h>
#include <thrust/reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/device_vector.h>

class TransposeFunctor1 : public thrust::unary_function<const int, int>{
        const int size1;
public:
        __host__ __device__ TransposeFunctor1() = delete;
        __host__ __device__ TransposeFunctor1(const int size)
                : size1(size)
        {}
        TransposeFunctor1(const TransposeFunctor1&) = default;
        __host__ __device__ int operator()(const int &idx){
		int row =idx/size1;
                int col =idx%size1;
                return row + col*size1;
        }
};

class DiffusionFunctor1 : public thrust::unary_function<const int, int>{
        const int* ptr;
	const int m_n;
public:
        __host__ __device__ DiffusionFunctor1() = delete;
        __host__ __device__ DiffusionFunctor1(const int* ptr, const int size)
                : ptr(ptr), m_n(size)
        {}
        DiffusionFunctor1(const DiffusionFunctor1&) = default;
        __host__ __device__ int operator()(const int &idx){
                return ptr[idx%m_n];
        }
};

struct ProductFunctor1 : public thrust::binary_function<int,int,int>{
        __device__ int operator() (const int&x, const int&y) const {return x*y;}
};

class LineFunctor1 : public thrust::unary_function<const int, int>{
        const int m_n;
public:
        __host__ __device__ LineFunctor1() = delete;
        __host__ __device__ LineFunctor1(const int size)
                : m_n(size)
        {}
        LineFunctor1(const LineFunctor1&) = default;
        __host__ __device__ int operator()(const int &idx){
                return idx/m_n*m_n;
        }
};

struct MultiplyTupleFunctor1 : public thrust::unary_function< thrust::tuple<int,int>, int>{
        __device__ int operator() (const thrust::tuple<int,int>&t) const {return thrust::get<0>(t) * thrust::get<1>(t);}
};


// Exercice 1
bool D_Matrix::Exo1IsDone() {
	return true; 
}

// returns this times that ...

D_Matrix D_Matrix::operator+(const D_Matrix& that) const {

	// do "d_val + that.d_val" 
        D_Matrix result(m_n);
	const int size =m_n*m_n; 
	thrust::transform(d_val,d_val+size,that.d_val,result.d_val,thrust::plus<int>());
	return result; 
}


//////////////////////////////////////////////////////////////////////////////////
// Exercice 2
bool D_Matrix::Exo2IsDone() {
	return true;
}


// define the Matrix::transpose function

D_Matrix D_Matrix::transpose() const
{
	D_Matrix result(m_n);
	const int size =m_n*m_n;

        auto iterator =
                thrust::make_transform_iterator(
                        thrust::make_counting_iterator<int>(0),
                        TransposeFunctor1(m_n)
                );

        thrust::scatter(
                d_val,
                d_val+size,
                iterator,
                result.d_val
        );

	return result;
}



//////////////////////////////////////////////////////////////////////////////////
// Exercice 3
bool D_Matrix::Exo3IsDone() {
	return true;
}

void D_Matrix::diffusion(const int line, D_Matrix& result) const 
{

        const int size =m_n*m_n;

	auto iterator =
                thrust::make_transform_iterator(
                        thrust::make_counting_iterator<int>(0),
                        DiffusionFunctor1(d_val.get()+m_n*line,m_n)
                );

	//ilf faut stocker la ligne qu'on veut copier dans un iterateur et la multiplier n fois.
	//on faut un functor qui stocke la ligne désirée dans un itérateur.

	//on fait un gather pour obtenir notre ligne désirée puis on la copie n fois dans le resultat.


		thrust::copy_n(
			iterator,
			size,
			result.d_val
        	);


}



//////////////////////////////////////////////////////////////////////////////////
// Exercice 4
bool D_Matrix::Exo4IsDone() {
	return true;
}
// returns this times that ...
D_Matrix D_Matrix::product1(const D_Matrix& that) const
{


	D_Matrix result(m_n);
	D_Matrix tb(m_n);
	D_Matrix tbi(m_n);
	int size = m_n*m_n;

	auto iterator1 =
                thrust::make_transform_iterator(
                        thrust::make_counting_iterator<int>(0),
                        LineFunctor1(m_n)
                );

	tb = that.transpose();



	for (int i=0; i<m_n; ++i){

		D_Matrix D(m_n);

		thrust::device_vector<int> column(m_n);

		thrust::device_vector<int> output_keys(m_n);

		tb.diffusion(i, tbi);

		thrust::transform(d_val, d_val+size,tbi.d_val, D.d_val, thrust::multiplies<int>());

		//functor pour les ligne pour input
		thrust::reduce_by_key(iterator1, iterator1+size, D.d_val, output_keys.begin(), column.begin());


        	thrust::scatter(
                	column.begin(),
                	column.end(),
                	output_keys.begin(),
                	result.d_val+i
        	);

/*
		thrust::scatter(
                        column.begin(),
                        column.end(),
                        iterator,
                        result.d_val
                );
*/
	}
	return result;
}


//////////////////////////////////////////////////////////////////////////////////
// Exercice 5
bool D_Matrix::Exo5IsDone() {
	return true;
}

// returns this times that ...
D_Matrix D_Matrix::product2(const D_Matrix& that) const {

	D_Matrix result(m_n);
        int size = m_n*m_n;
        auto iterator1 =
                thrust::make_transform_iterator(
                        thrust::make_counting_iterator<int>(0),
                        LineFunctor1(m_n)
                );

	thrust::device_vector<int> column(m_n);

        thrust::device_vector<int> output_keys(m_n);

	thrust::fill(result.d_val, result.d_val+size, 0);

	D_Matrix tb = that.transpose();

        for (int i=0; i<m_n; ++i){


		auto iterator2 =
                thrust::make_transform_iterator(
                        thrust::make_counting_iterator<int>(0),
                        DiffusionFunctor1(tb.d_val.get()+m_n*i,m_n)
                );

		auto iterator3 =
                thrust::make_transform_iterator(
                        thrust::make_zip_iterator(thrust::make_tuple(d_val, iterator2)),
                        MultiplyTupleFunctor1()
                );

                //functor pour les ligne pour input

                thrust::reduce_by_key(iterator1, iterator1+size, iterator3, output_keys.begin(), column.begin());

                thrust::copy(
                        column.begin(),
                        column.end(),
                        //output_keys.begin(),
                        result.d_val+i*m_n
                );

        }
        return result.transpose();

}
