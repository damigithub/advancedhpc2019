#include "student.h"


// ==========================================================================================
// Exercise 1

namespace {
	// Feel free to add any function you need, it is your file ;-)

	struct FlagFunctor1 : public thrust::unary_function<ColoredObject::Color, int>{

        	__device__ int operator() (const ColoredObject::Color&C) const {
			return (C == ColoredObject::Color::BLUE);
		}

	};

}

bool StudentWork1::isImplemented() const {
	return true;
}




// you should do this method to return the blue objects contained in the input parameter
thrust::device_vector<ColoredObject> StudentWork1::compactBlue( const thrust::device_vector<ColoredObject>& d_input ) {
	// it should work on GPU ;-)
	// use FLAG set to 1 for BLUE objects, 0 else
	// then do a SCAN to count the number of BLUE object, and obtain their relative position (+1)
	// At least scatter them into the anwser array !




	thrust::device_vector<int>resultFlag(d_input.size());

	thrust::device_vector<int>resultScanY(d_input.size());

	thrust::transform(d_input.begin(), d_input.end(), resultFlag.begin(), FlagFunctor1());

	thrust::inclusive_scan(resultFlag.begin(), resultFlag.end(),resultScanY.begin()); 

	thrust::device_vector<ColoredObject> result( resultScanY[ d_input.size() - 1 ] );

        thrust::scatter_if(
                d_input.begin(),
                d_input.end(),
                resultScanY.begin(),
		resultFlag.begin(),
                result.begin()-1
        );



	return result;




}
