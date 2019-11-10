
#include <stdio.h>
#include <include/labwork.h>
#include <cuda_runtime_api.h>
#include <omp.h>
#include <string.h>

#define ACTIVE_THREADS 4

int main(int argc, char **argv) {
    printf("USTH ICT Master 2018, Advanced Programming for HPC.\n");
    if (argc < 2) {
        printf("Usage: labwork <lwNum> <inputImage>\n");
        printf("   lwNum        labwork number\n");
        printf("   inputImage   the input file name, in JPEG format\n");
        return 0;
    }

    int lwNum = atoi(argv[1]);
    std::string inputFilename;
    std::string inputFilename2;

    // pre-initialize CUDA to avoid incorrect profiling
    printf("Warming up...\n");
    char *temp;
    cudaMalloc(&temp, 1024);

    Labwork labwork;
    if (lwNum != 2 ) {
        inputFilename = std::string(argv[2]);
        labwork.loadInputImage(inputFilename);


        //For Blending

        if (argc == 4){
     


        	inputFilename2 = std::string(argv[3]);
        	labwork.loadInputImage2(inputFilename2);


                //Resizing

                int w = labwork.getWidth();
                int h = labwork.getHeight();

                
		char * cmd = (char *)malloc(sizeof(char)*100); ;
                sprintf(cmd, "convert %s -resize %dx%d %s",argv[3], w, h, argv[3]);
		const char * command = cmd;




                system(command);

                labwork.loadInputImage2(inputFilename2);
 
        }
    }

    printf("Starting labwork %d\n", lwNum);
    Timer timer;
    timer.start();
    switch (lwNum) {
        case 1:
            labwork.labwork1_CPU();
            labwork.saveOutputImage("labwork2-cpu-out.jpg");
            printf("labwork 1 CPU ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            timer.start();
            labwork.labwork1_OpenMP();
            labwork.saveOutputImage("labwork2-openmp-out.jpg");
	    printf("labwork 1 OPENMP ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            break;
        case 2:
            labwork.labwork2_GPU();
            break;
        case 3:

            labwork.labwork1_CPU();
            labwork.saveOutputImage("labwork2-cpu-out.jpg");
            printf("labwork 1 CPU ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            timer.start();
            labwork.labwork3_GPU();
            labwork.saveOutputImage("labwork3-gpu-out.jpg");
	    printf("labwork 3 ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            break;
        case 4:
            labwork.labwork4_GPU();
            labwork.saveOutputImage("labwork4-gpu-out.jpg");
            printf("labwork 4 GPU ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            break;
        case 5:
            labwork.labwork5_GPU();
            labwork.saveOutputImage("labwork5-gpu-out.jpg");
            break;
        case 6:
            labwork.labwork6_GPU();
            labwork.saveOutputImage("labwork6-gpu-out.jpg");
            break;
        case 7:
            labwork.labwork7_GPU();
            printf("[ALGO ONLY] labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork7-gpu-out.jpg");
            break;
        case 8:
            labwork.labwork8_GPU();
            printf("[ALGO ONLY] labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork8-gpu-out.jpg");
            break;
        case 9:
            labwork.labwork9_GPU();
            printf("[ALGO ONLY] labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork9-gpu-out.jpg");
            break;
        case 10:
            labwork.labwork10_GPU();
            printf("[ALGO ONLY] labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork10-gpu-out.jpg");
            break;
    }
    printf("labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
}

void Labwork::loadInputImage(std::string inputFileName) {
    inputImage = jpegLoader.load(inputFileName);
}

void Labwork::loadInputImage2(std::string inputFileName) {
    inputImage2 = jpegLoader.load(inputFileName);
}

int Labwork::getWidth() {
	return inputImage->width;
}

int Labwork::getHeight() {
	return inputImage->height;
}

void Labwork::saveOutputImage(std::string outputFileName) {
    jpegLoader.save(outputFileName, outputImage, inputImage->width, inputImage->height, 90);
}

void Labwork::labwork1_CPU() {
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));
    for (int j = 0; j < 100; j++) {     // let's do it 100 times, otherwise it's too fast!
        for (int i = 0; i < pixelCount; i++) {
            outputImage[i * 3] = (char) (((int) inputImage->buffer[i * 3] + (int) inputImage->buffer[i * 3 + 1] +
                                          (int) inputImage->buffer[i * 3 + 2]) / 3);
            outputImage[i * 3 + 1] = outputImage[i * 3];
            outputImage[i * 3 + 2] = outputImage[i * 3];
        }
    }
}

void Labwork::labwork1_OpenMP() {
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));

	#pragma omp parallel for

        for (int j = 0; j < 100; j++) {     // let's do it 100 times, otherwise it's too fast!
        for (int i = 0; i < pixelCount; i++) {
            outputImage[i * 3] = (char) (((int) inputImage->buffer[i * 3] + (int) inputImage->buffer[i * 3 + 1] +
                                          (int) inputImage->buffer[i * 3 + 2]) / 3);
            outputImage[i * 3 + 1] = outputImage[i * 3];
            outputImage[i * 3 + 2] = outputImage[i * 3];
        }
    }
}

int getSPcores(cudaDeviceProp devProp) {
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major) {
        case 2: // Fermi
            if (devProp.minor == 1) cores = mp * 48;
            else cores = mp * 32;
            break;
        case 3: // Kepler
            cores = mp * 192;
            break;
        case 5: // Maxwell
            cores = mp * 128;
            break;
        case 6: // Pascal
            if (devProp.minor == 1) cores = mp * 128;
            else if (devProp.minor == 0) cores = mp * 64;
            else printf("Unknown device type\n");
            break;
        default:
            printf("Unknown device type\n");
            break;
    }
    return cores;
}

void Labwork::labwork2_GPU() {
    int nDevices = 0;
    // get all devices
    cudaGetDeviceCount(&nDevices);
    printf("Number total of GPU : %d\n\n", nDevices);
    for (int i = 0; i < nDevices; i++){
        // get informations from individual device
        cudaDeviceProp prop;
	printf("Filling GPU number : %d\n", i);
        cudaGetDeviceProperties(&prop, i);
        // something more here		

		//Core INFO
		printf("Device name of GPU number %d : %s\n", i, prop.name);
		printf("Clock rate: %d\n",prop.clockRate);
		int nbCores = getSPcores(prop);
		printf("Number of cores: %d\n", nbCores);
		printf("Number of multiprocessors on device : %d\n", prop.multiProcessorCount);
		printf("Warp Size : %d\n", prop.warpSize);

		//Memory INFO
		printf("Memory Clock Rate : %d\n",prop.memoryClockRate);
		printf("Memory Bus Width : %d\n", prop.memoryBusWidth);
		
		
	
    }

}

//Write a grey scale kernel here :
 __global__ void grayScale(uchar3 *input, uchar3 *output) {
       int tid = threadIdx.x + blockIdx.x * blockDim.x;
       output[tid].x = (input[tid].x + input[tid].y +
                       input[tid].z) / 3;
       output[tid].z = output[tid].y = output[tid].x;
}
//This should be executed on a device core.

void Labwork::labwork3_GPU() {
    // Calculate number of pixels
 
    int pixelCount = inputImage->width * inputImage->height ;

    outputImage = static_cast<char *>(malloc(pixelCount * 3));

    //Allocate CUDA Memory

    uchar3 *devInput;
    uchar3 *devOutput;
    cudaMalloc(&devInput, pixelCount *sizeof(uchar3));
    cudaMalloc(&devOutput, pixelCount *sizeof(uchar3));


    //Copy CUDA Memory from CPU to GPU

    cudaMemcpy(devInput, inputImage->buffer, pixelCount * sizeof(uchar3), cudaMemcpyHostToDevice);

    //Processing

    int blockSize = 64;
    int numBlock = pixelCount  / blockSize;
    printf("numblock %d\n", numBlock);
    grayScale<<<numBlock, blockSize>>>(devInput , devOutput);
    
    //Copy CUDA Memory from GPU to CPU

    cudaMemcpy(outputImage, devOutput, pixelCount * sizeof(uchar3), cudaMemcpyDeviceToHost);

    //Cleaning

    cudaFree(devInput);
    cudaFree(devOutput);


}


//Write a grey scale kernel here :
 __global__ void grayScale2(uchar3 *input, uchar3 *output,int width, int height) {

       int x = threadIdx.x + blockIdx.x * blockDim.x;
       int y = threadIdx.y + blockIdx.y * blockDim.y;
       int w = blockDim.x * gridDim.x;

       //if ((gridDim.x * gridDim.y) < width * height){
       
       	int tid = y*w + x; 

        output[tid].x = (input[tid].x + input[tid].y +
                       input[tid].z) / 3;
        output[tid].z = output[tid].y = output[tid].x;

      // }
}

void Labwork::labwork4_GPU() {

    // Calculate number of pixels
 
    int pixelCount = inputImage->width * inputImage->height ;

    outputImage = static_cast<char *>(malloc(pixelCount * 3));

    //Allocate CUDA Memory

    uchar3 *devInput;
    uchar3 *devOutput;
    cudaMalloc(&devInput, pixelCount *sizeof(uchar3));
    cudaMalloc(&devOutput, pixelCount *sizeof(uchar3));


    //Copy CUDA Memory from CPU to GPU

    cudaMemcpy(devInput, inputImage->buffer, pixelCount * sizeof(uchar3), cudaMemcpyHostToDevice);

    //Processing

    dim3 blockSize = dim3(8, 8);
//    int rx = inputImage->width%blockSize.x;
//    int ry = inputImage->height%blockSize.y;
    dim3 gridSize = dim3 (inputImage->width/blockSize.x,inputImage->height/blockSize.y);  
    grayScale2<<<gridSize, blockSize>>>(devInput, devOutput, inputImage->width, inputImage->height);    

    //Copy CUDA Memory from GPU to CPU

    cudaMemcpy(outputImage, devOutput, pixelCount * sizeof(uchar3), cudaMemcpyDeviceToHost);

    //Cleaning

    cudaFree(devInput);
    cudaFree(devOutput);


}

//Write a grey scale kernel here :
 __global__ void grayScale3(uchar3 *input, uchar3 *output,int width, int height) {

        
       int x = threadIdx.x + blockIdx.x * blockDim.x;
       int y = threadIdx.y + blockIdx.y * blockDim.y;

       //if ((gridDim.x * gridDim.y) < width * height){
       
        int tid = y*width + x; 

        if (x<width){

        if (y<height){ 


        output[tid].x = (input[tid].x + input[tid].y +
                       input[tid].z) / 3;

        output[tid].z = output[tid].y = output[tid].x;
        }
        }

      // }
}

//Write a grey scale kernel here :
 __global__ void blur(uchar3 *input, uchar3 *output,int width, int height) {

       int matrix[7][7] = {{0,0,1,2,1,0,0},{0,3,13,22,13,3,0},{1,3,59,97,59,13,1},{2,22,97,159,97,22,2},{1,3,59,97,59,3,1},{0,3,13,22,13,3,0},{0,0,1,2,1,0,0}};


       int x = threadIdx.x + blockIdx.x * blockDim.x;
       int y = threadIdx.y + blockIdx.y * blockDim.y;

       //if ((gridDim.x * gridDim.y) < width * height){
   
        int tid = y*width + x; 

        int outputTemp = 0;

        int sommeCoef = 0;

        if (x<width){ 

        if (y<height){  

        if (x>3 && x<width-3 && y>3 && y<height-3){ 

                for (int i=0; i<7; i++){

                	for (int j=0; j<7; j++){

        			outputTemp += input[(y-3+i)*width+(x-3+j)].x*matrix[j][i]; 
                                
                                sommeCoef += matrix[j][i];

                        }

                        
                        output[tid].x = outputTemp / sommeCoef;

   			output[tid].z = output[tid].y = output[tid].x;

		}
        }

        }

        }

      // }
}






  void Labwork::labwork5_CPU() {

  // Calculate number of pixels
 
    int pixelCount = inputImage->width * inputImage->height ;

    outputImage = static_cast<char *>(malloc(pixelCount * 3));

    for (int j = 0; j < 100; j++) {     // let's do it 100 times, otherwise it's too fast!
        for (int i = 0; i < pixelCount; i++) {
            outputImage[i * 3] = (char) (( (int) inputImage->buffer[i * 3] + (int) inputImage->buffer[i * 3 + 1] +
                                          (int) inputImage->buffer[i * 3 + 2]) / 3);
            outputImage[i * 3 + 1] = outputImage[i * 3];
            outputImage[i * 3 + 2] = outputImage[i * 3];
        }
    }

       int matrix[7][7] = {{0,0,1,2,1,0,0},{0,3,13,22,13,3,0},{1,3,59,97,59,13,1},{2,22,97,159,97,22,2},{1,3,59,97,59,3,1},{0,3,13,22,13,3,0},{0,0,1,2,1,0,0}};



       int outputTemp = 0;

       int sommeCoef = 0;

       int width = inputImage->width; 

       for (int k = 0; k< inputImage->height;k++){ 

       for (int l = 0; l< width; l++){

                for (int i=0; i<7; i++){

                        for (int j=0; j<7; j++){

                                outputTemp += ((int) inputImage->buffer[(k-3+i)*width+(l-3+j)] ) *matrix[j][i]; 
                                
                                sommeCoef += matrix[j][i];

                        }

                        
                        outputImage[i*3] = (char) (outputTemp / sommeCoef);
                        outputImage[i * 3 + 1] = outputImage[i * 3];
                        outputImage[i * 3 + 2] = outputImage[i * 3];

                }

        }

        }




} 


//Labwork5_GPU() without shared memory.

void Labwork::labwork5_GPU() {



  // Calculate number of pixels
 

    int pixelCount = inputImage->width * inputImage->height ;


    outputImage = static_cast<char *>(malloc(pixelCount * 3));

    //Allocate CUDA Memory

    uchar3 *devInput;
    uchar3 *devOutput;
    uchar3 *devGray;
    cudaMalloc(&devInput, pixelCount *sizeof(uchar3));
    cudaMalloc(&devOutput, pixelCount *sizeof(uchar3));
    cudaMalloc(&devGray, pixelCount *sizeof(uchar3));

    //Copy CUDA Memory from CPU to GPU

    cudaMemcpy(devInput, inputImage->buffer, pixelCount * sizeof(uchar3), cudaMemcpyHostToDevice);

    //Processing

    dim3 blockSize = dim3(32, 32);

//    int rx = inputImage->width%blockSize.x;
//    int ry = inputImage->height%blockSize.y;

    int numBlockx = inputImage-> width / (blockSize.x) ;
    int numBlocky = inputImage-> height / (blockSize.y) ;
    if ((inputImage-> width % (blockSize.x)) > 0) {
    	numBlockx++ ;                                                                
    }                 
    if ((inputImage-> height % (blockSize.y)) > 0){ 
        numBlocky++ ;                                            
    }                                                             

    dim3 gridSize = dim3 (numBlockx,numBlocky);  
    grayScale3<<<gridSize, blockSize>>>(devInput, devGray, inputImage->width, inputImage->height);    
    blur<<<gridSize, blockSize>>>(devGray, devOutput, inputImage->width, inputImage->height);

    //Copy CUDA Memory from GPU to CPU

    cudaMemcpy(outputImage, devOutput, pixelCount * sizeof(uchar3), cudaMemcpyDeviceToHost);

    //Cleaning

    cudaFree(devInput);
    cudaFree(devOutput);

}

 __global__ void binary(uchar3 *input, uchar3 *output,int width, int height,int threshold) {

        
       int x = threadIdx.x + blockIdx.x * blockDim.x;
       int y = threadIdx.y + blockIdx.y * blockDim.y;

       
        int tid = y*width + x; 

        if (x<width){

        if (y<height){ 

        output[tid].x = (input[tid].x + input[tid].y +
                       input[tid].z) / 3;


        if (output[tid].x >= threshold){

                output[tid].x = 255;
        	output[tid].z = output[tid].y = output[tid].x;

        }

        else {
	        output[tid].x = 0;
        	output[tid].z = output[tid].y = output[tid].x;
        }

        }
        }

      // }
}

 __global__ void brightness(uchar3 *input, uchar3 *output,int width, int height,int brightness) {

        
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;
       
        int tid = y*width + x; 

        if (x<width){

        if (y<height){ 

        output[tid].x = (input[tid].x + input[tid].y +
                       input[tid].z) / 3;

        if (brightness> 50 && brightness != 100){

        	if (output[tid].x + brightness <= 255){ 

        		output[tid].x += brightness;

        	}

        	else{
        
        		output[tid].x = 255;

        	}

        }

        if (brightness < 50 && brightness !=0){
	
	  if (output[tid].x - brightness >= 0){ 

                        output[tid].x -= (50-brightness);

                }

                else{
        
                        output[tid].x = 0;

                }



        }

        if (brightness == 100){
    
            output[tid].x = 255;
        }

        if (brightness == 0){ 
    
            output[tid].x = 0;
        }

        output[tid].z = output[tid].y = output[tid].x;

        }
        }

}

 __global__ void blendingGray(uchar3 *input, uchar3 *input2, uchar3 *output,int width, int height,float coefficient) {

        
       int x = threadIdx.x + blockIdx.x * blockDim.x;
       int y = threadIdx.y + blockIdx.y * blockDim.y;

       
        int tid = y*width + x; 

        int nbPixels = width * height;
	float prod = coefficient * (float) nbPixels;
        int prodfin = (int) prod;


        if (x<width){

        if (y<height){ 

        if (tid <= prodfin){

        	output[tid].x = input[tid].x;

        	output[tid].z = output[tid].y = output[tid].x;

        }
        else{

        	output[tid].x = input2[tid].x; 

                output[tid].z = output[tid].y = output[tid].x;

        }

        }
        }

      
}


void Labwork::labwork6_GPU() {

      // Calculate number of pixels
 

    int pixelCount = inputImage->width * inputImage->height ;


    outputImage = static_cast<char *>(malloc(pixelCount * 3));

    //Allocate CUDA Memory

    uchar3 *devInput;
    uchar3 *devOutput;
    uchar3 *devGray;
    cudaMalloc(&devInput, pixelCount *sizeof(uchar3));
    cudaMalloc(&devOutput, pixelCount *sizeof(uchar3));
    cudaMalloc(&devGray, pixelCount *sizeof(uchar3));

    //For Blending
    uchar3 *devInput2;
    uchar3 *devGray2;
    cudaMalloc(&devInput2, pixelCount *sizeof(uchar3));
    cudaMalloc(&devGray2, pixelCount *sizeof(uchar3));

    //Copy CUDA Memory from CPU to GPU

    cudaMemcpy(devInput, inputImage->buffer, pixelCount * sizeof(uchar3), cudaMemcpyHostToDevice);

    //cudaMemcpy(devInput2, inputImage2->buffer, pixelCount * sizeof(uchar3), cudaMemcpyHostToDevice);

    //Processing

    dim3 blockSize = dim3(32, 32);

    int threshold = 128;

    int brightnessVar = 50; //choose your brightness between 0 and 100. A value of 50 will leave the brightness unchanged.
    
    float coefficient = 0.5;  //Blending Coefficient.

    int numBlockx = inputImage-> width / (blockSize.x) ;
    int numBlocky = inputImage-> height / (blockSize.y) ;
    if ((inputImage-> width % (blockSize.x)) > 0) {
        numBlockx++ ;
    }
    if ((inputImage-> height % (blockSize.y)) > 0){
        numBlocky++ ;
    }

    dim3 gridSize = dim3 (numBlockx,numBlocky);  

    grayScale3<<<gridSize, blockSize>>>(devInput, devGray, inputImage->width, inputImage->height);

    cudaFree(devInput);

    cudaMemcpy(devInput2, inputImage2->buffer, pixelCount * sizeof(uchar3), cudaMemcpyHostToDevice);

    grayScale3<<<gridSize, blockSize>>>(devInput2, devGray2, inputImage->width, inputImage->height);

    blendingGray<<<gridSize, blockSize>>>(devGray, devGray2, devOutput, inputImage->width, inputImage->height, coefficient);

    //Copy CUDA Memory from GPU to CPU

    cudaMemcpy(outputImage, devOutput, pixelCount * sizeof(uchar3), cudaMemcpyDeviceToHost);

    //Cleaning

    //cudaFree(devInput);
    cudaFree(devInput2);
    cudaFree(devOutput);

}

void Labwork::labwork7_GPU() {
}

void Labwork::labwork8_GPU() {
}

void Labwork::labwork9_GPU() {

}

void Labwork::labwork10_GPU(){
}


























