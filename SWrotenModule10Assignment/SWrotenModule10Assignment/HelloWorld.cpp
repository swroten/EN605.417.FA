//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Timothy Mattson, James Fung, Dan Ginsburg
// ISBN-10:   0-321-74964-2
// ISBN-13:   978-0-321-74964-2
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/9780132488006/
//            http://www.openclprogrammingguide.com
//

// HelloWorld.cpp
//
//    This is a simple example that demonstrates basic OpenCL setup and
//    use.

#include <iostream>
#include <fstream>
#include <sstream>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

//  Constants
const int ARRAY_SIZE = 500;

//  Create an OpenCL context on the first available platform using
//  either a GPU or CPU depending on what is available.
cl_context CreateContext()
{
	cl_int errNum;
	cl_uint numPlatforms;
	cl_platform_id firstPlatformId;
	cl_context context = NULL;

	// First, select an OpenCL platform to run on.  For this example, we
	// simply choose the first available platform.  Normally, you would
	// query for all available platforms and select the most appropriate one.
	errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);
	if (errNum != CL_SUCCESS || numPlatforms <= 0)
	{
		std::cerr << "Failed to find any OpenCL platforms." << std::endl;
		return NULL;
	}

	// Next, create an OpenCL context on the platform.  Attempt to
	// create a GPU-based context, and if that fails, try to create
	// a CPU-based context.
	cl_context_properties contextProperties[] =
	{
		CL_CONTEXT_PLATFORM,
		(cl_context_properties)firstPlatformId,
		0
	};
	context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU, NULL, NULL, &errNum);
	if (errNum != CL_SUCCESS)
	{
		std::cout << "Could not create GPU context, trying CPU..." << std::endl;
		context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_CPU, NULL, NULL, &errNum);
		if (errNum != CL_SUCCESS)
		{
			std::cerr << "Failed to create an OpenCL GPU or CPU context." << std::endl;
			return NULL;
		}
	}

	return context;
}

//  Create a command queue on the first device available on the context
cl_command_queue CreateCommandQueue(cl_context context, cl_device_id *device)
{
	cl_int errNum;
	cl_device_id *devices;
	cl_command_queue commandQueue = NULL;
	size_t deviceBufferSize = -1;

	// First get the size of the devices buffer
	errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);
	if (errNum != CL_SUCCESS)
	{
		std::cerr << "Failed call to clGetContextInfo(...,GL_CONTEXT_DEVICES,...)";
		return NULL;
	}

	if (deviceBufferSize <= 0)
	{
		std::cerr << "No devices available.";
		return NULL;
	}

	// Allocate memory for the devices buffer
	devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
	errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);
	if (errNum != CL_SUCCESS)
	{
		delete[] devices;
		std::cerr << "Failed to get device IDs";
		return NULL;
	}

	// In this example, we just choose the first available device.  In a
	// real program, you would likely use all available devices or choose
	// the highest performance device based on OpenCL device queries
	commandQueue = clCreateCommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, NULL);
	if (commandQueue == NULL)
	{
		delete[] devices;
		std::cerr << "Failed to create commandQueue for device 0";
		return NULL;
	}

	*device = devices[0];
	delete[] devices;
	return commandQueue;
}

//  Create an OpenCL program from the kernel source file
cl_program CreateProgram(cl_context context, cl_device_id device, const char* fileName)
{
	cl_int errNum;
	cl_program program;

	std::ifstream kernelFile(fileName, std::ios::in);
	if (!kernelFile.is_open())
	{
		std::cerr << "Failed to open file for reading: " << fileName << std::endl;
		return NULL;
	}

	std::ostringstream oss;
	oss << kernelFile.rdbuf();

	std::string srcStdStr = oss.str();
	const char *srcStr = srcStdStr.c_str();
	program = clCreateProgramWithSource(context, 1,
		(const char**)&srcStr,
		NULL, NULL);
	if (program == NULL)
	{
		std::cerr << "Failed to create CL program from source." << std::endl;
		return NULL;
	}

	errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (errNum != CL_SUCCESS)
	{
		// Determine the reason for the error
		char buildLog[16384];
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
			sizeof(buildLog), buildLog, NULL);

		std::cerr << "Error in kernel: " << std::endl;
		std::cerr << buildLog;
		clReleaseProgram(program);
		return NULL;
	}

	return program;
}

//  Create memory objects used as the arguments to the kernel
//  The kernel takes three arguments: result (output), a (input),
//  and b (input)
bool CreateMemObjects(cl_context context, cl_mem memObjects[3],
	float *a, float *b)
{
	memObjects[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) * ARRAY_SIZE, a, NULL);
	memObjects[1] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) * ARRAY_SIZE, b, NULL);
	memObjects[2] = clCreateBuffer(context, CL_MEM_READ_WRITE,
		sizeof(float) * ARRAY_SIZE, NULL, NULL);

	if (memObjects[0] == NULL || memObjects[1] == NULL || memObjects[2] == NULL)
	{
		std::cerr << "Error creating memory objects." << std::endl;
		return false;
	}

	return true;
}

//  Cleanup any created OpenCL resources
void Cleanup(cl_context context, cl_command_queue commandQueue, cl_program program, cl_mem memObjects[3])
{
	for (int i = 0; i < 3; i++)
	{
		if (memObjects[i] != 0)
			clReleaseMemObject(memObjects[i]);
	}
	if (commandQueue != 0)
		clReleaseCommandQueue(commandQueue);
	
	if (program != 0)
		clReleaseProgram(program);

	if (context != 0)
		clReleaseContext(context);

}

cl_ulong RunKernel(cl_kernel kernel, cl_context context, cl_command_queue commandQueue, cl_program program, cl_mem memObjects[3])
{
	// Initialize Variables
	cl_int errNum;
	cl_int profiling_info;
	cl_ulong stopTime = 0;
	cl_ulong startTime = 0;
	cl_ulong kernelTime = 0;
	float result[ARRAY_SIZE];
	cl_event startStopEvent = 0;
	size_t localWorkSize[1] = { 1 };
	size_t globalWorkSize[1] = { ARRAY_SIZE };
	
	// Check if failed to create
	if (kernel == NULL)
	{
		// Log Failure in Console
		std::cerr << "Failed to create kernel" << std::endl;

		// Clean up Kernel
		if (kernel != 0)
		{
			clReleaseKernel(kernel);
		}

		// Clean Up
		Cleanup(context, commandQueue, program, memObjects);

		// Return
		return 1;
	}

	// Set the kernel arguments (result, a, b)
	errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &memObjects[0]);
	errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &memObjects[1]);
	errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &memObjects[2]);

	// Check if failed to set Kernel Arguments
	if (errNum != CL_SUCCESS)
	{
		// Log Failure in Console
		std::cerr << "Error setting kernel arguments." << std::endl;

		// Clean Up
		Cleanup(context, commandQueue, program, memObjects);

		// Return
		return 1;
	}

	// Queue the kernel up for execution across the array
	errNum = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, &startStopEvent);

	// Check if failed to Queue Kernel for Execution
	if (errNum != CL_SUCCESS)
	{
		// Log Failure in Console
		std::cerr << "Error queuing kernel for execution." << std::endl;

		// Clean Up
		Cleanup(context, commandQueue, program, memObjects);

		// Return
		return 1;
	}

	// Wait for Events to End to Get Start and Stop Time
	profiling_info = clWaitForEvents(1, &startStopEvent);
	profiling_info |= clGetEventProfilingInfo(startStopEvent, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &startTime, NULL);
	profiling_info |= clGetEventProfilingInfo(startStopEvent, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &stopTime, NULL);

	if (profiling_info != CL_SUCCESS)
	{
		// Log Failure in Console
		std::cerr << "Error computing execution time." << std::endl;

		// Clean Up
		Cleanup(context, commandQueue, program, memObjects);

		// Return
		return 1;
	}

	// Compute Kernel Time
	kernelTime = (stopTime - startTime);

	// Read the output buffer back to the Host
	errNum = clEnqueueReadBuffer(commandQueue, memObjects[2], CL_TRUE, 0, ARRAY_SIZE * sizeof(float), result, 0, NULL, NULL);

	// Check if failed to Read Output buffer back to host
	if (errNum != CL_SUCCESS)
	{
		// Log Failure in Console
		std::cerr << "Error reading result buffer." << std::endl;

		// Clean Up
		Cleanup(context, commandQueue, program, memObjects);

		// Return
		return 1;
	}

	// Output the result buffer
	for (int i = 0; i < ARRAY_SIZE; i++)
	{
		std::cout << result[i] << " ";
	}

	// Add Line for Spacing
	std::cout << std::endl;

	// Clean up
	clReleaseKernel(kernel);
	clReleaseEvent(startStopEvent);
		
	// Return Success
	return kernelTime;
}

int main(int argc, char** argv)
{
	// Initialize Variables
	float a[ARRAY_SIZE];
	float b[ARRAY_SIZE];
	cl_ulong subKernelTime;
	cl_ulong mulKernelTime;
	cl_ulong divKernelTime;
	cl_ulong powKernelTime;
	cl_ulong addKernelTime;
	cl_context context = 0;
	cl_program program = 0;
	cl_device_id device = 0;
	cl_command_queue commandQueue = 0;
	cl_mem memObjects[3] = { 0, 0, 0 };

	// Create an OpenCL context on first available platform
	context = CreateContext();

	// Verify Successfully created
	if (context == NULL)
	{
		// Otherwise log error
		std::cerr << "Failed to create OpenCL context." << std::endl;

		// Return 
		return 1;
	}

	// Create memory objects that will be used as arguments to
	// kernel.  First create host memory arrays that will be
	// used to store the arguments to the kernel
	for (int i = 0; i < ARRAY_SIZE; i++)
	{
		a[i] = (float)i;
		b[i] = (float)(i * 2);
	}

	// Verify Successfully created
	if (!CreateMemObjects(context, memObjects, a, b))
	{
		// Otherwise Clean-up
		Cleanup(context, commandQueue, program, memObjects);

		// Return 
		return 1;
	}

	// Create a command-queue on the first device available
	// on the created context
	commandQueue = CreateCommandQueue(context, &device);

	// Verify Successfully created
	if (commandQueue == NULL)
	{
		// Otherwise Clean-up
		Cleanup(context, commandQueue, program, memObjects);

		// Return 
		return 1;
	}

	// Create OpenCL program from HelloWorld.cl kernel source
	program = CreateProgram(context, device, "HelloWorld.cl");

	// Verify Successfully created
	if (program == NULL)
	{
		// Otherwise Clean-up
		Cleanup(context, commandQueue, program, memObjects);

		// Return 
		return 1;
	}
	
	// Write Heading to Console
	std::cout << std::endl;
	std::cout << "Performing Add Kernel:" << std::endl;

	// Run the Add Kernel
	addKernelTime = RunKernel(clCreateKernel(program, "add", NULL), context, commandQueue, program, memObjects);

	// Write Heading to Console
	std::cout << std::endl;
	std::cout << "Performing Subtract Kernel:" << std::endl;

	// Run the Subtract Kernel
	subKernelTime = RunKernel(clCreateKernel(program, "sub", NULL), context, commandQueue, program, memObjects);

	// Write Heading to Console
	std::cout << std::endl;
	std::cout << "Performing Mutliply Kernel:" << std::endl;

	// Run the Multiply Kernel
	mulKernelTime = RunKernel(clCreateKernel(program, "mult", NULL), context, commandQueue, program, memObjects);

	// Write Heading to Console
	std::cout << std::endl;
	std::cout << "Performing Divide Kernel:" << std::endl;

	// Run the Divide Kernel
	divKernelTime = RunKernel(clCreateKernel(program, "div", NULL), context, commandQueue, program, memObjects);

	// Write Heading to Console
	std::cout << std::endl;
	std::cout << "Performing Power Kernel:" << std::endl;

	// Run the Power Kernel
	powKernelTime = RunKernel(clCreateKernel(program, "power", NULL), context, commandQueue, program, memObjects);
	
	// Output Kernel Times
	std::cout << std::endl;
	std::cout << "Kernel Execution Times:" << std::endl;
	std::cout << "  ADD Kernel: " << addKernelTime << " ns" << std::endl;
	std::cout << "  SUB Kernel: " << subKernelTime << " ns" << std::endl;
	std::cout << "  MUL Kernel: " << mulKernelTime << " ns" << std::endl;
	std::cout << "  DIV Kernel: " << divKernelTime << " ns" << std::endl;
	std::cout << "  POW Kernel: " << powKernelTime << " ns" << std::endl;
	std::cout << std::endl;

	// Write to Console Success
	std::cout << "Executed program succesfully." << std::endl;
	std::cout << std::endl;

	// Clean up all Allocated Resources
	Cleanup(context, commandQueue, program, memObjects);

	// Return Success
	return 0;
}
