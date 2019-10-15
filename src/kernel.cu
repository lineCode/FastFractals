/**
 * Matthew Smith
 * github.com/mattonaise
 * Created on Oct 11, 2019
 *
 * kernel.cu
 **/

__global__ void kernel(float4* ptr, int numPoints)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // If needed for performance, move curand_init to seperate kernel and store
    // states in device memory
    curandState state;
    curand_init((unsigned long long) clock(), index, 0, &state);

    for(int i = index; i < numPoints; i += stride)
    {
        ptr[i].x = curand_uniform(&state) * 2 - 1;
        ptr[i].y = curand_uniform(&state) * 2 - 1;
        ptr[i].z = curand_uniform(&state) * 2 - 1;
        ptr[i].w = 1.0f;
    }
}
