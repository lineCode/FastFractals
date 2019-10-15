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

    // Set up translation vertices once per block in shared memory
    __shared__ float2 vertices[3];
    if(threadIdx.x == 0)
    {
        vertices[0] = {-1.0f, -1.0f};
        vertices[1] = {1.0f, -1.0f};
        vertices[2] = {-1.0f, 1.0f};
    }
    __syncthreads();

    // Initially start at a translation vertex to guarantee we stay inside the
    // iterated function system
    int currentTarget = index % 3;
    float2 currentPosition = vertices[currentTarget];
   
    for(int i = index; i < numPoints; i += stride)
    {
        // set the current vertex to the currentPosition
        ptr[i].x = currentPosition.x;
        ptr[i].y = currentPosition.y;

        // set the iteration percentage and current target vertex
        ptr[i].z = (float) i / numPoints;
        ptr[i].w = currentTarget;

        // pick a random translation vertex and move halfway there
        currentTarget = curand_uniform(&state) * 3;
        currentPosition.x = (currentPosition.x + vertices[currentTarget].x)/2;
        currentPosition.y = (currentPosition.y + vertices[currentTarget].y)/2;
    }
}
