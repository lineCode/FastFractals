/**
 * Matthew Smith
 * github.com/mattonaise
 * Created on Oct 11, 2019
 *
 * kernel.cu
 **/

__global__ void kernel(float4* d_pointData, int numPoints,
        mapping* d_mappings, int numMappings)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // If needed for performance, move curand_init to seperate kernel and store
    // states in device memory
    curandState state;
    curand_init((unsigned long long) clock(), index, 0, &state);

    // Set up transformation mapping once per block in shared memory
    extern __shared__ mapping maps[];
    if(threadIdx.x == 0)
    {
        for(int i = 0; i < numMappings; i++)
            maps[i] = d_mappings[i];
    }
    __syncthreads();

    // Initially start at a mapping vertex to guarantee we stay inside the
    // iterated function system
    int currentTarget = index % numMappings;
    float2 currentPosition, newPosition;
    currentPosition.x = maps[currentTarget].x;
    currentPosition.y = maps[currentTarget].y;
   
    for(int i = index; i < numPoints; i += stride)
    {
        // set the current vertex to the currentPosition
        d_pointData[i].x = currentPosition.x;
        d_pointData[i].y = currentPosition.y;

        // set the iteration percentage and current target mapping
        d_pointData[i].z =  i / (float) numPoints;
        d_pointData[i].w = currentTarget;

        // find random target with given mapping probabilities
        // If needed for performance, find method to remove thread divergence
        // Note: changing 4 to numMappings in for loop reduced performance 50%
        float currentProb = curand_uniform(&state);
        float totalProb = 0.0f;
        for(int j = 0; j < numMappings; j++)
        {
            totalProb += maps[j].p;
            if(currentProb < totalProb)
            {
                currentTarget = j;
                break;
            }
        }

        // calculate the transformation
        // (x_n+1) = (a b)(x_n) + (e)
        // (y_n+1)   (c d)(y_n)   (f)
        newPosition.x = maps[currentTarget].a * currentPosition.x +
                        maps[currentTarget].b * currentPosition.y +
                        maps[currentTarget].x;
        newPosition.y = maps[currentTarget].c * currentPosition.x +
                        maps[currentTarget].d * currentPosition.y +
                        maps[currentTarget].y;
        currentPosition = newPosition;
    }
}
