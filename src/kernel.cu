/**
 * Matthew Smith
 * github.com/mattonaise
 * Created on Oct 11, 2019
 *
 * kernel.cu
 **/

__global__ void kernel(float4* d_pointData, int numPoints)
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
        maps[0] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.16f, 0.01f};
        maps[1] = {0.0f, 1.6f, 0.85f, 0.04f, -0.04f, 0.85f, 0.85f};
        maps[2] = {0.0f, 1.6f, 0.2f, -0.26f, 0.23f, 0.22f, 0.07f};
        maps[3] = {0.0f, 0.44f, -0.15f, 0.28f, 0.26f, 0.24f, 0.07};
    }
    __syncthreads();

    // Initially start at a mapping vertex to guarantee we stay inside the
    // iterated function system
    int currentTarget = index % 4;
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
        float currentProb = curand_uniform(&state);
        float totalProb = 0.0f;
        for(int j = 0; j < 4; j++)
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
