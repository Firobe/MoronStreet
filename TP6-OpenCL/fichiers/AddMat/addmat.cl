
__kernel void addmat(__global float *A,
		     __global float *B,
		     __global float *C)
{
  int index = get_global_id(1) * get_global_size(0) + get_global_id(0);
 
  C[index] = A[index] + B[index];
}
