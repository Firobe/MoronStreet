
__kernel void vecmul(__global float *vec,
		     __global float *res,
		     float k)
{
  int index = get_global_id(0);
  int i = 0;
  for(i=0;i<10;i++){
    res[index] = vec[index]*k;
  }
}
