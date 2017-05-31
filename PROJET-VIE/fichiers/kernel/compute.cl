__kernel void transpose (__global unsigned *in, __global unsigned *out)
{
  __local unsigned tile [TILEX][TILEY+1];
  int x = get_global_id (0);
  int y = get_global_id (1);
  int xloc = get_local_id (0);
  int yloc = get_local_id (1);

  tile [xloc][yloc] = in [y * DIM + x];

  barrier (CLK_LOCAL_MEM_FENCE);

  out [(x - xloc + yloc) * DIM + y - yloc + xloc] = tile [yloc][xloc];
}

int countNeighbors(__global unsigned* in, int i, int j) {
  int sum = 0;
  sum += (in[(j - 1) * DIM + (i - 1)] != 0);
  sum += (in[(j - 1) * DIM + (i - 0)] != 0);
  sum += (in[(j - 1) * DIM + (i + 1)] != 0);
  sum += (in[(j - 0) * DIM + (i - 1)] != 0);
  sum += (in[(j - 0) * DIM + (i + 1)] != 0);
  sum += (in[(j + 1) * DIM + (i - 1)] != 0);
  sum += (in[(j + 1) * DIM + (i - 0)] != 0);
  sum += (in[(j + 1) * DIM + (i + 1)] != 0);
  return sum;
}

__kernel void simpleMoron (__global unsigned *in, __global unsigned *out)
{
  int i = get_global_id (0);
  int j = get_global_id (1);
  unsigned value = in [j * DIM + i];
  if(i <= 0 || j <= 0 || i >= DIM - 1 || j >= DIM - 1) return;
  int sum = countNeighbors(in, i, j);
  out[j * DIM + i] = 0xFFFF00FF * ((value == 0 && sum == 3) || (value != 0 && (sum == 2 || sum == 3)));
}

/**
 * Jeu de la vie OpenCL (basique)
 */
__kernel void advancedRetard (__global unsigned *in, __global unsigned *out)
{
  int i = get_global_id (0);
  int j = get_global_id (1);
  unsigned value = in [j * DIM + i];
  if(i <= 0 || j <= 0 || i >= DIM - 1 || j >= DIM - 1) return;
  
  int sum = 0;
  sum += (in[(j - 1) * DIM + (i - 1)] != 0);
  sum += (in[(j - 1) * DIM + (i - 0)] != 0);
  sum += (in[(j - 1) * DIM + (i + 1)] != 0);
  sum += (in[(j - 0) * DIM + (i - 1)] != 0);
  sum += (in[(j - 0) * DIM + (i + 1)] != 0);
  sum += (in[(j + 1) * DIM + (i - 1)] != 0);
  sum += (in[(j + 1) * DIM + (i - 0)] != 0);
  sum += (in[(j + 1) * DIM + (i + 1)] != 0);

  out[j * DIM + i] = 0xFFFF00FF * ((value == 0 && sum == 3) || (value != 0 && (sum == 2 || sum == 3)));
}


// NE PAS MODIFIER
static unsigned color_mean (unsigned c1, unsigned c2)
{
  uchar4 c;

  c.x = ((unsigned)(((uchar4 *) &c1)->x) + (unsigned)(((uchar4 *) &c2)->x)) / 2;
  c.y = ((unsigned)(((uchar4 *) &c1)->y) + (unsigned)(((uchar4 *) &c2)->y)) / 2;
  c.z = ((unsigned)(((uchar4 *) &c1)->z) + (unsigned)(((uchar4 *) &c2)->z)) / 2;
  c.w = ((unsigned)(((uchar4 *) &c1)->w) + (unsigned)(((uchar4 *) &c2)->w)) / 2;

  return (unsigned) c;
}

// NE PAS MODIFIER
static int4 color_to_int4 (unsigned c)
{
  uchar4 ci = *(uchar4 *) &c;
  return convert_int4 (ci);
}

// NE PAS MODIFIER
static unsigned int4_to_color (int4 i)
{
  return (unsigned) convert_uchar4 (i);
}



// NE PAS MODIFIER
static float4 color_scatter (unsigned c)
{
  uchar4 ci;

  ci.s0123 = (*((uchar4 *) &c)).s3210;
  return convert_float4 (ci) / (float4) 255;
}

// NE PAS MODIFIER: ce noyau est appelé lorsqu'une mise à jour de la
// texture de l'image affichée est requise
__kernel void update_texture (__global unsigned *cur, __write_only image2d_t tex)
{
  int y = get_global_id (1);
  int x = get_global_id (0);
  int2 pos = (int2)(x, y);
  unsigned c;

  c = cur [y * DIM + x];

  write_imagef (tex, pos, color_scatter (c));
}
