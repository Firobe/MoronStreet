
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

static unsigned blur_mean (unsigned c1, unsigned c2, unsigned c3, unsigned c4, unsigned c5)
{
  uchar4 c;

  c.x = ((unsigned)(((uchar4 *) &c1)->x) + (unsigned)(((uchar4 *) &c2)->x) +
	(unsigned)(((uchar4 *) &c3)->x) + (unsigned)(((uchar4 *) &c4)->x) +
	(unsigned)(((uchar4 *) &c5)->x)) / 5;

  c.y = ((unsigned)(((uchar4 *) &c1)->y) + (unsigned)(((uchar4 *) &c2)->y) + 
	(unsigned)(((uchar4 *) &c3)->y) + (unsigned)(((uchar4 *) &c4)->y) +
	(unsigned)(((uchar4 *) &c5)->y)) / 5;

  c.z = ((unsigned)(((uchar4 *) &c1)->z) + (unsigned)(((uchar4 *) &c2)->z) +
	(unsigned)(((uchar4 *) &c3)->z) + (unsigned)(((uchar4 *) &c4)->z) +
	(unsigned)(((uchar4 *) &c5)->z)) / 5;

  c.w = ((unsigned)(((uchar4 *) &c1)->w) + (unsigned)(((uchar4 *) &c2)->w) +
	(unsigned)(((uchar4 *) &c3)->w) + (unsigned)(((uchar4 *) &c4)->w) +
	(unsigned)(((uchar4 *) &c5)->w)) / 5;

  return (unsigned) c;
}

__kernel void scrollup (__global unsigned *in, __global unsigned *out)
{
  int y = get_global_id (1);
  int x = get_global_id (0);
  unsigned couleur;

  couleur = in [y * DIM + x];

  y = (y ? y - 1 : get_global_size (1) - 1);

  out [y * DIM + x] = couleur;
}

static unsigned saturate (unsigned input)
{
  for (int i = 0; i < 7; i++)
    input = (input << 1) | input;

  return input;
}

__kernel void divergence (__global unsigned *in, __global unsigned *out)
{
  int y = get_global_id (1);
  int x = get_global_id (0);
  int b = 3;

  if (((x >> b) & 1) == 0)
    out [y * DIM + x] = saturate (0x01000001); // will become red
  else
    out [y * DIM + x] = saturate (0x01010001); // will be yellow
}


__kernel void transpose_naif (__global unsigned *in, __global unsigned *out)
{
  int x = get_global_id (0);
  int y = get_global_id (1);

  out [x * DIM + y] = in [y * DIM + x];
}


__kernel void transpose (__global unsigned *in, __global unsigned *out)
{
  __local unsigned tile[TILEY][TILEX+1];
  int x_loc = get_local_id(0);
  int y_loc = get_local_id(1);
  int x_xloc = get_group_id(0)*TILEX;
  int y_yloc = get_group_id(1)*TILEY;
  
  tile[x_loc][y_loc] = in[(y_yloc + y_loc) * DIM + x_xloc + x_loc];
  barrier(CLK_LOCAL_MEM_FENCE);
  out[(x_xloc + y_loc) * DIM + y_yloc + x_loc] = tile[y_loc][x_loc];
}


#define PIX_BLOC 16

// ATTENTION: les tuiles doivent être multiples de PIX_BLOC x PIX_BLOC
__kernel void pixellize (__global unsigned *in, __global unsigned *out)
{
  __local unsigned tile[TILEY][TILEX];
  int x_loc = get_local_id(0);
  int y_loc = get_local_id(1);
  int x = get_global_id(0);
  int y = get_global_id(1);
  tile[y_loc][x_loc] = in[y * DIM + x];
  int m = 1;
  int pm = 0;
  int n = 1;

  while(n < PIX_BLOC) {
	barrier(CLK_LOCAL_MEM_FENCE);
  	if((x_loc & m) == 0){
		if((y_loc & pm) == 0)
			tile[y_loc][x_loc] = color_mean(tile[y_loc][x_loc], tile[y_loc][x_loc + n]); 
		barrier(CLK_LOCAL_MEM_FENCE);
		if((y_loc & m) == 0)
			tile[y_loc][x_loc] = color_mean(tile[y_loc][x_loc], tile[y_loc + n][x_loc]); 
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	pm = m;
	n *= 2;
	m = (m << 1) + 1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  unsigned mask = ~(PIX_BLOC - 1);
  out[y * DIM + x] = tile[y_loc & mask][x_loc & mask];
}



// Flou utilisant la moyenne pondérée des 8 pixels environnants :
// celui du centre pèse 8, les autres 1. Les pixels se trouvant sur
// les bords sont également traités (mais ils ont moins de 8 voisins).
__kernel void blur (__global unsigned *in, __global unsigned *out)
{
  __local unsigned tile[TILEY + 2][TILEX + 2];
  int x_loc = get_local_id(0)+1;
  int y_loc = get_local_id(1)+1;
  int x = get_global_id(0);
  int y = get_global_id(1);
  tile[y_loc][x_loc] = in[y * DIM + x];
  if(x_loc == 1)
  	tile[y_loc][0] = in[y * DIM + (x - (x > 0))];
  if(y_loc == 1)
  	tile[0][x_loc] = in[(y - (y > 0)) * DIM + x];
  if(x_loc == TILEX)
  	tile[y_loc][TILEX + 1] = in[y * DIM + (x + (x < TILEX))];
  if(y_loc == TILEY)
  	tile[TILEY + 1][x_loc] = in[(y + (y < TILEY)) * DIM + x];
  barrier(CLK_LOCAL_MEM_FENCE);
  out[y * DIM + x] = 
  	blur_mean(tile[y_loc][x_loc], tile[y_loc - 1][x_loc], tile[y_loc + 1][x_loc], tile[y_loc][x_loc - 1], tile[y_loc][x_loc + 1]);
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
