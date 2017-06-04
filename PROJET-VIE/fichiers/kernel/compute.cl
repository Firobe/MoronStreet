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

int newVal(unsigned oldVal, int neighbors) {
	return 0xFFFF00FF * ((oldVal == 0 && neighbors == 3) || (oldVal != 0 && (neighbors == 2 || neighbors == 3)));

}

/**
 * Jeu de la vie OpenCL (basique)
 */
__kernel void simpleMoron (__global unsigned *in, __global unsigned *out)
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
	out[j * DIM + i] = newVal(value, sum);
}


int countNeighbors(__local unsigned tile[TILEY + 2][TILEX + 2], int xloc,
						   int yloc) {
	int sum = 0;
	sum += (tile[yloc - 1][xloc - 1] != 0);
	sum += (tile[yloc - 1][xloc + 0] != 0);
	sum += (tile[yloc - 1][xloc + 1] != 0);
	sum += (tile[yloc + 0][xloc - 1] != 0);
	sum += (tile[yloc + 0][xloc + 1] != 0);
	sum += (tile[yloc + 1][xloc - 1] != 0);
	sum += (tile[yloc + 1][xloc + 0] != 0);
	sum += (tile[yloc + 1][xloc + 1] != 0);
	return sum;
}

__kernel void tiledMoron(__global unsigned *in, __global unsigned *out) {
	__local unsigned tile [TILEY + 2][TILEX + 2];
	int x = get_global_id (0);
	int y = get_global_id (1);
	int xloc = get_local_id (0) + 1;
	int yloc = get_local_id (1) + 1;

	//Tile initialization
	tile[yloc][xloc] = in[y * DIM + x];
	//Borders
	if(xloc == 1 && x != 0) tile[yloc][0] = in[y * DIM + (x - 1)];
	if(yloc == 1 && y != 0) tile[0][xloc] = in[(y - 1) * DIM + x];
	if(xloc == TILEX && x != DIM - 1)
		tile[yloc][TILEX + 1] = in[y * DIM + (x + 1)];
	if(yloc == TILEY && y != DIM - 1)
		tile[TILEY + 1][xloc] = in[(y + 1) * DIM + x];

	barrier (CLK_LOCAL_MEM_FENCE);

	//Actual computation
	if(x <= 0 || y <= 0 || x >= DIM - 1 || y >= DIM - 1) return;
	int sum = countNeighbors(tile, xloc, yloc);
	out[y * DIM + x] = newVal(tile[yloc][xloc], sum);
}

__kernel void advancedRetard(__global unsigned *in, __global unsigned *out,
				__global unsigned *inStag, __global unsigned *outStag) {
	//Booleans (check if a modif was made on West,East,North,South
	//borders or in the Center)
	__local unsigned stagW, stagE, stagN, stagS, stagC;
	__local unsigned tile [TILEY + 2][TILEX + 2];
	unsigned new = 0, old = 0;
	int x = get_global_id (0);
	int y = get_global_id (1);
	int xloc = get_local_id (0) + 1;
	int yloc = get_local_id (1) + 1;
	int tX = x / TILEX, tY = y / TILEY;

	//Init local var (by the first worker)
	if(xloc == 1 && yloc == 1)
		stagW = stagE = stagN = stagS = stagC = 0;

	//Ignore if stagnating
	if(inStag[tY * (DIM / TILEX) + tX] == 0) {

		//Tile initialization
		tile[yloc][xloc] = in[y * DIM + x];
		if(xloc == 1 && x != 0) tile[yloc][0] = in[y * DIM + (x - 1)];
		if(yloc == 1 && y != 0) tile[0][xloc] = in[(y - 1) * DIM + x];
		if(xloc == TILEX && x != DIM - 1)
			tile[yloc][TILEX + 1] = in[y * DIM + (x + 1)];
		if(yloc == TILEY && y != DIM - 1)
			tile[TILEY + 1][xloc] = in[(y + 1) * DIM + x];

		barrier (CLK_LOCAL_MEM_FENCE);

		//Actual computation
		if(x > 0 && y > 0 && x < DIM - 1 && y < DIM - 1) {
			old = tile[yloc][xloc]; //Old value
			int sum = countNeighbors(tile, xloc, yloc);
			new = newVal(old, sum); //New computed value
		}
		out[y * DIM + x] = new;

		//Check if a change happened
		unsigned change = (new != old);
		stagC += change;

		//Check if the change was on a border
		stagW += xloc == 1 && change;
		stagE += xloc == TILEX && change;
		stagN += yloc == 1 && change;
		stagS += yloc == TILEY && change;

		barrier (CLK_LOCAL_MEM_FENCE);
		
		//Update stagnation buffer (by the first worker)
		if(stagC != 0 && xloc == 1 && yloc == 1)
			for(int j = tY - (stagN != 0) ; j <= tY + (stagS != 0) ; ++j) 
				for(int i = tX - (stagW != 0) ; i <= tX + (stagE != 0) ; ++i) 
					if(i < DIM / TILEX && j < DIM / TILEY && i >= 0 && j >= 0)
						outStag[j * (DIM / TILEX) + i] = 0;
	}
	else {
		//Ensure that the kernel always meet two kernels
		barrier (CLK_LOCAL_MEM_FENCE);
		barrier (CLK_LOCAL_MEM_FENCE);
	}

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
