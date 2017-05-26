
#include "compute.h"
#include "graphics.h"
#include "debug.h"
#include "ocl.h"

#include <stdbool.h>

extern unsigned couleur;
unsigned version = 0;

void first_touch_v1 (void);
void first_touch_v2 (void);

unsigned compute_v0 (unsigned nb_iter);
unsigned compute_v1 (unsigned nb_iter);
unsigned compute_v2 (unsigned nb_iter);
unsigned compute_v3 (unsigned nb_iter);
unsigned compute_v4 (unsigned nb_iter);
unsigned compute_v5 (unsigned nb_iter);
unsigned compute_v6 (unsigned nb_iter);
unsigned compute_v7 (unsigned nb_iter);
unsigned compute_v8 (unsigned nb_iter);

void_func_t first_touch [] = {
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL
};

int_func_t compute [] = {
    compute_v0,
    compute_v1,
    compute_v2,
    compute_v3,
    compute_v4,
    compute_v5,
    compute_v6,
    compute_v7,
    compute_v8
};

char *version_name [] = {
    "Séquentielle",
    "Séquentielle tuilée",
    "Séquentielle optimisée",
    "OpenMP (for)",
    "OpenMP (for) tuilée",
    "OpenMP (for) optimisée",
    "OpenMP (task) tuilée",
    "OpenMP (task) optimisée",
    "OpenCL",
};

unsigned opencl_used [] = {
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1
};

///////////////////////////// Version séquentielle simple

int neighborCount(int i, int j) {
    int sum = 0;
    for(int a = i - 1 ; a <= i + 1 ; ++a)
	for(int b = j - 1 ; b <= j + 1 ; ++b)
	    if(a >= 0 && b >= 0 && a < DIM && b < DIM && (a != i || b != j))
		sum += (cur_img(a, b) == couleur);
    return sum;
}

/**
 * Version séquentielle basique
 */
unsigned compute_v0 (unsigned nb_iter)
{
    for (unsigned it = 1; it <= nb_iter; it ++) {
	for (int i = 0; i < DIM; i++)
	    for (int j = 0; j < DIM; j++){
		int count = neighborCount(i, j);
		if(cur_img(i, j) != 0 && (count != 2 && count != 3))
		    next_img(i, j) = 0;
		else if(cur_img(i, j) == 0 && count == 3)
		    next_img(i, j) = couleur;
		else next_img (i, j) = cur_img (i, j);
	    }

	swap_images ();
    }
    // retourne le nombre d'étapes nécessaires à la
    // stabilisation du calcul ou bien 0 si le calcul n'est pas
    // stabilisé au bout des nb_iter itérations
    return 0;
}

#define TILE_X (32)
#define TILE_Y (32)

void computeOneTile(int xT, int yT) {
    for (int x = 0; x < TILE_X; x++)
	for (int y = 0; y < TILE_Y; y++){
	    int i = xT * TILE_X + x;
	    int j = yT * TILE_Y + y;
	    int count = neighborCount(i, j);

	    if(cur_img(i, j) != 0 && (count != 2 && count != 3))
		next_img(i, j) = 0;
	    else if(cur_img(i, j) == 0 && count == 3)
		next_img(i, j) = couleur;
	    else next_img (i, j) = cur_img (i, j);
	}
}

/**
 * Version séquentielle tuilée
 */
unsigned compute_v1(unsigned nb_iter){
    for (unsigned it = 1; it <= nb_iter; it ++) {

	for (int xT = 0; xT < DIM/TILE_X; xT++)
	    for (int yT = 0; yT < DIM/TILE_Y; yT++)
		computeOneTile(xT, yT);
	swap_images ();
    }

    return 0; // on ne s'arrête jamais
}

void setMayChange(int matrix[DIM/TILE_X][DIM/TILE_Y], int xT, int yT, int x, int y){
    // xT
    matrix[xT][yT] = 1;
    if(y == 0 && yT > 0) matrix[xT][yT-1] = 1;
    else if(y == TILE_Y-1 && yT < DIM/TILE_Y - 1) matrix[xT][yT+1] = 1;

    // xT-1
    if(x == 0 && xT != 0){
	matrix[xT-1][yT] = 1;
	if(y == 0 && yT > 0) matrix[xT-1][yT-1] = 1;
	else if(y == TILE_Y-1 && yT < DIM/TILE_Y - 1) matrix[xT-1][yT+1] = 1;
    }
    // xT+1
    else if(x == TILE_X-1 && xT != DIM/TILE_X - 1){
	matrix[xT+1][yT] = 1;
	if(y == 0 && yT > 0) matrix[xT+1][yT-1] = 1;
	else if(y == TILE_Y-1 && yT < DIM/TILE_Y - 1) matrix[xT+1][yT+1] = 1;
    }
}


/**
 * Version séquentielle optimisée
 */
unsigned compute_v2(unsigned nb_iter) {
    int mayChangeCurr[DIM/TILE_X][DIM/TILE_Y];
    int mayChangeNext[DIM/TILE_X][DIM/TILE_Y];

    // Initialize mayChangeCurr at 0.
    for (int xT = 0; xT < DIM/TILE_X; xT++)
	for (int yT = 0; yT < DIM/TILE_Y; yT++)
	    mayChangeCurr[xT][yT] = 1;

    for (unsigned it = 1; it <= nb_iter; it ++) {
	// Reset mayChangeNext to 0.
	for (int xT = 0; xT < DIM/TILE_X; xT++)
	    for (int yT = 0; yT < DIM/TILE_Y; yT++)
		mayChangeNext[xT][yT] = 0;

	// Compute
	for (int xT = 0; xT < DIM/TILE_X; xT++)
	    for (int yT = 0; yT < DIM/TILE_Y; yT++){
		// No computing if not needed
		if(!mayChangeCurr[xT][yT]){ 
		    //Copy the tile
		    for (int x = 0; x < TILE_X; x++)
			for (int y = 0; y < TILE_Y; y++){
			    int i = xT * TILE_X + x;
			    int j = yT * TILE_Y + y;
			    next_img (i, j) = cur_img (i, j);
			}
		    continue; //Next tile
		}

		//Iterate over the tile
		for (int x = 0; x < TILE_X; x++)
		    for (int y = 0; y < TILE_Y; y++){
			int i = xT * TILE_X + x;
			int j = yT * TILE_Y + y;
			int count = neighborCount(i, j);

			if(cur_img(i, j) != 0 && (count != 2 && count != 3)){
			    next_img(i, j) = 0;
			    //A change occured
			    setMayChange(mayChangeNext, xT, yT, x, y);
			} else if(cur_img(i, j) == 0 && count == 3){
			    next_img(i, j) = couleur;
			    //A change occured
			    setMayChange(mayChangeNext, xT, yT, x, y);
			} else
			    next_img (i, j) = cur_img (i, j);
		    }
	    }

	// Copy mayChangeNext to mayChangeCurr
	for (int xT = 0; xT < DIM/TILE_X; xT++)
	    for (int yT = 0; yT < DIM/TILE_Y; yT++)
		mayChangeCurr[xT][yT] = mayChangeNext[xT][yT];

	swap_images ();
    }
    return 0;
}

/**
 * Version OpenMP (for) DE BASE
 */
void first_touch_v3 ()
{
    int i,j ;

#pragma omp parallel for
    for(i=0; i<DIM ; i++) {
	for(j=0; j < DIM ; j += 512)
	    next_img (i, j) = cur_img (i, j) = 0 ;
    }
}

// Renvoie le nombre d'itérations effectuées avant stabilisation, ou 0
unsigned compute_v3(unsigned nb_iter) {
    for (unsigned it = 1; it <= nb_iter; it ++) {
#pragma omp parallel for schedule(static, 100) collapse(2)
	for (int i = 0; i < DIM; i++)
	    for (int j = 0; j < DIM; j++){
		int count = neighborCount(i, j);
		if(cur_img(i, j) != 0 && (count != 2 && count != 3))
		    next_img(i, j) = 0;
		else if(cur_img(i, j) == 0 && count == 3)
		    next_img(i, j) = couleur;
		else next_img (i, j) = cur_img (i, j);
	    }

	swap_images ();
    }
    return 0;
}

/**
 * Version OpenMP (for) tuilée
 */
unsigned compute_v4(unsigned nb_iter) {
    return 0;
}

/**
 * Version OpenMP (for) optimisée
 */
unsigned compute_v5(unsigned nb_iter) {
    return 0;
}

/**
 * Version OpenMP (task) tuilée
 */
unsigned compute_v6(unsigned nb_iter) {
    return 0;
}

/**
 * Version OpenMP (task) optimisée
 */
unsigned compute_v7(unsigned nb_iter) {
    return 0;
}

/**
 * Version OpenCL (kernels simpleMoron et advancedRetard)
 */
unsigned compute_v8 (unsigned nb_iter)
{
    return ocl_compute (nb_iter);
}
