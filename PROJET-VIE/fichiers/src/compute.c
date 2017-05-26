
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

/**
 * Version séquentielle tuilée
 */
unsigned compute_v1(unsigned nb_iter){
	int tileX = 32;
	int tileY = 32;
	
    for (unsigned it = 1; it <= nb_iter; it ++) {
	
	for (int xT = 0; xT < DIM/tileX; xT++)
	for (int yT = 0; yT < DIM/tileY; yT++)
	for (int x = 0; x < tileX; x++)
	for (int y = 0; y < tileY; y++){
		int i = xT * tileX + x;
		int j = yT * tileY + y;
		int count = neighborCount(i, j);
					
		if(cur_img(i, j) != 0 && (count != 2 && count != 3))
			next_img(i, j) = 0;
		else if(cur_img(i, j) == 0 && count == 3)
			next_img(i, j) = couleur;
		else next_img (i, j) = cur_img (i, j);
	}

	swap_images ();
    }
    
    return 0; // on ne s'arrête jamais
}


/// 1. Ca serait pas mal de savoir ou tileX, tileY sont définie (pour l'instant pas globale je crois)
/// 2. A partir de la, on peut avoir mayChangeNext (matrix) en globale aussi.
void setMayChange(int* matrix, int tileX, int tileY, int xT, int yT, int x, int y){
	// xT
	matrix[xT][yT] = 1;
	if(y == 0 && yT > 0) matrix[xT][yT-1] = 1;
	else if(y == tileY-1 && yT < DIM/tileY - 1) matrix[xT][yT+1] = 1;
	
	// xT-1
	if(x == 0 && xT != 0){
		matrix[xT-1][yT] = 1;
		if(y == 0 && yT > 0) matrix[xT-1][yT-1] = 1;
		else if(y == tileY-1 && yT < DIM/tileY - 1) matrix[xT-1][yT+1] = 1;
	}
	// xT+1
	else if(x == tileX-1 && xT == DIM/tileX - 1){
		matrix[xT+1][yT] = 1;
		if(y == 0 && yT > 0) matrix[xT+1][yT-1] = 1;
		else if(y == tileY-1 && yT < DIM/tileY - 1) matrix[xT+1][yT+1] = 1;
	}
}


/**
 * Version séquentielle optimisée
 */
unsigned compute_v2(unsigned nb_iter) {
	int tileX = 32;
	int tileY = 32;
	int mayChangeCurr[DIM/tileX][DIM/tileY];
	int mayChangeNext[DIM/tileX][DIM/tileY];
	
	// Initialize mayChangeCurr at 0.
	for (int xT = 0; xT < DIM/tileX; xT++)
	for (int yT = 0; yT < DIM/tileY; yT++)
	mayChangeCurr[xT][yT] = 0;
	
    for (unsigned it = 1; it <= nb_iter; it ++) {
	// Reset mayChangeNext to 0.
	for (int xT = 0; xT < DIM/tileX; xT++)
	for (int yT = 0; yT < DIM/tileY; yT++)
	mayChangeNext[xT][yT] = 0;
	
	// Compute
	for (int xT = 0; xT < DIM/tileX; xT++)
	for (int yT = 0; yT < DIM/tileY; yT++){
		if(!mayChangeCurr[xT,yT]){ // No computing if not needed
			for (int x = 0; x < tileX; x++)
			for (int y = 0; y < tileY; y++){
				int i = xT * tileX + x;
				int j = yT * tileY + y;
				next_img (i, j) = cur_img (i, j);
			}
			
			continue; 
		}
	
		for (int x = 0; x < tileX; x++)
		for (int y = 0; y < tileY; y++){
			int i = xT * tileX + x;
			int j = yT * tileY + y;
			int count = neighborCount(i, j);
			
			if(cur_img(i, j) != 0 && (count != 2 && count != 3)){
				next_img(i, j) = 0;
				setMayChange(mayChangeNext,tileX,tileY,xT,yT,x,y);
			} else if(cur_img(i, j) == 0 && count == 3){
				next_img(i, j) = couleur;
				setMayChange(mayChangeNext,tileX,tileY,xT,yT,x,y);
			} else
				next_img (i, j) = cur_img (i, j);
		}
	}
	
	// Copy mayChangeNext to mayChangeCurr
	for (int xT = 0; xT < DIM/tileX; xT++)
	for (int yT = 0; yT < DIM/tileY; yT++)
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
