/* Euclidean bundle adjustment demo using the sba package */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <sba.h>
#include "../compiler.h"
#include "eucsbademo.h"
#include "readparams.h"

#define CLOCKS_PER_MSEC (CLOCKS_PER_SEC/1000.0)

#define MAXITER         100
#define MAXITER2        150


/* pointers to additional data, used for computed image projections and their jacobians */
struct globs_{
	double *rot0params; /* initial rotation parameters, combined with a local rotation parameterization */
	double *intrcalib; /* the 5 intrinsic calibration parameters in the order [fu, u0, v0, ar, skew],
                      * where ar is the aspect ratio fv/fu.
                      * Used only when calibration is fixed for all cameras;
                      * otherwise, it is null and the intrinsic parameters are
                      * included in the set of motion parameters for each camera
                      */
  int nccalib; /* number of calibration parameters that must be kept constant.
                * 0: all parameters are free 
                * 1: skew is fixed to its initial value, all other parameters vary (i.e. fu, u0, v0, ar) 
                * 2: skew and aspect ratio are fixed to their initial values, all other parameters vary (i.e. fu, u0, v0)
                * 3: meaningless
                * 4: skew, aspect ratio and principal point are fixed to their initial values, only the focal length varies (i.e. fu)
                * 5: all intrinsics are kept fixed to their initial values
                * >5: meaningless
                * Used only when calibration varies among cameras
                */

  int ncdist; /* number of distortion parameters in Bouguet's model that must be kept constant.
               * 0: all parameters are free 
               * 1: 6th order radial distortion term (kc[4]) is fixed
               * 2: 6th order radial distortion and one of the tangential distortion terms (kc[3]) are fixed
               * 3: 6th order radial distortion and both tangential distortion terms (kc[3], kc[2]) are fixed [i.e., only 2nd & 4th order radial dist.]
               * 4: 4th & 6th order radial distortion terms and both tangential distortion ones are fixed [i.e., only 2nd order radial dist.]
               * 5: all distortion parameters are kept fixed to their initial values
               * >5: meaningless
               * Used only when calibration varies among cameras and distortion is to be estimated
               */
  int cnp, pnp, mnp; /* dimensions */

	double *ptparams; /* needed only when bundle adjusting for camera parameters only */
	double *camparams; /* needed only when bundle adjusting for structure parameters only */
} globs;

/* unit quaternion from vector part */
#define _MK_QUAT_FRM_VEC(q, v){                                     \
  (q)[1]=(v)[0]; (q)[2]=(v)[1]; (q)[3]=(v)[2];                      \
  (q)[0]=sqrt(1.0 - (q)[1]*(q)[1] - (q)[2]*(q)[2]- (q)[3]*(q)[3]);  \
}

/*
 * multiplication of the two quaternions in q1 and q2 into p
 */
inline static void quatMult(double q1[FULLQUATSZ], double q2[FULLQUATSZ], double p[FULLQUATSZ])
{
  p[0]=q1[0]*q2[0]-q1[1]*q2[1]-q1[2]*q2[2]-q1[3]*q2[3];
  p[1]=q1[0]*q2[1]+q2[0]*q1[1]+q1[2]*q2[3]-q1[3]*q2[2];
  p[2]=q1[0]*q2[2]+q2[0]*q1[2]+q2[1]*q1[3]-q1[1]*q2[3];
  p[3]=q1[0]*q2[3]+q2[0]*q1[3]+q1[1]*q2[2]-q2[1]*q1[2];
}

/*
 * fast multiplication of the two quaternions in q1 and q2 into p
 * this is the second of the two schemes derived in pg. 8 of
 * T. D. Howell, J.-C. Lafon, The complexity of the quaternion product, TR 75-245, Cornell Univ., June 1975.
 *
 * total additions increase from 12 to 27 (28), but total multiplications decrease from 16 to 9 (12)
 */
inline static void quatMultFast(double q1[FULLQUATSZ], double q2[FULLQUATSZ], double p[FULLQUATSZ])
{
double t1, t2, t3, t4, t5, t6, t7, t8, t9;
//double t10, t11, t12;

  t1=(q1[0]+q1[1])*(q2[0]+q2[1]);
  t2=(q1[3]-q1[2])*(q2[2]-q2[3]);
  t3=(q1[1]-q1[0])*(q2[2]+q2[3]);
  t4=(q1[2]+q1[3])*(q2[1]-q2[0]);
  t5=(q1[1]+q1[3])*(q2[1]+q2[2]);
  t6=(q1[1]-q1[3])*(q2[1]-q2[2]);
  t7=(q1[0]+q1[2])*(q2[0]-q2[3]);
  t8=(q1[0]-q1[2])*(q2[0]+q2[3]);

#if 0
  t9 =t5+t6;
  t10=t7+t8;
  t11=t5-t6;
  t12=t7-t8;

  p[0]= t2 + 0.5*(-t9+t10);
  p[1]= t1 - 0.5*(t9+t10);
  p[2]=-t3 + 0.5*(t11+t12);
  p[3]=-t4 + 0.5*(t11-t12);
#endif

  /* following fragment it equivalent to the one above */
  t9=0.5*(t5-t6+t7+t8);
  p[0]= t2 + t9-t5;
  p[1]= t1 - t9-t6;
  p[2]=-t3 + t9-t8;
  p[3]=-t4 + t9-t7;
}


/* Routines to estimate the estimated measurement vector (i.e. "func") and
 * its sparse jacobian (i.e. "fjac") needed in BA. Code below makes use of the
 * routines calcImgProj() and calcImgProjJacXXX() which
 * compute the predicted projection & jacobian of a SINGLE 3D point (see imgproj.c).
 * In the terminology of TR-340, these routines compute Q and its jacobians A=dQ/da, B=dQ/db.
 * Notice also that what follows is two pairs of "func" and corresponding "fjac" routines.
 * The first is to be used in full (i.e. motion + structure) BA, the second in 
 * motion only BA.
 */

static const double zerorotquat[FULLQUATSZ]={1.0, 0.0, 0.0, 0.0};

/****************************************************************************************/
/* MEASUREMENT VECTOR AND JACOBIAN COMPUTATION FOR VARYING CAMERA POSE AND 3D STRUCTURE */
/****************************************************************************************/

/*** MEASUREMENT VECTOR AND JACOBIAN COMPUTATION FOR THE SIMPLE DRIVERS ***/

/* FULL BUNDLE ADJUSTMENT, I.E. SIMULTANEOUS ESTIMATION OF CAMERA AND STRUCTURE PARAMETERS */

/* Given the parameter vectors aj and bi of camera j and point i, computes in xij the
 * predicted projection of point i on image j
 */
static void img_projRTS(int j, int i, double *aj, double *bi, double *xij, void *adata)
{
  double *Kparms, *pr0;
  struct globs_ *gl;

  gl=(struct globs_ *)adata;
  Kparms=gl->intrcalib;
  pr0=gl->rot0params+j*FULLQUATSZ; // full quat for initial rotation estimate

  calcImgProj(Kparms, pr0, aj, aj+3, bi, xij); // 3 is the quaternion's vector part length
}

/* Given the parameter vectors aj and bi of camera j and point i, computes in Aij, Bij the
 * jacobian of the predicted projection of point i on image j
 */
static void img_projRTS_jac(int j, int i, double *aj, double *bi, double *Aij, double *Bij, void *adata)
{
  double *Kparms, *pr0;
  struct globs_ *gl;
  
  gl=(struct globs_ *)adata;
  Kparms=gl->intrcalib;
  pr0=gl->rot0params+j*FULLQUATSZ; // full quat for initial rotation estimate

  calcImgProjJacRTS(Kparms, pr0, aj, aj+3, bi, (double (*)[6])Aij, (double (*)[3])Bij); // 3 is the quaternion's vector part length
}

/* BUNDLE ADJUSTMENT FOR CAMERA PARAMETERS ONLY */

/* Given the parameter vector aj of camera j, computes in xij the
 * predicted projection of point i on image j
 */
static void img_projRT(int j, int i, double *aj, double *xij, void *adata)
{
  int pnp;

  double *Kparms, *pr0, *ptparams;
  struct globs_ *gl;

  gl=(struct globs_ *)adata;
  pnp=gl->pnp;
  Kparms=gl->intrcalib;
  pr0=gl->rot0params+j*FULLQUATSZ; // full quat for initial rotation estimate
  ptparams=gl->ptparams;

  calcImgProj(Kparms, pr0, aj, aj+3, ptparams+i*pnp, xij); // 3 is the quaternion's vector part length
}

/* Given the parameter vector aj of camera j, computes in Aij
 * the jacobian of the predicted projection of point i on image j
 */
static void img_projRT_jac(int j, int i, double *aj, double *Aij, void *adata)
{
  int pnp;

  double *Kparms, *ptparams, *pr0;
  struct globs_ *gl;
  
  gl=(struct globs_ *)adata;
  pnp=gl->pnp;
  Kparms=gl->intrcalib;
  pr0=gl->rot0params+j*FULLQUATSZ; // full quat for initial rotation estimate
  ptparams=gl->ptparams;

  calcImgProjJacRT(Kparms, pr0, aj, aj+3, ptparams+i*pnp, (double (*)[6])Aij); // 3 is the quaternion's vector part length
}

/* BUNDLE ADJUSTMENT FOR STRUCTURE PARAMETERS ONLY */

/* Given the parameter vector bi of point i, computes in xij the
 * predicted projection of point i on image j
 */
static void img_projS(int j, int i, double *bi, double *xij, void *adata)
{
  int cnp;

  double *Kparms, *camparams, *aj;
  struct globs_ *gl;

  gl=(struct globs_ *)adata;
  cnp=gl->cnp;
  Kparms=gl->intrcalib;
  camparams=gl->camparams;
  aj=camparams+j*cnp;

  calcImgProjFullR(Kparms, aj, aj+3, bi, xij); // 3 is the quaternion's vector part length
  //calcImgProj(Kparms, (double *)zerorotquat, aj, aj+3, bi, xij); // 3 is the quaternion's vector part length
}

/* Given the parameter vector bi of point i, computes in Bij
 * the jacobian of the predicted projection of point i on image j
 */
static void img_projS_jac(int j, int i, double *bi, double *Bij, void *adata)
{
  int cnp;

  double *Kparms, *camparams, *aj;
  struct globs_ *gl;
  
  gl=(struct globs_ *)adata;
  cnp=gl->cnp;
  Kparms=gl->intrcalib;
  camparams=gl->camparams;
  aj=camparams+j*cnp;

  calcImgProjJacS(Kparms, (double *)zerorotquat, aj, aj+3, bi, (double (*)[3])Bij); // 3 is the quaternion's vector part length
}

/*** MEASUREMENT VECTOR AND JACOBIAN COMPUTATION FOR THE EXPERT DRIVERS ***/

/* FULL BUNDLE ADJUSTMENT, I.E. SIMULTANEOUS ESTIMATION OF CAMERA AND STRUCTURE PARAMETERS */

/* Given a parameter vector p made up of the 3D coordinates of n points and the parameters of m cameras, compute in
 * hx the prediction of the measurements, i.e. the projections of 3D points in the m images. The measurements
 * are returned in the order (hx_11^T, .. hx_1m^T, ..., hx_n1^T, .. hx_nm^T)^T, where hx_ij is the predicted
 * projection of the i-th point on the j-th camera.
 * Notice that depending on idxij, some of the hx_ij might be missing
 *
 */
static void img_projsRTS_x(double *p, struct sba_crsm *idxij, int *rcidxs, int *rcsubs, double *hx, void *adata)
{
  register int i, j;
  int cnp, pnp, mnp;
  double *pa, *pb, *pqr, *pt, *ppt, *pmeas, *Kparms, *pr0, lrot[FULLQUATSZ], trot[FULLQUATSZ];
  //int n;
  int m, nnz;
  struct globs_ *gl;

  gl=(struct globs_ *)adata;
  cnp=gl->cnp; pnp=gl->pnp; mnp=gl->mnp;
  Kparms=gl->intrcalib;

  //n=idxij->nr;
  m=idxij->nc;
  pa=p; pb=p+m*cnp;

  for(j=0; j<m; ++j){
    /* j-th camera parameters */
    pqr=pa+j*cnp;
    pt=pqr+3; // quaternion vector part has 3 elements
    pr0=gl->rot0params+j*FULLQUATSZ; // full quat for initial rotation estimate
    _MK_QUAT_FRM_VEC(lrot, pqr);
    quatMultFast(lrot, pr0, trot); // trot=lrot*pr0

    nnz=sba_crsm_col_elmidxs(idxij, j, rcidxs, rcsubs); /* find nonzero hx_ij, i=0...n-1 */

    for(i=0; i<nnz; ++i){
      ppt=pb + rcsubs[i]*pnp;
      pmeas=hx + idxij->val[rcidxs[i]]*mnp; // set pmeas to point to hx_ij

      calcImgProjFullR(Kparms, trot, pt, ppt, pmeas); // evaluate Q in pmeas
      //calcImgProj(Kparms, pr0, pqr, pt, ppt, pmeas); // evaluate Q in pmeas
    }
  }
}

/* Given a parameter vector p made up of the 3D coordinates of n points and the parameters of m cameras, compute in
 * jac the jacobian of the predicted measurements, i.e. the jacobian of the projections of 3D points in the m images.
 * The jacobian is returned in the order (A_11, ..., A_1m, ..., A_n1, ..., A_nm, B_11, ..., B_1m, ..., B_n1, ..., B_nm),
 * where A_ij=dx_ij/db_j and B_ij=dx_ij/db_i (see HZ).
 * Notice that depending on idxij, some of the A_ij, B_ij might be missing
 *
 */
static void img_projsRTS_jac_x(double *p, struct sba_crsm *idxij, int *rcidxs, int *rcsubs, double *jac, void *adata)
{
  register int i, j;
  int cnp, pnp, mnp;
  double *pa, *pb, *pqr, *pt, *ppt, *pA, *pB, *Kparms, *pr0;
  //int n;
  int m, nnz, Asz, Bsz, ABsz;
  struct globs_ *gl;
  
  gl=(struct globs_ *)adata;
  cnp=gl->cnp; pnp=gl->pnp; mnp=gl->mnp;
  Kparms=gl->intrcalib;

  //n=idxij->nr;
  m=idxij->nc;
  pa=p; pb=p+m*cnp;
  Asz=mnp*cnp; Bsz=mnp*pnp; ABsz=Asz+Bsz;

  for(j=0; j<m; ++j){
    /* j-th camera parameters */
    pqr=pa+j*cnp;
    pt=pqr+3; // quaternion vector part has 3 elements
    pr0=gl->rot0params+j*FULLQUATSZ; // full quat for initial rotation estimate

    nnz=sba_crsm_col_elmidxs(idxij, j, rcidxs, rcsubs); /* find nonzero hx_ij, i=0...n-1 */

    for(i=0; i<nnz; ++i){
      ppt=pb + rcsubs[i]*pnp;
      pA=jac + idxij->val[rcidxs[i]]*ABsz; // set pA to point to A_ij
      pB=pA  + Asz; // set pB to point to B_ij

      calcImgProjJacRTS(Kparms, pr0, pqr, pt, ppt, (double (*)[6])pA, (double (*)[3])pB); // evaluate dQ/da, dQ/db in pA, pB
    }
  }
}

/* BUNDLE ADJUSTMENT FOR CAMERA PARAMETERS ONLY */

/* Given a parameter vector p made up of the parameters of m cameras, compute in
 * hx the prediction of the measurements, i.e. the projections of 3D points in the m images.
 * The measurements are returned in the order (hx_11^T, .. hx_1m^T, ..., hx_n1^T, .. hx_nm^T)^T,
 * where hx_ij is the predicted projection of the i-th point on the j-th camera.
 * Notice that depending on idxij, some of the hx_ij might be missing
 *
 */
static void img_projsRT_x(double *p, struct sba_crsm *idxij, int *rcidxs, int *rcsubs, double *hx, void *adata)
{
  register int i, j;
  int cnp, pnp, mnp;
  double *pqr, *pt, *ppt, *pmeas, *Kparms, *ptparams, *pr0, lrot[FULLQUATSZ], trot[FULLQUATSZ];
  //int n;
  int m, nnz;
  struct globs_ *gl;

  gl=(struct globs_ *)adata;
  cnp=gl->cnp; pnp=gl->pnp; mnp=gl->mnp;
  Kparms=gl->intrcalib;
  ptparams=gl->ptparams;

  //n=idxij->nr;
  m=idxij->nc;

  for(j=0; j<m; ++j){
    /* j-th camera parameters */
    pqr=p+j*cnp;
    pt=pqr+3; // quaternion vector part has 3 elements
    pr0=gl->rot0params+j*FULLQUATSZ; // full quat for initial rotation estimate
    _MK_QUAT_FRM_VEC(lrot, pqr);
    quatMultFast(lrot, pr0, trot); // trot=lrot*pr0

    nnz=sba_crsm_col_elmidxs(idxij, j, rcidxs, rcsubs); /* find nonzero hx_ij, i=0...n-1 */

    for(i=0; i<nnz; ++i){
	    ppt=ptparams + rcsubs[i]*pnp;
      pmeas=hx + idxij->val[rcidxs[i]]*mnp; // set pmeas to point to hx_ij

      calcImgProjFullR(Kparms, trot, pt, ppt, pmeas); // evaluate Q in pmeas
      //calcImgProj(Kparms, pr0, pqr, pt, ppt, pmeas); // evaluate Q in pmeas
    }
  }
}

/* Given a parameter vector p made up of the parameters of m cameras, compute in jac
 * the jacobian of the predicted measurements, i.e. the jacobian of the projections of 3D points in the m images.
 * The jacobian is returned in the order (A_11, ..., A_1m, ..., A_n1, ..., A_nm),
 * where A_ij=dx_ij/db_j (see HZ).
 * Notice that depending on idxij, some of the A_ij might be missing
 *
 */
static void img_projsRT_jac_x(double *p, struct sba_crsm *idxij, int *rcidxs, int *rcsubs, double *jac, void *adata)
{
  register int i, j;
  int cnp, pnp, mnp;
  double *pqr, *pt, *ppt, *pA, *Kparms, *ptparams, *pr0;
  //int n;
  int m, nnz, Asz;
  struct globs_ *gl;
  
  gl=(struct globs_ *)adata;
  cnp=gl->cnp; pnp=gl->pnp; mnp=gl->mnp;
  Kparms=gl->intrcalib;
  ptparams=gl->ptparams;

  //n=idxij->nr;
  m=idxij->nc;
  Asz=mnp*cnp;

  for(j=0; j<m; ++j){
    /* j-th camera parameters */
    pqr=p+j*cnp;
    pt=pqr+3; // quaternion vector part has 3 elements
    pr0=gl->rot0params+j*FULLQUATSZ; // full quat for initial rotation estimate

    nnz=sba_crsm_col_elmidxs(idxij, j, rcidxs, rcsubs); /* find nonzero hx_ij, i=0...n-1 */

    for(i=0; i<nnz; ++i){
      ppt=ptparams + rcsubs[i]*pnp;
      pA=jac + idxij->val[rcidxs[i]]*Asz; // set pA to point to A_ij

      calcImgProjJacRT(Kparms, pr0, pqr, pt, ppt, (double (*)[6])pA); // evaluate dQ/da in pA
    }
  }
}

/* BUNDLE ADJUSTMENT FOR STRUCTURE PARAMETERS ONLY */

/* Given a parameter vector p made up of the 3D coordinates of n points, compute in
 * hx the prediction of the measurements, i.e. the projections of 3D points in the m images. The measurements
 * are returned in the order (hx_11^T, .. hx_1m^T, ..., hx_n1^T, .. hx_nm^T)^T, where hx_ij is the predicted
 * projection of the i-th point on the j-th camera.
 * Notice that depending on idxij, some of the hx_ij might be missing
 *
 */
static void img_projsS_x(double *p, struct sba_crsm *idxij, int *rcidxs, int *rcsubs, double *hx, void *adata)
{
  register int i, j;
  int cnp, pnp, mnp;
  double *pqr, *pt, *ppt, *pmeas, *Kparms, *camparams, trot[FULLQUATSZ];
  //int n;
  int m, nnz;
  struct globs_ *gl;

  gl=(struct globs_ *)adata;
  cnp=gl->cnp; pnp=gl->pnp; mnp=gl->mnp;
  Kparms=gl->intrcalib;
  camparams=gl->camparams;

  //n=idxij->nr;
  m=idxij->nc;

  for(j=0; j<m; ++j){
    /* j-th camera parameters */
    pqr=camparams+j*cnp;
    pt=pqr+3; // quaternion vector part has 3 elements
    _MK_QUAT_FRM_VEC(trot, pqr);

    nnz=sba_crsm_col_elmidxs(idxij, j, rcidxs, rcsubs); /* find nonzero hx_ij, i=0...n-1 */

    for(i=0; i<nnz; ++i){
      ppt=p + rcsubs[i]*pnp;
      pmeas=hx + idxij->val[rcidxs[i]]*mnp; // set pmeas to point to hx_ij

      calcImgProjFullR(Kparms, trot, pt, ppt, pmeas); // evaluate Q in pmeas
      //calcImgProj(Kparms, (double *)zerorotquat, pqr, pt, ppt, pmeas); // evaluate Q in pmeas
    }
  }
}

/* Given a parameter vector p made up of the 3D coordinates of n points, compute in
 * jac the jacobian of the predicted measurements, i.e. the jacobian of the projections of 3D points in the m images.
 * The jacobian is returned in the order (B_11, ..., B_1m, ..., B_n1, ..., B_nm),
 * where B_ij=dx_ij/db_i (see HZ).
 * Notice that depending on idxij, some of the B_ij might be missing
 *
 */
static void img_projsS_jac_x(double *p, struct sba_crsm *idxij, int *rcidxs, int *rcsubs, double *jac, void *adata)
{
  register int i, j;
  int cnp, pnp, mnp;
  double *pqr, *pt, *ppt, *pB, *Kparms, *camparams;
  //int n;
  int m, nnz, Bsz;
  struct globs_ *gl;
  
  gl=(struct globs_ *)adata;
  cnp=gl->cnp; pnp=gl->pnp; mnp=gl->mnp;
  Kparms=gl->intrcalib;
  camparams=gl->camparams;

  //n=idxij->nr;
  m=idxij->nc;
  Bsz=mnp*pnp;

  for(j=0; j<m; ++j){
    /* j-th camera parameters */
    pqr=camparams+j*cnp;
    pt=pqr+3; // quaternion vector part has 3 elements

    nnz=sba_crsm_col_elmidxs(idxij, j, rcidxs, rcsubs); /* find nonzero hx_ij, i=0...n-1 */

    for(i=0; i<nnz; ++i){
      ppt=p + rcsubs[i]*pnp;
      pB=jac + idxij->val[rcidxs[i]]*Bsz; // set pB to point to B_ij

      calcImgProjJacS(Kparms, (double *)zerorotquat, pqr, pt, ppt, (double (*)[3])pB); // evaluate dQ/da in pB
    }
  }
}

/****************************************************************************************************/
/* MEASUREMENT VECTOR AND JACOBIAN COMPUTATION FOR VARYING CAMERA INTRINSICS, POSE AND 3D STRUCTURE */
/****************************************************************************************************/

/*** MEASUREMENT VECTOR AND JACOBIAN COMPUTATION FOR THE SIMPLE DRIVERS ***/

/* A note about the computation of Jacobians below:
 *
 * When performing BA that includes the camera intrinsics, it would be
 * very desirable to allow for certain parameters such as skew, aspect
 * ratio and principal point to be fixed. The straighforward way to
 * implement this would be to code a separate version of the Jacobian
 * computation routines for each subset of non-fixed parameters. Here,
 * this is bypassed by developing only one set of Jacobian computation
 * routines which estimate the former for all 5 intrinsics and then set
 * the columns corresponding to fixed parameters to zero.
 */

/* FULL BUNDLE ADJUSTMENT, I.E. SIMULTANEOUS ESTIMATION OF CAMERA AND STRUCTURE PARAMETERS */

/* Given the parameter vectors aj and bi of camera j and point i, computes in xij the
 * predicted projection of point i on image j
 */
static void img_projKRTS(int j, int i, double *aj, double *bi, double *xij, void *adata)
{
  double *pr0;
  struct globs_ *gl;

  gl=(struct globs_ *)adata;
  pr0=gl->rot0params+j*FULLQUATSZ; // full quat for initial rotation estimate

  calcImgProj(aj, pr0, aj+5, aj+5+3, bi, xij); // 5 for the calibration + 3 for the quaternion's vector part
}

/* Given the parameter vectors aj and bi of camera j and point i, computes in Aij, Bij the
 * jacobian of the predicted projection of point i on image j
 */
static void img_projKRTS_jac(int j, int i, double *aj, double *bi, double *Aij, double *Bij, void *adata)
{
struct globs_ *gl;
double *pr0;
int ncK;

  gl=(struct globs_ *)adata;
  pr0=gl->rot0params+j*FULLQUATSZ; // full quat for initial rotation estimate
  calcImgProjJacKRTS(aj, pr0, aj+5, aj+5+3, bi, (double (*)[5+6])Aij, (double (*)[3])Bij); // 5 for the calibration + 3 for the quaternion's vector part

  /* clear the columns of the Jacobian corresponding to fixed calibration parameters */
  gl=(struct globs_ *)adata;
  ncK=gl->nccalib;
  if(ncK){
    int cnp, mnp, j0;

    cnp=gl->cnp;
    mnp=gl->mnp;
    j0=5-ncK;

    for(i=0; i<mnp; ++i, Aij+=cnp)
      for(j=j0; j<5; ++j)
        Aij[j]=0.0; // Aij[i*cnp+j]=0.0;
  }
}

/* BUNDLE ADJUSTMENT FOR CAMERA PARAMETERS ONLY */

/* Given the parameter vector aj of camera j, computes in xij the
 * predicted projection of point i on image j
 */
static void img_projKRT(int j, int i, double *aj, double *xij, void *adata)
{
  int pnp;

  double *ptparams, *pr0;
  struct globs_ *gl;

  gl=(struct globs_ *)adata;
  pnp=gl->pnp;
  ptparams=gl->ptparams;
  pr0=gl->rot0params+j*FULLQUATSZ; // full quat for initial rotation estimate

  calcImgProj(aj, pr0, aj+5, aj+5+3, ptparams+i*pnp, xij); // 5 for the calibration + 3 for the quaternion's vector part
}

/* Given the parameter vector aj of camera j, computes in Aij
 * the jacobian of the predicted projection of point i on image j
 */
static void img_projKRT_jac(int j, int i, double *aj, double *Aij, void *adata)
{
struct globs_ *gl;
double *ptparams, *pr0;
int pnp, ncK;
  
  gl=(struct globs_ *)adata;
  pnp=gl->pnp;
  ptparams=gl->ptparams;
  pr0=gl->rot0params+j*FULLQUATSZ; // full quat for initial rotation estimate

  calcImgProjJacKRT(aj, pr0, aj+5, aj+5+3, ptparams+i*pnp, (double (*)[5+6])Aij); // 5 for the calibration + 3 for the quaternion's vector part

  /* clear the columns of the Jacobian corresponding to fixed calibration parameters */
  ncK=gl->nccalib;
  if(ncK){
    int cnp, mnp, j0;

    cnp=gl->cnp;
    mnp=gl->mnp;
    j0=5-ncK;

    for(i=0; i<mnp; ++i, Aij+=cnp)
      for(j=j0; j<5; ++j)
        Aij[j]=0.0; // Aij[i*cnp+j]=0.0;
  }
}

/* BUNDLE ADJUSTMENT FOR STRUCTURE PARAMETERS ONLY */

/* Given the parameter vector bi of point i, computes in xij the
 * predicted projection of point i on image j
 */
static void img_projKS(int j, int i, double *bi, double *xij, void *adata)
{
  int cnp;

  double *camparams, *aj;
  struct globs_ *gl;

  gl=(struct globs_ *)adata;
  cnp=gl->cnp;
  camparams=gl->camparams;
  aj=camparams+j*cnp;

  calcImgProjFullR(aj, aj+5, aj+5+3, bi, xij); // 5 for the calibration + 3 for the quaternion's vector part
  //calcImgProj(aj, (double *)zerorotquat, aj+5, aj+5+3, bi, xij); // 5 for the calibration + 3 for the quaternion's vector part
}

/* Given the parameter vector bi of point i, computes in Bij
 * the jacobian of the predicted projection of point i on image j
 */
static void img_projKS_jac(int j, int i, double *bi, double *Bij, void *adata)
{
  int cnp;

  double *camparams, *aj;
  struct globs_ *gl;
  
  gl=(struct globs_ *)adata;
  cnp=gl->cnp;
  camparams=gl->camparams;
  aj=camparams+j*cnp;

  calcImgProjJacS(aj, (double *)zerorotquat, aj+5, aj+5+3, bi, (double (*)[3])Bij); // 5 for the calibration + 3 for the quaternion's vector part
}

/*** MEASUREMENT VECTOR AND JACOBIAN COMPUTATION FOR THE EXPERT DRIVERS ***/

/* FULL BUNDLE ADJUSTMENT, I.E. SIMULTANEOUS ESTIMATION OF CAMERA AND STRUCTURE PARAMETERS */

/* Given a parameter vector p made up of the 3D coordinates of n points and the parameters of m cameras, compute in
 * hx the prediction of the measurements, i.e. the projections of 3D points in the m images. The measurements
 * are returned in the order (hx_11^T, .. hx_1m^T, ..., hx_n1^T, .. hx_nm^T)^T, where hx_ij is the predicted
 * projection of the i-th point on the j-th camera.
 * Notice that depending on idxij, some of the hx_ij might be missing
 *
 */
static void img_projsKRTS_x(double *p, struct sba_crsm *idxij, int *rcidxs, int *rcsubs, double *hx, void *adata)
{
  register int i, j;
  int cnp, pnp, mnp;
  double *pa, *pb, *pqr, *pt, *ppt, *pmeas, *pcalib, *pr0, lrot[FULLQUATSZ], trot[FULLQUATSZ];
  //int n;
  int m, nnz;
  struct globs_ *gl;

  gl=(struct globs_ *)adata;
  cnp=gl->cnp; pnp=gl->pnp; mnp=gl->mnp;

  //n=idxij->nr;
  m=idxij->nc;
  pa=p; pb=p+m*cnp;

  for(j=0; j<m; ++j){
    /* j-th camera parameters */
    pcalib=pa+j*cnp;
    pqr=pcalib+5;
    pt=pqr+3; // quaternion vector part has 3 elements
    pr0=gl->rot0params+j*FULLQUATSZ; // full quat for initial rotation estimate
    _MK_QUAT_FRM_VEC(lrot, pqr);
    quatMultFast(lrot, pr0, trot); // trot=lrot*pr0

    nnz=sba_crsm_col_elmidxs(idxij, j, rcidxs, rcsubs); /* find nonzero hx_ij, i=0...n-1 */

    for(i=0; i<nnz; ++i){
      ppt=pb + rcsubs[i]*pnp;
      pmeas=hx + idxij->val[rcidxs[i]]*mnp; // set pmeas to point to hx_ij

      calcImgProjFullR(pcalib, trot, pt, ppt, pmeas); // evaluate Q in pmeas
      //calcImgProj(pcalib, pr0, pqr, pt, ppt, pmeas); // evaluate Q in pmeas
    }
  }
}

/* Given a parameter vector p made up of the 3D coordinates of n points and the parameters of m cameras, compute in
 * jac the jacobian of the predicted measurements, i.e. the jacobian of the projections of 3D points in the m images.
 * The jacobian is returned in the order (A_11, ..., A_1m, ..., A_n1, ..., A_nm, B_11, ..., B_1m, ..., B_n1, ..., B_nm),
 * where A_ij=dx_ij/db_j and B_ij=dx_ij/db_i (see HZ).
 * Notice that depending on idxij, some of the A_ij, B_ij might be missing
 *
 */
static void img_projsKRTS_jac_x(double *p, struct sba_crsm *idxij, int *rcidxs, int *rcsubs, double *jac, void *adata)
{
  register int i, j, ii, jj;
  int cnp, pnp, mnp, ncK;
  double *pa, *pb, *pqr, *pt, *ppt, *pA, *pB, *pcalib, *pr0;
  //int n;
  int m, nnz, Asz, Bsz, ABsz;
  struct globs_ *gl;
  
  gl=(struct globs_ *)adata;
  cnp=gl->cnp; pnp=gl->pnp; mnp=gl->mnp;
  ncK=gl->nccalib;

  //n=idxij->nr;
  m=idxij->nc;
  pa=p; pb=p+m*cnp;
  Asz=mnp*cnp; Bsz=mnp*pnp; ABsz=Asz+Bsz;

  for(j=0; j<m; ++j){
    /* j-th camera parameters */
    pcalib=pa+j*cnp;
    pqr=pcalib+5;
    pt=pqr+3; // quaternion vector part has 3 elements
    pr0=gl->rot0params+j*FULLQUATSZ; // full quat for initial rotation estimate

    nnz=sba_crsm_col_elmidxs(idxij, j, rcidxs, rcsubs); /* find nonzero hx_ij, i=0...n-1 */

    for(i=0; i<nnz; ++i){
      ppt=pb + rcsubs[i]*pnp;
      pA=jac + idxij->val[rcidxs[i]]*ABsz; // set pA to point to A_ij
      pB=pA  + Asz; // set pB to point to B_ij

      calcImgProjJacKRTS(pcalib, pr0, pqr, pt, ppt, (double (*)[5+6])pA, (double (*)[3])pB); // evaluate dQ/da, dQ/db in pA, pB

      /* clear the columns of the Jacobian corresponding to fixed calibration parameters */
      if(ncK){
        int jj0=5-ncK;

        for(ii=0; ii<mnp; ++ii, pA+=cnp)
          for(jj=jj0; jj<5; ++jj)
            pA[jj]=0.0; // pA[ii*cnp+jj]=0.0;
      }
    }
  }
}

/* BUNDLE ADJUSTMENT FOR CAMERA PARAMETERS ONLY */

/* Given a parameter vector p made up of the parameters of m cameras, compute in
 * hx the prediction of the measurements, i.e. the projections of 3D points in the m images.
 * The measurements are returned in the order (hx_11^T, .. hx_1m^T, ..., hx_n1^T, .. hx_nm^T)^T,
 * where hx_ij is the predicted projection of the i-th point on the j-th camera.
 * Notice that depending on idxij, some of the hx_ij might be missing
 *
 */
static void img_projsKRT_x(double *p, struct sba_crsm *idxij, int *rcidxs, int *rcsubs, double *hx, void *adata)
{
  register int i, j;
  int cnp, pnp, mnp;
  double *pqr, *pt, *ppt, *pmeas, *pcalib, *ptparams, *pr0, lrot[FULLQUATSZ], trot[FULLQUATSZ];
  //int n;
  int m, nnz;
  struct globs_ *gl;

  gl=(struct globs_ *)adata;
  cnp=gl->cnp; pnp=gl->pnp; mnp=gl->mnp;
  ptparams=gl->ptparams;

  //n=idxij->nr;
  m=idxij->nc;

  for(j=0; j<m; ++j){
    /* j-th camera parameters */
    pcalib=p+j*cnp;
    pqr=pcalib+5;
    pt=pqr+3; // quaternion vector part has 3 elements
    pr0=gl->rot0params+j*FULLQUATSZ; // full quat for initial rotation estimate
    _MK_QUAT_FRM_VEC(lrot, pqr);
    quatMultFast(lrot, pr0, trot); // trot=lrot*pr0

    nnz=sba_crsm_col_elmidxs(idxij, j, rcidxs, rcsubs); /* find nonzero hx_ij, i=0...n-1 */

    for(i=0; i<nnz; ++i){
	    ppt=ptparams + rcsubs[i]*pnp;
      pmeas=hx + idxij->val[rcidxs[i]]*mnp; // set pmeas to point to hx_ij

      calcImgProjFullR(pcalib, trot, pt, ppt, pmeas); // evaluate Q in pmeas
      //calcImgProj(pcalib, pr0, pqr, pt, ppt, pmeas); // evaluate Q in pmeas
    }
  }
}

/* Given a parameter vector p made up of the parameters of m cameras, compute in jac
 * the jacobian of the predicted measurements, i.e. the jacobian of the projections of 3D points in the m images.
 * The jacobian is returned in the order (A_11, ..., A_1m, ..., A_n1, ..., A_nm),
 * where A_ij=dx_ij/db_j (see HZ).
 * Notice that depending on idxij, some of the A_ij might be missing
 *
 */
static void img_projsKRT_jac_x(double *p, struct sba_crsm *idxij, int *rcidxs, int *rcsubs, double *jac, void *adata)
{
  register int i, j, ii, jj;
  int cnp, pnp, mnp, ncK;
  double *pqr, *pt, *ppt, *pA, *pcalib, *ptparams, *pr0;
  //int n;
  int m, nnz, Asz;
  struct globs_ *gl;
  
  gl=(struct globs_ *)adata;
  cnp=gl->cnp; pnp=gl->pnp; mnp=gl->mnp;
  ncK=gl->nccalib;
  ptparams=gl->ptparams;

  //n=idxij->nr;
  m=idxij->nc;
  Asz=mnp*cnp;

  for(j=0; j<m; ++j){
    /* j-th camera parameters */
    pcalib=p+j*cnp;
    pqr=pcalib+5;
    pt=pqr+3; // quaternion vector part has 3 elements
    pr0=gl->rot0params+j*FULLQUATSZ; // full quat for initial rotation estimate

    nnz=sba_crsm_col_elmidxs(idxij, j, rcidxs, rcsubs); /* find nonzero hx_ij, i=0...n-1 */

    for(i=0; i<nnz; ++i){
      ppt=ptparams + rcsubs[i]*pnp;
      pA=jac + idxij->val[rcidxs[i]]*Asz; // set pA to point to A_ij

      calcImgProjJacKRT(pcalib, pr0, pqr, pt, ppt, (double (*)[5+6])pA); // evaluate dQ/da in pA

      /* clear the columns of the Jacobian corresponding to fixed calibration parameters */
      if(ncK){
        int jj0;

        jj0=5-ncK;
        for(ii=0; ii<mnp; ++ii, pA+=cnp)
          for(jj=jj0; jj<5; ++jj)
            pA[jj]=0.0; // pA[ii*cnp+jj]=0.0;
      }
    }
  }
}

/* BUNDLE ADJUSTMENT FOR STRUCTURE PARAMETERS ONLY */

/* Given a parameter vector p made up of the 3D coordinates of n points, compute in
 * hx the prediction of the measurements, i.e. the projections of 3D points in the m images. The measurements
 * are returned in the order (hx_11^T, .. hx_1m^T, ..., hx_n1^T, .. hx_nm^T)^T, where hx_ij is the predicted
 * projection of the i-th point on the j-th camera.
 * Notice that depending on idxij, some of the hx_ij might be missing
 *
 */
static void img_projsKS_x(double *p, struct sba_crsm *idxij, int *rcidxs, int *rcsubs, double *hx, void *adata)
{
  register int i, j;
  int cnp, pnp, mnp;
  double *pqr, *pt, *ppt, *pmeas, *pcalib, *camparams, trot[FULLQUATSZ];
  //int n;
  int m, nnz;
  struct globs_ *gl;

  gl=(struct globs_ *)adata;
  cnp=gl->cnp; pnp=gl->pnp; mnp=gl->mnp;
  camparams=gl->camparams;

  //n=idxij->nr;
  m=idxij->nc;

  for(j=0; j<m; ++j){
    /* j-th camera parameters */
    pcalib=camparams+j*cnp;
    pqr=pcalib+5;
    pt=pqr+3; // quaternion vector part has 3 elements
    _MK_QUAT_FRM_VEC(trot, pqr);

    nnz=sba_crsm_col_elmidxs(idxij, j, rcidxs, rcsubs); /* find nonzero hx_ij, i=0...n-1 */

    for(i=0; i<nnz; ++i){
      ppt=p + rcsubs[i]*pnp;
      pmeas=hx + idxij->val[rcidxs[i]]*mnp; // set pmeas to point to hx_ij

      calcImgProjFullR(pcalib, trot, pt, ppt, pmeas); // evaluate Q in pmeas
      //calcImgProj(pcalib, (double *)zerorotquat, pqr, pt, ppt, pmeas); // evaluate Q in pmeas
    }
  }
}

/* Given a parameter vector p made up of the 3D coordinates of n points, compute in
 * jac the jacobian of the predicted measurements, i.e. the jacobian of the projections of 3D points in the m images.
 * The jacobian is returned in the order (B_11, ..., B_1m, ..., B_n1, ..., B_nm),
 * where B_ij=dx_ij/db_i (see HZ).
 * Notice that depending on idxij, some of the B_ij might be missing
 *
 */
static void img_projsKS_jac_x(double *p, struct sba_crsm *idxij, int *rcidxs, int *rcsubs, double *jac, void *adata)
{
  register int i, j;
  int cnp, pnp, mnp;
  double *pqr, *pt, *ppt, *pB, *pcalib, *camparams;
  //int n;
  int m, nnz, Bsz;
  struct globs_ *gl;
  
  gl=(struct globs_ *)adata;
  cnp=gl->cnp; pnp=gl->pnp; mnp=gl->mnp;
  camparams=gl->camparams;

  //n=idxij->nr;
  m=idxij->nc;
  Bsz=mnp*pnp;

  for(j=0; j<m; ++j){
    /* j-th camera parameters */
    pcalib=camparams+j*cnp;
    pqr=pcalib+5;
    pt=pqr+3; // quaternion vector part has 3 elements

    nnz=sba_crsm_col_elmidxs(idxij, j, rcidxs, rcsubs); /* find nonzero hx_ij, i=0...n-1 */

    for(i=0; i<nnz; ++i){
      ppt=p + rcsubs[i]*pnp;
      pB=jac + idxij->val[rcidxs[i]]*Bsz; // set pB to point to B_ij

      calcImgProjJacS(pcalib, (double *)zerorotquat, pqr, pt, ppt, (double (*)[3])pB); // evaluate dQ/da in pB
    }
  }
}

/****************************************************************************************************************/
/* MEASUREMENT VECTOR AND JACOBIAN COMPUTATION FOR VARYING CAMERA INTRINSICS, DISTORTION, POSE AND 3D STRUCTURE */
/****************************************************************************************************************/

/*** MEASUREMENT VECTOR AND JACOBIAN COMPUTATION FOR THE SIMPLE DRIVERS ***/

/* A note about the computation of Jacobians below:
 *
 * When performing BA that includes the camera intrinsics & distortion, it would be
 * very desirable to allow for certain parameters such as skew, aspect ratio and principal
 * point (also high order radial distortion, tangential distortion), to be fixed. The
 * straighforward way to implement this would be to code a separate version of the
 * Jacobian computation routines for each subset of non-fixed parameters. Here,
 * this is bypassed by developing only one set of Jacobian computation
 * routines which estimate the former for all 5 intrinsics and all 5 distortion
 * coefficients and then set the columns corresponding to fixed parameters to zero.
 */

/* FULL BUNDLE ADJUSTMENT, I.E. SIMULTANEOUS ESTIMATION OF CAMERA AND STRUCTURE PARAMETERS */

/* Given the parameter vectors aj and bi of camera j and point i, computes in xij the
 * predicted projection of point i on image j
 */
static void img_projKDRTS(int j, int i, double *aj, double *bi, double *xij, void *adata)
{
  double *pr0;
  struct globs_ *gl;

  gl=(struct globs_ *)adata;
  pr0=gl->rot0params+j*FULLQUATSZ; // full quat for initial rotation estimate

  calcDistImgProj(aj, aj+5, pr0, aj+5+5, aj+5+5+3, bi, xij); // 5 for the calibration + 5 for the distortion + 3 for the quaternion's vector part
}

/* Given the parameter vectors aj and bi of camera j and point i, computes in Aij, Bij the
 * jacobian of the predicted projection of point i on image j
 */
static void img_projKDRTS_jac(int j, int i, double *aj, double *bi, double *Aij, double *Bij, void *adata)
{
struct globs_ *gl;
double *pA, *pr0;
int nc;

  gl=(struct globs_ *)adata;
  pr0=gl->rot0params+j*FULLQUATSZ; // full quat for initial rotation estimate
  calcDistImgProjJacKDRTS(aj, aj+5, pr0, aj+5+5, aj+5+5+3, bi, (double (*)[5+5+6])Aij, (double (*)[3])Bij); // 5 for the calibration + 5 for the distortion + 3 for the quaternion's vector part

  /* clear the columns of the Jacobian corresponding to fixed calibration parameters */
  gl=(struct globs_ *)adata;
  nc=gl->nccalib;
  if(nc){
    int cnp, mnp, j0;

    pA=Aij;
    cnp=gl->cnp;
    mnp=gl->mnp;
    j0=5-nc;

    for(i=0; i<mnp; ++i, pA+=cnp)
      for(j=j0; j<5; ++j)
        pA[j]=0.0; // pA[i*cnp+j]=0.0;
  }

  /* clear the columns of the Jacobian corresponding to fixed distortion parameters */
  nc=gl->ncdist;
  if(nc){
    int cnp, mnp, j0;

    pA=Aij;
    cnp=gl->cnp;
    mnp=gl->mnp;
    j0=5-nc;

    for(i=0; i<mnp; ++i, pA+=cnp)
      for(j=j0; j<5; ++j)
        pA[5+j]=0.0; // pA[i*cnp+5+j]=0.0;
  }
}

/* BUNDLE ADJUSTMENT FOR CAMERA PARAMETERS ONLY */

/* Given the parameter vector aj of camera j, computes in xij the
 * predicted projection of point i on image j
 */
static void img_projKDRT(int j, int i, double *aj, double *xij, void *adata)
{
  int pnp;

  double *ptparams, *pr0;
  struct globs_ *gl;

  gl=(struct globs_ *)adata;
  pnp=gl->pnp;
  ptparams=gl->ptparams;
  pr0=gl->rot0params+j*FULLQUATSZ; // full quat for initial rotation estimate

  calcDistImgProj(aj, aj+5, pr0, aj+5+5, aj+5+5+3, ptparams+i*pnp, xij); // 5 for the calibration + 5 for the distortion + 3 for the quaternion's vector part
}

/* Given the parameter vector aj of camera j, computes in Aij
 * the jacobian of the predicted projection of point i on image j
 */
static void img_projKDRT_jac(int j, int i, double *aj, double *Aij, void *adata)
{
struct globs_ *gl;
double *pA, *ptparams, *pr0;
int pnp, nc;
  
  gl=(struct globs_ *)adata;
  pnp=gl->pnp;
  ptparams=gl->ptparams;
  pr0=gl->rot0params+j*FULLQUATSZ; // full quat for initial rotation estimate

  calcDistImgProjJacKDRT(aj, aj+5, pr0, aj+5+5, aj+5+5+3, ptparams+i*pnp, (double (*)[5+5+6])Aij); // 5 for the calibration + 5 for the distortion + 3 for the quaternion's vector part

  /* clear the columns of the Jacobian corresponding to fixed calibration parameters */
  nc=gl->nccalib;
  if(nc){
    int cnp, mnp, j0;

    pA=Aij;
    cnp=gl->cnp;
    mnp=gl->mnp;
    j0=5-nc;

    for(i=0; i<mnp; ++i, pA+=cnp)
      for(j=j0; j<5; ++j)
        pA[j]=0.0; // pA[i*cnp+j]=0.0;
  }
  nc=gl->ncdist;
  if(nc){
    int cnp, mnp, j0;

    pA=Aij;
    cnp=gl->cnp;
    mnp=gl->mnp;
    j0=5-nc;

    for(i=0; i<mnp; ++i, pA+=cnp)
      for(j=j0; j<5; ++j)
        pA[5+j]=0.0; // pA[i*cnp+5+j]=0.0;
  }
}

/* BUNDLE ADJUSTMENT FOR STRUCTURE PARAMETERS ONLY */

/* Given the parameter vector bi of point i, computes in xij the
 * predicted projection of point i on image j
 */
static void img_projKDS(int j, int i, double *bi, double *xij, void *adata)
{
  int cnp;

  double *camparams, *aj;
  struct globs_ *gl;

  gl=(struct globs_ *)adata;
  cnp=gl->cnp;
  camparams=gl->camparams;
  aj=camparams+j*cnp;

  calcDistImgProjFullR(aj, aj+5, aj+5+5, aj+5+5+3, bi, xij); // 5 for the calibration + 5 for the distortion + 3 for the quaternion's vector part
  //calcDistImgProj(aj, aj+5, (double *)zerorotquat, aj+5+5, aj+5+5+3, bi, xij); // 5 for the calibration + 5 for the distortion + 3 for the quaternion's vector part
}

/* Given the parameter vector bi of point i, computes in Bij the
 * jacobian of the predicted projection of point i on image j
 */
static void img_projKDS_jac(int j, int i, double *bi, double *Bij, void *adata)
{
  int cnp;

  double *camparams, *aj;
  struct globs_ *gl;

  gl=(struct globs_ *)adata;
  cnp=gl->cnp;
  camparams=gl->camparams;
  aj=camparams+j*cnp;

  calcDistImgProjJacS(aj, aj+5, (double *)zerorotquat, aj+5+5, aj+5+5+3, bi, (double (*)[3])Bij); // 5 for the calibration + 5 for the distortion + 3 for the quaternion's vector part
}


/*** MEASUREMENT VECTOR AND JACOBIAN COMPUTATION FOR THE EXPERT DRIVERS ***/

/* FULL BUNDLE ADJUSTMENT, I.E. SIMULTANEOUS ESTIMATION OF CAMERA AND STRUCTURE PARAMETERS */

/* Given a parameter vector p made up of the 3D coordinates of n points and the parameters of m cameras, compute in
 * hx the prediction of the measurements, i.e. the projections of 3D points in the m images. The measurements
 * are returned in the order (hx_11^T, .. hx_1m^T, ..., hx_n1^T, .. hx_nm^T)^T, where hx_ij is the predicted
 * projection of the i-th point on the j-th camera.
 * Notice that depending on idxij, some of the hx_ij might be missing
 *
 */
static void img_projsKDRTS_x(double *p, struct sba_crsm *idxij, int *rcidxs, int *rcsubs, double *hx, void *adata)
{
  register int i, j;
  int cnp, pnp, mnp;
  double *pa, *pb, *pqr, *pt, *ppt, *pmeas, *pcalib, *pdist, *pr0, lrot[FULLQUATSZ], trot[FULLQUATSZ];
  //int n;
  int m, nnz;
  struct globs_ *gl;

  gl=(struct globs_ *)adata;
  cnp=gl->cnp; pnp=gl->pnp; mnp=gl->mnp;

  //n=idxij->nr;
  m=idxij->nc;
  pa=p; pb=p+m*cnp;

  for(j=0; j<m; ++j){
    /* j-th camera parameters */
    pcalib=pa+j*cnp;
    pdist=pcalib+5;
    pqr=pdist+5;
    pt=pqr+3; // quaternion vector part has 3 elements
    pr0=gl->rot0params+j*FULLQUATSZ; // full quat for initial rotation estimate
    _MK_QUAT_FRM_VEC(lrot, pqr);
    quatMultFast(lrot, pr0, trot); // trot=lrot*pr0

    nnz=sba_crsm_col_elmidxs(idxij, j, rcidxs, rcsubs); /* find nonzero hx_ij, i=0...n-1 */

    for(i=0; i<nnz; ++i){
      ppt=pb + rcsubs[i]*pnp;
      pmeas=hx + idxij->val[rcidxs[i]]*mnp; // set pmeas to point to hx_ij

      calcDistImgProjFullR(pcalib, pdist, trot, pt, ppt, pmeas); // evaluate Q in pmeas
      //calcDistImgProj(pcalib, pdist, pr0, pqr, pt, ppt, pmeas); // evaluate Q in pmeas
    }
  }
}

/* Given a parameter vector p made up of the 3D coordinates of n points and the parameters of m cameras, compute in
 * jac the jacobian of the predicted measurements, i.e. the jacobian of the projections of 3D points in the m images.
 * The jacobian is returned in the order (A_11, ..., A_1m, ..., A_n1, ..., A_nm, B_11, ..., B_1m, ..., B_n1, ..., B_nm),
 * where A_ij=dx_ij/db_j and B_ij=dx_ij/db_i (see HZ).
 * Notice that depending on idxij, some of the A_ij, B_ij might be missing
 *
 */
static void img_projsKDRTS_jac_x(double *p, struct sba_crsm *idxij, int *rcidxs, int *rcsubs, double *jac, void *adata)
{
  register int i, j, ii, jj;
  int cnp, pnp, mnp, ncK, ncD;
  double *pa, *pb, *pqr, *pt, *ppt, *pA, *pB, *ptr, *pcalib, *pdist, *pr0;
  //int n;
  int m, nnz, Asz, Bsz, ABsz;
  struct globs_ *gl;
  
  gl=(struct globs_ *)adata;
  cnp=gl->cnp; pnp=gl->pnp; mnp=gl->mnp;
  ncK=gl->nccalib;
  ncD=gl->ncdist;

  //n=idxij->nr;
  m=idxij->nc;
  pa=p; pb=p+m*cnp;
  Asz=mnp*cnp; Bsz=mnp*pnp; ABsz=Asz+Bsz;

  for(j=0; j<m; ++j){
    /* j-th camera parameters */
    pcalib=pa+j*cnp;
    pdist=pcalib+5;
    pqr=pdist+5;
    pt=pqr+3; // quaternion vector part has 3 elements
    pr0=gl->rot0params+j*FULLQUATSZ; // full quat for initial rotation estimate

    nnz=sba_crsm_col_elmidxs(idxij, j, rcidxs, rcsubs); /* find nonzero hx_ij, i=0...n-1 */

    for(i=0; i<nnz; ++i){
      ppt=pb + rcsubs[i]*pnp;
      pA=jac + idxij->val[rcidxs[i]]*ABsz; // set pA to point to A_ij
      pB=pA  + Asz; // set pB to point to B_ij

      calcDistImgProjJacKDRTS(pcalib, pdist, pr0, pqr, pt, ppt, (double (*)[5+5+6])pA, (double (*)[3])pB); // evaluate dQ/da, dQ/db in pA, pB

      /* clear the columns of the Jacobian corresponding to fixed calibration parameters */
      if(ncK){
        int jj0=5-ncK;

        ptr=pA;
        for(ii=0; ii<mnp; ++ii, ptr+=cnp)
          for(jj=jj0; jj<5; ++jj)
            ptr[jj]=0.0; // ptr[ii*cnp+jj]=0.0;
      }

      /* clear the columns of the Jacobian corresponding to fixed distortion parameters */
      if(ncD){
        int jj0=5-ncD;

        ptr=pA;
        for(ii=0; ii<mnp; ++ii, ptr+=cnp)
          for(jj=jj0; jj<5; ++jj)
            ptr[5+jj]=0.0; // ptr[ii*cnp+5+jj]=0.0;
      }
    }
  }
}

/* BUNDLE ADJUSTMENT FOR CAMERA PARAMETERS ONLY */

/* Given a parameter vector p made up of the parameters of m cameras, compute in
 * hx the prediction of the measurements, i.e. the projections of 3D points in the m images.
 * The measurements are returned in the order (hx_11^T, .. hx_1m^T, ..., hx_n1^T, .. hx_nm^T)^T,
 * where hx_ij is the predicted projection of the i-th point on the j-th camera.
 * Notice that depending on idxij, some of the hx_ij might be missing
 *
 */
static void img_projsKDRT_x(double *p, struct sba_crsm *idxij, int *rcidxs, int *rcsubs, double *hx, void *adata)
{
  register int i, j;
  int cnp, pnp, mnp;
  double *pqr, *pt, *ppt, *pmeas, *pcalib, *pdist, *ptparams, *pr0, lrot[FULLQUATSZ], trot[FULLQUATSZ];
  //int n;
  int m, nnz;
  struct globs_ *gl;

  gl=(struct globs_ *)adata;
  cnp=gl->cnp; pnp=gl->pnp; mnp=gl->mnp;
  ptparams=gl->ptparams;

  //n=idxij->nr;
  m=idxij->nc;

  for(j=0; j<m; ++j){
    /* j-th camera parameters */
    pcalib=p+j*cnp;
    pdist=pcalib+5;
    pqr=pdist+5;
    pt=pqr+3; // quaternion vector part has 3 elements
    pr0=gl->rot0params+j*FULLQUATSZ; // full quat for initial rotation estimate
    _MK_QUAT_FRM_VEC(lrot, pqr);
    quatMultFast(lrot, pr0, trot); // trot=lrot*pr0

    nnz=sba_crsm_col_elmidxs(idxij, j, rcidxs, rcsubs); /* find nonzero hx_ij, i=0...n-1 */

    for(i=0; i<nnz; ++i){
	    ppt=ptparams + rcsubs[i]*pnp;
      pmeas=hx + idxij->val[rcidxs[i]]*mnp; // set pmeas to point to hx_ij

      calcDistImgProjFullR(pcalib, pdist, trot, pt, ppt, pmeas); // evaluate Q in pmeas
      //calcDistImgProj(pcalib, pdist, pr0, pqr, pt, ppt, pmeas); // evaluate Q in pmeas
    }
  }
}

/* Given a parameter vector p made up of the parameters of m cameras, compute in jac
 * the jacobian of the predicted measurements, i.e. the jacobian of the projections of 3D points in the m images.
 * The jacobian is returned in the order (A_11, ..., A_1m, ..., A_n1, ..., A_nm),
 * where A_ij=dx_ij/db_j (see HZ).
 * Notice that depending on idxij, some of the A_ij might be missing
 *
 */
static void img_projsKDRT_jac_x(double *p, struct sba_crsm *idxij, int *rcidxs, int *rcsubs, double *jac, void *adata)
{
  register int i, j, ii, jj;
  int cnp, pnp, mnp, ncK, ncD;
  double *pqr, *pt, *ppt, *pA, *ptr, *pcalib, *pdist, *ptparams, *pr0;
  //int n;
  int m, nnz, Asz;
  struct globs_ *gl;
  
  gl=(struct globs_ *)adata;
  cnp=gl->cnp; pnp=gl->pnp; mnp=gl->mnp;
  ncK=gl->nccalib;
  ncD=gl->ncdist;
  ptparams=gl->ptparams;

  //n=idxij->nr;
  m=idxij->nc;
  Asz=mnp*cnp;

  for(j=0; j<m; ++j){
    /* j-th camera parameters */
    pcalib=p+j*cnp;
    pdist=pcalib+5;
    pqr=pdist+5;
    pt=pqr+3; // quaternion vector part has 3 elements
    pr0=gl->rot0params+j*FULLQUATSZ; // full quat for initial rotation estimate

    nnz=sba_crsm_col_elmidxs(idxij, j, rcidxs, rcsubs); /* find nonzero hx_ij, i=0...n-1 */

    for(i=0; i<nnz; ++i){
      ppt=ptparams + rcsubs[i]*pnp;
      pA=jac + idxij->val[rcidxs[i]]*Asz; // set pA to point to A_ij

      calcDistImgProjJacKDRT(pcalib, pdist, pr0, pqr, pt, ppt, (double (*)[5+5+6])pA); // evaluate dQ/da in pA

      /* clear the columns of the Jacobian corresponding to fixed calibration parameters */
      if(ncK){
        int jj0;

        ptr=pA;
        jj0=5-ncK;
        for(ii=0; ii<mnp; ++ii, ptr+=cnp)
          for(jj=jj0; jj<5; ++jj)
            ptr[jj]=0.0; // ptr[ii*cnp+jj]=0.0;
      }

      /* clear the columns of the Jacobian corresponding to fixed distortion parameters */
      if(ncD){
        int jj0;

        ptr=pA;
        jj0=5-ncD;
        for(ii=0; ii<mnp; ++ii, ptr+=cnp)
          for(jj=jj0; jj<5; ++jj)
            ptr[5+jj]=0.0; // ptr[ii*cnp+5+jj]=0.0;
      }
    }
  }
}

/* BUNDLE ADJUSTMENT FOR STRUCTURE PARAMETERS ONLY */

/* Given a parameter vector p made up of the 3D coordinates of n points, compute in
 * hx the prediction of the measurements, i.e. the projections of 3D points in the m images. The measurements
 * are returned in the order (hx_11^T, .. hx_1m^T, ..., hx_n1^T, .. hx_nm^T)^T, where hx_ij is the predicted
 * projection of the i-th point on the j-th camera.
 * Notice that depending on idxij, some of the hx_ij might be missing
 *
 */
static void img_projsKDS_x(double *p, struct sba_crsm *idxij, int *rcidxs, int *rcsubs, double *hx, void *adata)
{
  register int i, j;
  int cnp, pnp, mnp;
  double *pqr, *pt, *ppt, *pmeas, *pcalib, *pdist, *camparams, trot[FULLQUATSZ];
  //int n;
  int m, nnz;
  struct globs_ *gl;

  gl=(struct globs_ *)adata;
  cnp=gl->cnp; pnp=gl->pnp; mnp=gl->mnp;
  camparams=gl->camparams;

  //n=idxij->nr;
  m=idxij->nc;

  for(j=0; j<m; ++j){
    /* j-th camera parameters */
    pcalib=camparams+j*cnp;
    pdist=pcalib+5;
    pqr=pdist+5;
    pt=pqr+3; // quaternion vector part has 3 elements
    _MK_QUAT_FRM_VEC(trot, pqr);

    nnz=sba_crsm_col_elmidxs(idxij, j, rcidxs, rcsubs); /* find nonzero hx_ij, i=0...n-1 */

    for(i=0; i<nnz; ++i){
      ppt=p + rcsubs[i]*pnp;
      pmeas=hx + idxij->val[rcidxs[i]]*mnp; // set pmeas to point to hx_ij

      calcDistImgProjFullR(pcalib, pdist, trot, pt, ppt, pmeas); // evaluate Q in pmeas
      //calcDistImgProj(pcalib, pdist, (double *)zerorotquat, pqr, pt, ppt, pmeas); // evaluate Q in pmeas
    }
  }
}

/* Given a parameter vector p made up of the 3D coordinates of n points, compute in
 * jac the jacobian of the predicted measurements, i.e. the jacobian of the projections of 3D points in the m images.
 * The jacobian is returned in the order (B_11, ..., B_1m, ..., B_n1, ..., B_nm),
 * where B_ij=dx_ij/db_i (see HZ).
 * Notice that depending on idxij, some of the B_ij might be missing
 *
 */
static void img_projsKDS_jac_x(double *p, struct sba_crsm *idxij, int *rcidxs, int *rcsubs, double *jac, void *adata)
{
  register int i, j;
  int cnp, pnp, mnp;
  double *pqr, *pt, *ppt, *pB, *pcalib, *pdist, *camparams;
  //int n;
  int m, nnz, Bsz;
  struct globs_ *gl;
  
  gl=(struct globs_ *)adata;
  cnp=gl->cnp; pnp=gl->pnp; mnp=gl->mnp;
  camparams=gl->camparams;

  //n=idxij->nr;
  m=idxij->nc;
  Bsz=mnp*pnp;

  for(j=0; j<m; ++j){
    /* j-th camera parameters */
    pcalib=camparams+j*cnp;
    pdist=pcalib+5;
    pqr=pdist+5;
    pt=pqr+3; // quaternion vector part has 3 elements

    nnz=sba_crsm_col_elmidxs(idxij, j, rcidxs, rcsubs); /* find nonzero hx_ij, i=0...n-1 */

    for(i=0; i<nnz; ++i){
      ppt=p + rcsubs[i]*pnp;
      pB=jac + idxij->val[rcidxs[i]]*Bsz; // set pB to point to B_ij

      calcDistImgProjJacS(pcalib, pdist, (double *)zerorotquat, pqr, pt, ppt, (double (*)[3])pB); // dQ/db in pB
    }
  }
}



/* Driver for sba_xxx_levmar */
void sba_driver(char *camsfname, char *ptsfname, char *calibfname, int cnp, int pnp, int mnp,
                void (*caminfilter)(double *pin, int nin, double *pout, int nout),
                void (*camoutfilter)(double *pin, int nin, double *pout, int nout),
                int filecnp, char *refcamsfname, char *refptsfname)
{
  double *motstruct, *motstruct_copy, *imgpts, *covimgpts, *initrot;
  double K[9], ical[5]; // intrinsic calibration matrix & temp. storage for its params
  char *vmask, tbuf[32];
  double opts[SBA_OPTSSZ], info[SBA_INFOSZ], phi;
  int howto, expert, analyticjac, fixedcal, havedist, n, prnt, verbose=0;
  int nframes, numpts3D, numprojs, nvars;
  const int nconstframes=0;
  register int i;
  FILE *fp;

  static char *howtoname[]={"BA_MOTSTRUCT", "BA_MOT", "BA_STRUCT", "BA_MOT_MOTSTRUCT"};

  clock_t start_time, end_time;

  /* Notice the various BA options demonstrated below */

  /* minimize motion & structure, motion only, or
   * motion and possibly motion & structure in a 2nd pass?
   */
  howto=BA_MOTSTRUCT;
  //howto=BA_MOT;
  //howto=BA_STRUCT;
  //howto=BA_MOT_MOTSTRUCT;

  /* simple or expert drivers? */
  //expert=0;
  expert=1;

  /* analytic or approximate jacobian? */
  //analyticjac=0;
  analyticjac=1;

  /* print motion & structure estimates,
   * motion only or structure only upon completion?
   */
  prnt=BA_NONE;
  //prnt=BA_MOTSTRUCT;
  //prnt=BA_MOT;
  //prnt=BA_STRUCT;


  /* NOTE: readInitialSBAEstimate() sets covimgpts to NULL if no covariances are supplied */
  readInitialSBAEstimate(camsfname, ptsfname, cnp, pnp, mnp, caminfilter, filecnp, //NULL, 0, 
                         &nframes, &numpts3D, &numprojs, &motstruct, &initrot, &imgpts, &covimgpts, &vmask);

  //printSBAData(stdout, motstruct, cnp, pnp, mnp, camoutfilter, filecnp, nframes, numpts3D, imgpts, numprojs, vmask);

  if(howto!=BA_STRUCT){
    /* initialize the local rotation estimates to 0, corresponding to local quats (1, 0, 0, 0) */
    for(i=0; i<nframes; ++i){
      register int j;

      j=(i+1)*cnp; // note the +1, below we move from right to left, assuming 3 parameters for the translation!
      motstruct[j-4]=motstruct[j-5]=motstruct[j-6]=0.0; // clear rotation
    }
  }
  /* note that if howto==BA_STRUCT the rotation parts of motstruct actually equal the initial rotations! */

  /* set up globs structure */
  globs.cnp=cnp; globs.pnp=pnp; globs.mnp=mnp;
  globs.rot0params=initrot;
  if(calibfname){ // read intrinsics only if fixed for all cameras
    readCalibParams(calibfname, K);
    ical[0]=K[0]; // fu
    ical[1]=K[2]; // u0
    ical[2]=K[5]; // v0
    ical[3]=K[4]/K[0]; // ar
    ical[4]=K[1]; // s
    globs.intrcalib=ical;
    fixedcal=1; /* use fixed intrinsics */
    havedist=0;
  }
  else{ // intrinsics are to be found in the cameras parameter file
    globs.intrcalib=NULL;
    /* specify the number of intrinsic parameters that are to be fixed
     * equal to their initial values, as follows:
     * 0: all free, 1: skew fixed, 2: skew, ar fixed, 4: skew, ar, ppt fixed
     * Note that a value of 3 does not make sense
     */
    globs.nccalib=2; /* number of intrinsics to keep fixed, must be between 0 and 5 */
    fixedcal=0; /* varying intrinsics */

    if(cnp==16){ // 16 = 5+5+6
      havedist=1; /* with distortion */
      globs.ncdist=3; /* number of distortion params to keep fixed, must be between 0 and 5 */
    }
    else{
      havedist=0;
      globs.ncdist=-9999;
    }
  }

  globs.ptparams=NULL;
  globs.camparams=NULL;

  /* call sparse LM routine */
  opts[0]=SBA_INIT_MU; opts[1]=SBA_STOP_THRESH; opts[2]=SBA_STOP_THRESH;
  opts[3]=SBA_STOP_THRESH;
  //opts[3]=0.05*numprojs; // uncomment to force termination if the average reprojection error drops below 0.05
  opts[4]=0.0;
  //opts[4]=1E-05; // uncomment to force termination if the relative reduction in the RMS reprojection error drops below 1E-05

  start_time=clock();
  switch(howto){
    case BA_MOTSTRUCT: /* BA for motion & structure */
      nvars=nframes*cnp+numpts3D*pnp;
      if(expert)
        n=sba_motstr_levmar_x(numpts3D, 0, nframes, nconstframes, vmask, motstruct, cnp, pnp, imgpts, covimgpts, mnp,
                            fixedcal? img_projsRTS_x : (havedist? img_projsKDRTS_x : img_projsKRTS_x),
                            analyticjac? (fixedcal? img_projsRTS_jac_x : (havedist? img_projsKDRTS_jac_x : img_projsKRTS_jac_x)) : NULL,
                            (void *)(&globs), MAXITER2, verbose, opts, info);
      else
        n=sba_motstr_levmar(numpts3D, 0, nframes, nconstframes, vmask, motstruct, cnp, pnp, imgpts, covimgpts, mnp,
                            fixedcal? img_projRTS : (havedist? img_projKDRTS : img_projKRTS),
                            analyticjac? (fixedcal? img_projRTS_jac : (havedist? img_projKDRTS_jac : img_projKRTS_jac)) : NULL,
                            (void *)(&globs), MAXITER2, verbose, opts, info);
    break;

    case BA_MOT: /* BA for motion only */
      globs.ptparams=motstruct+nframes*cnp;
      nvars=nframes*cnp;
      if(expert)
        n=sba_mot_levmar_x(numpts3D, nframes, nconstframes, vmask, motstruct, cnp, imgpts, covimgpts, mnp,
                          fixedcal? img_projsRT_x : (havedist? img_projsKDRT_x : img_projsKRT_x),
                          analyticjac? (fixedcal? img_projsRT_jac_x : (havedist? img_projsKDRT_jac_x : img_projsKRT_jac_x)) : NULL,
                          (void *)(&globs), MAXITER, verbose, opts, info);
      else
        n=sba_mot_levmar(numpts3D, nframes, nconstframes, vmask, motstruct, cnp, imgpts, covimgpts, mnp,
                          fixedcal? img_projRT : (havedist? img_projKDRT : img_projKRT),
                          analyticjac? (fixedcal? img_projRT_jac : (havedist? img_projKDRT_jac : img_projKRT_jac)) : NULL,
                          (void *)(&globs), MAXITER, verbose, opts, info);
    break;

    case BA_STRUCT: /* BA for structure only */
      globs.camparams=motstruct;
      nvars=numpts3D*pnp;
      if(expert)
        n=sba_str_levmar_x(numpts3D, 0, nframes, vmask, motstruct+nframes*cnp, pnp, imgpts, covimgpts, mnp,
                          fixedcal? img_projsS_x : (havedist? img_projsKDS_x : img_projsKS_x), 
                          analyticjac? (fixedcal? img_projsS_jac_x : (havedist? img_projsKDS_jac_x : img_projsKS_jac_x)) : NULL,
                          (void *)(&globs), MAXITER, verbose, opts, info);
      else
        n=sba_str_levmar(numpts3D, 0, nframes, vmask, motstruct+nframes*cnp, pnp, imgpts, covimgpts, mnp,
                          fixedcal? img_projS : (havedist? img_projKDS : img_projKS),
                          analyticjac? (fixedcal? img_projS_jac : (havedist? img_projKDS_jac : img_projKS_jac)) : NULL,
                          (void *)(&globs), MAXITER, verbose, opts, info);
    break;

    case BA_MOT_MOTSTRUCT: /* BA for motion only; if error too large, then BA for motion & structure */
      if((motstruct_copy=(double *)malloc((nframes*cnp + numpts3D*pnp)*sizeof(double)))==NULL){
        fprintf(stderr, "memory allocation failed in sba_driver()!\n");
        exit(1);
      }

      memcpy(motstruct_copy, motstruct, (nframes*cnp + numpts3D*pnp)*sizeof(double)); // save starting point for later use
      globs.ptparams=motstruct+nframes*cnp;
      nvars=nframes*cnp;

      if(expert)
        n=sba_mot_levmar_x(numpts3D, nframes, nconstframes, vmask, motstruct, cnp, imgpts, covimgpts, mnp,
                          fixedcal? img_projsRT_x : (havedist? img_projsKDRT_x : img_projsKRT_x),
                          analyticjac? (fixedcal? img_projsRT_jac_x : (havedist? img_projsKDRT_jac_x : img_projsKRT_jac_x)) : NULL,
                          (void *)(&globs), MAXITER, verbose, opts, info);
      else
        n=sba_mot_levmar(numpts3D, nframes, nconstframes, vmask, motstruct, cnp, imgpts, covimgpts, mnp,
                        fixedcal? img_projRT : (havedist? img_projKDRT : img_projKRT),
                        analyticjac? (fixedcal? img_projRT_jac : (havedist? img_projKDRT_jac : img_projKRT_jac)) : NULL,
                        (void *)(&globs), MAXITER, verbose, opts, info);

      if((phi=info[1]/numprojs)>SBA_MAX_REPROJ_ERROR){
        fflush(stdout); fprintf(stdout, "Refining structure (motion only error %g)...\n", phi); fflush(stdout);
        memcpy(motstruct, motstruct_copy, (nframes*cnp + numpts3D*pnp)*sizeof(double)); // reset starting point

        if(expert)
          n=sba_motstr_levmar_x(numpts3D, 0, nframes, nconstframes, vmask, motstruct, cnp, pnp, imgpts, covimgpts, mnp,
                                fixedcal? img_projsRTS_x : (havedist? img_projsKDRTS_x : img_projsKRTS_x),
                                analyticjac? (fixedcal? img_projsRTS_jac_x : (havedist? img_projsKDRTS_jac_x : img_projsKRTS_jac_x)) : NULL,
                                (void *)(&globs), MAXITER2, verbose, opts, info);
        else
          n=sba_motstr_levmar(numpts3D, 0, nframes, nconstframes, vmask, motstruct, cnp, pnp, imgpts, covimgpts, mnp,
                              fixedcal? img_projRTS : (havedist? img_projKDRTS : img_projKRTS),
                              analyticjac? (fixedcal? img_projRTS_jac : (havedist? img_projKDRTS_jac : img_projKRTS_jac)) : NULL,
                              (void *)(&globs), MAXITER2, verbose, opts, info);
      }
      free(motstruct_copy);

    break;

    default:
      fprintf(stderr, "unknown BA method \"%d\" in sba_driver()!\n", howto);
      exit(1);
  }
  end_time=clock();

  if(n==SBA_ERROR) goto cleanup;
  
  if(howto!=BA_STRUCT){
    /* combine the local rotation estimates with the initial ones */
    for(i=0; i<nframes; ++i){
      double *v, qs[FULLQUATSZ], *q0, prd[FULLQUATSZ];

      /* retrieve the vector part */
      v=motstruct + (i+1)*cnp - 6; // note the +1, we access the motion parameters from the right, assuming 3 for translation!
      _MK_QUAT_FRM_VEC(qs, v);

      q0=initrot+i*FULLQUATSZ;
      quatMultFast(qs, q0, prd); // prd=qs*q0

      /* copy back vector part making sure that the scalar part is non-negative */
      if(prd[0]>=0.0){
        v[0]=prd[1];
        v[1]=prd[2];
        v[2]=prd[3];
      }
      else{ // negate since two quaternions q and -q represent the same rotation
        v[0]=-prd[1];
        v[1]=-prd[2];
        v[2]=-prd[3];
      }
    }
  }

	fflush(stdout);
  fprintf(stdout, "SBA using %d 3D pts, %d frames and %d image projections, %d variables\n", numpts3D, nframes, numprojs, nvars);
  if(havedist) sprintf(tbuf, " (%d fixed)", globs.ncdist);
  fprintf(stdout, "\nMethod %s, %s driver, %s Jacobian, %s covariances, %s distortion%s, %s intrinsics", howtoname[howto],
                  expert? "expert" : "simple",
                  analyticjac? "analytic" : "approximate",
                  covimgpts? "with" : "without",
                  havedist? "variable" : "without",
                  havedist? tbuf : "",
                  fixedcal? "fixed" : "variable");
  if(!fixedcal) fprintf(stdout, " (%d fixed)", globs.nccalib);
  fputs("\n\n", stdout); 
  fprintf(stdout, "SBA returned %d in %g iter, reason %g, error %g [initial %g], %d/%d func/fjac evals, %d lin. systems\n", n,
                    info[5], info[6], info[1]/numprojs, info[0]/numprojs, (int)info[7], (int)info[8], (int)info[9]);
  fprintf(stdout, "Elapsed time: %.2lf seconds, %.2lf msecs\n", ((double) (end_time - start_time)) / CLOCKS_PER_SEC,
                  ((double) (end_time - start_time)) / CLOCKS_PER_MSEC);
  fflush(stdout);

  /* refined motion and structure are now in motstruct */
  switch(prnt){
    case BA_NONE:
    break;

    case BA_MOTSTRUCT:
      if(!refcamsfname || refcamsfname[0]=='-') fp=stdout;
      else if((fp=fopen(refcamsfname, "w"))==NULL){
        fprintf(stderr, "error opening file %s for writing in sba_driver()!\n", refcamsfname);
        exit(1);
      }
      printSBAMotionData(fp, motstruct, nframes, cnp, camoutfilter, filecnp);
      if(fp!=stdout) fclose(fp);

      if(!refptsfname || refptsfname[0]=='-') fp=stdout;
      else if((fp=fopen(refptsfname, "w"))==NULL){
        fprintf(stderr, "error opening file %s for writing in sba_driver()!\n", refptsfname);
        exit(1);
      }
      printSBAStructureData(fp, motstruct, nframes, numpts3D, cnp, pnp);
      if(fp!=stdout) fclose(fp);
    break;

    case BA_MOT:
      if(!refcamsfname || refcamsfname[0]=='-') fp=stdout;
      else if((fp=fopen(refcamsfname, "w"))==NULL){
        fprintf(stderr, "error opening file %s for writing in sba_driver()!\n", refcamsfname);
        exit(1);
      }
      printSBAMotionData(fp, motstruct, nframes, cnp, camoutfilter, filecnp);
      if(fp!=stdout) fclose(fp);
    break;

    case BA_STRUCT:
      if(!refptsfname || refptsfname[0]=='-') fp=stdout;
      else if((fp=fopen(refptsfname, "w"))==NULL){
        fprintf(stderr, "error opening file %s for writing in sba_driver()!\n", refptsfname);
        exit(1);
      }
      printSBAStructureData(fp, motstruct, nframes, numpts3D, cnp, pnp);
      if(fp!=stdout) fclose(fp);
    break;

    default:
      fprintf(stderr, "unknown print option \"%d\" in sba_driver()!\n", prnt);
      exit(1);
  }

  //printSBAData(stdout, motstruct, cnp, pnp, mnp, camoutfilter, filecnp, nframes, numpts3D, imgpts, numprojs, vmask);


cleanup:
  /* just in case... */
  globs.intrcalib=NULL;
  globs.nccalib=0;
  globs.ncdist=0;

  free(motstruct);
  free(imgpts);
  free(initrot); globs.rot0params=NULL;
  if(covimgpts) free(covimgpts);
  free(vmask);
}

/* convert a vector of camera parameters so that rotation is represented by
 * the vector part of the input quaternion. The function converts the
 * input quaternion into a unit one with a non-negative scalar part. Remaining
 * parameters are left unchanged.
 *
 * Input parameter layout: intrinsics (5, optional), distortion (5, optional), rot. quaternion (4), translation (3)
 * Output parameter layout: intrinsics (5, optional), distortion (5, optional), rot. quaternion vector part (3), translation (3)
 */
void quat2vec(double *inp, int nin, double *outp, int nout)
{
double mag, sg;
register int i;

  /* intrinsics & distortion */
  if(nin>7) // are they present?
    for(i=0; i<nin-7; ++i)
      outp[i]=inp[i];
  else
    i=0;

  /* rotation */
  /* normalize and ensure that the quaternion's scalar component is non-negative;
   * if not, negate the quaternion since two quaternions q and -q represent the
   * same rotation
   */
  mag=sqrt(inp[i]*inp[i] + inp[i+1]*inp[i+1] + inp[i+2]*inp[i+2] + inp[i+3]*inp[i+3]);
  sg=(inp[i]>=0.0)? 1.0 : -1.0;
  mag=sg/mag;
  outp[i]  =inp[i+1]*mag;
  outp[i+1]=inp[i+2]*mag;
  outp[i+2]=inp[i+3]*mag;
  i+=3;

  /* translation*/
  for( ; i<nout; ++i)
    outp[i]=inp[i+1];
}

/* convert a vector of camera parameters so that rotation is represented by
 * a full unit quaternion instead of its input 3-vector part. Remaining
 * parameters are left unchanged.
 *
 * Input parameter layout: intrinsics (5, optional), distortion (5, optional), rot. quaternion vector part (3), translation (3)
 * Output parameter layout: intrinsics (5, optional), distortion (5, optional), rot. quaternion (4), translation (3)
 */
void vec2quat(double *inp, int nin, double *outp, int nout)
{
double *v, q[FULLQUATSZ];
register int i;

  /* intrinsics & distortion */
  if(nin>7-1) // are they present?
    for(i=0; i<nin-(7-1); ++i)
      outp[i]=inp[i];
  else
    i=0;

  /* rotation */
  /* recover the quaternion from the vector */
  v=inp+i;
  _MK_QUAT_FRM_VEC(q, v);
  outp[i]  =q[0];
  outp[i+1]=q[1];
  outp[i+2]=q[2];
  outp[i+3]=q[3];
  i+=FULLQUATSZ;

  /* translation */
  for( ; i<nout; ++i)
    outp[i]=inp[i-1];
}


int main(int argc, char *argv[])
{
int cnp=6, /* 3 rot params + 3 trans params */
    pnp=3, /* euclidean 3D points */
    mnp=2; /* image points are 2D */

  if(argc!=3 && argc!=4){
    fprintf(stderr, "Usage is %s <camera params> <point params> [<intrinsic calibration params>]\n", argv[0]);
    exit(1);
  }

  if(argc==4){ /* fixed intrinsics */
    fprintf(stderr, "Starting BA with fixed intrinsic parameters\n");
    sba_driver(argv[1], argv[2], argv[3], cnp, pnp, mnp, quat2vec, vec2quat, cnp+1, "-", "-");
  }
  else{
    cnp+=5; /* 5 more params for calibration */
    if(readNumParams(argv[1])-1==cnp+5){ /* with distortion */
      cnp+=5; /* 5 more params for distortion */
      fprintf(stderr, "Starting BA with varying intrinsic parameters & distortion\n");
      sba_driver(argv[1], argv[2], NULL, cnp, pnp, mnp, quat2vec, vec2quat, cnp+1, "-", "-");
    }
    else{
      fprintf(stderr, "Starting BA with varying intrinsic parameters\n");
      sba_driver(argv[1], argv[2], NULL, cnp, pnp, mnp, quat2vec, vec2quat, cnp+1, "-", "-");
    }
  }

  return 0;
}
