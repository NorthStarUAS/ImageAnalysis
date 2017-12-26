#include <math.h>

/* Sample C code for the projection and Jacobian functions for Euclidean BA,
 * to be loaded from a shared (i.e., dynamic) library by sba's matlab MEX interface
 *
 * code automatically generated with maple
 */

/* Compilation instructions:
 *
 * Un*x/GCC:    gcc -fPIC -O3 -shared -o projac.so projac.c
 * Win32/MSVC:  cl /nologo /O2 projac.c /link /dll /out:projac.dll
 */

#if defined(_MSC_VER) /* DLL directives for MSVC */
#define API_MOD    __declspec(dllexport)
#define CALL_CONV  __cdecl
#else /* define empty */
#define API_MOD 
#define CALL_CONV
#endif /* _MSC_VER */

API_MOD void CALL_CONV imgproj_motstr(int j, int i, double *rt, double *xyz, double *m, double **adata)
{
double *qr0, *r0=adata[0], *a=adata[1];
double t1, t2, t3, t5, t10, t16, t22, t24, t30, t34, t39, t44, t52, t58, t61;

  qr0=r0+j*4;

  t1 = pow(rt[0], 0.2e1);
  t2 = pow(rt[1], 0.2e1);
  t3 = pow(rt[2], 0.2e1);
  t5 = sqrt(0.1e1 - t1 - t2 - t3);
  t10 = t5 * qr0[1] + qr0[0] * rt[0] + rt[1] * qr0[3] - rt[2] * qr0[2];
  t16 = t5 * qr0[2] + qr0[0] * rt[1] + rt[2] * qr0[1] - rt[0] * qr0[3];
  t22 = t5 * qr0[3] + qr0[0] * rt[2] + rt[0] * qr0[2] - rt[1] * qr0[1];
  t24 = -t10 * xyz[0] - t16 * xyz[1] - t22 * xyz[2];
  t30 = t5 * qr0[0] - rt[0] * qr0[1] - rt[1] * qr0[2] - rt[2] * qr0[3];
  t34 = t30 * xyz[0] + t16 * xyz[2] - t22 * xyz[1];
  t39 = t30 * xyz[1] + t22 * xyz[0] - t10 * xyz[2];
  t44 = t30 * xyz[2] + t10 * xyz[1] - t16 * xyz[0];
  t52 = -t24 * t16 + t30 * t39 - t44 * t10 + t34 * t22 + rt[4];
  t58 = -t24 * t22 + t30 * t44 - t34 * t16 + t39 * t10 + rt[5];
  t61 = 0.1e1 / t58;
  m[0] = (a[0] * (-t24 * t10 + t30 * t34 - t39 * t22 + t44 * t16 + rt[3]) + a[1] * t52 + a[2] * t58) * t61;
  m[1] = (a[3] * t52 + a[4] * t58) * t61;
}

API_MOD void CALL_CONV imgprojac_motstr(int j, int i, double *rt, double *xyz, double *jrt, double *jst, double **adata)
{
double *qr0, *r0=adata[0], *a=adata[1];
double t1, t2, t3, t5, t6, t7, t9, t11, t13, t15, t17, t19, t24, t31, t37,
t39, t41, t43, t48, t52, t57, t62, t67, t72, t77, t89, t99, t106,
t107, t119, t123, t124, t125, t129, t132, t135, t137, t141, t146,
t151, t157, t170, t180, t187, t190, t193, t195, t199, t204, t209, 
t215, t228, t238, t255, t10, t16, t22, t28, t29, t32, t35, t36,
t38, t42, t53, t58, t63, t65, t66, t78, t82, t83, t84, t92, t93,
t95, t98, t111, t114, t127;

  qr0=r0+j*4;

  t1 = pow(rt[0], 0.2e1);
  t2 = pow(rt[1], 0.2e1);
  t3 = pow(rt[2], 0.2e1);
  t5 = sqrt(0.1e1 - t1 - t2 - t3);
  t6 = 0.1e1 / t5;
  t7 = t6 * qr0[1];
  t9 = -t7 * rt[0] + qr0[0];
  t11 = t6 * qr0[2];
  t13 = -t11 * rt[0] - qr0[3];
  t15 = t6 * qr0[3];
  t17 = -t15 * rt[0] + qr0[2];
  t19 = -t9 * xyz[0] - t13 * xyz[1] - t17 * xyz[2];
  t24 = -t5 * qr0[1] - qr0[0] * rt[0] - rt[1] * qr0[3] + rt[2] * qr0[2];
  t31 = t5 * qr0[2] + qr0[0] * rt[1] + rt[2] * qr0[1] - rt[0] * qr0[3];
  t37 = t5 * qr0[3] + qr0[0] * rt[2] + rt[0] * qr0[2] - rt[1] * qr0[1];
  t39 = t24 * xyz[0] - t31 * xyz[1] - t37 * xyz[2];
  t41 = t6 * qr0[0];
  t43 = -t41 * rt[0] - qr0[1];
  t48 = t5 * qr0[0] - rt[0] * qr0[1] - rt[1] * qr0[2] - rt[2] * qr0[3];
  t52 = t48 * xyz[0] + t31 * xyz[2] - t37 * xyz[1];
  t57 = t43 * xyz[0] + t13 * xyz[2] - t17 * xyz[1];
  t62 = t43 * xyz[1] + t17 * xyz[0] - t9 * xyz[2];
  t67 = t48 * xyz[1] + t37 * xyz[0] + t24 * xyz[2];
  t72 = t43 * xyz[2] + t9 * xyz[1] - t13 * xyz[0];
  t77 = t48 * xyz[2] - t24 * xyz[1] - t31 * xyz[0];
  t89 = -t19 * t31 - t39 * t13 + t43 * t67 + t48 * t62 + t72 * t24 - t77 * t9 + t57 * t37 + t52 * t17;
  t99 = -t19 * t37 - t39 * t17 + t43 * t77 + t48 * t72 - t57 * t31 - t52 * t13 - t62 * t24 + t67 * t9;
  t106 = -t39 * t37 + t48 * t77 - t52 * t31 - t67 * t24 + rt[5];
  t107 = 0.1e1 / t106;
  t119 = -t39 * t31 + t48 * t67 + t77 * t24 + t52 * t37 + rt[4];
  t123 = t106 * t106;
  t124 = 0.1e1 / t123;
  t125 = (a[0] * (t39 * t24 + t48 * t52 - t67 * t37 + t77 * t31 + rt[3]) + a[1] * t119 + a[2] * t106) * t124;
  t129 = -t7 * rt[1] + qr0[3];
  t132 = -t11 * rt[1] + qr0[0];
  t135 = -t15 * rt[1] - qr0[1];
  t137 = -t129 * xyz[0] - t132 * xyz[1] - t135 * xyz[2];
  t141 = -t41 * rt[1] - qr0[2];
  t146 = t141 * xyz[0] + t132 * xyz[2] - t135 * xyz[1];
  t151 = t141 * xyz[1] + t135 * xyz[0] - t129 * xyz[2];
  t157 = t141 * xyz[2] + t129 * xyz[1] - t132 * xyz[0];
  t170 = -t137 * t31 - t39 * t132 + t141 * t67 + t48 * t151 + t157 * t24 - t77 * t129 + t146 * t37 + t52 * t135;
  t180 = -t137 * t37 - t39 * t135 + t141 * t77 + t48 * t157 - t146 * t31 - t52 * t132 - t151 * t24 + t67 * t129;
  t187 = -t7 * rt[2] - qr0[2];
  t190 = -t11 * rt[2] + qr0[1];
  t193 = -t15 * rt[2] + qr0[0];
  t195 = -t187 * xyz[0] - t190 * xyz[1] - t193 * xyz[2];
  t199 = -t41 * rt[2] - qr0[3];
  t204 = t199 * xyz[0] + t190 * xyz[2] - t193 * xyz[1];
  t209 = t199 * xyz[1] + t193 * xyz[0] - t187 * xyz[2];
  t215 = t199 * xyz[2] + t187 * xyz[1] - t190 * xyz[0];
  t228 = -t195 * t31 - t39 * t190 + t199 * t67 + t48 * t209 + t215 * t24 - t77 * t187 + t204 * t37 + t52 * t193;
  t238 = -t195 * t37 - t39 * t193 + t199 * t77 + t48 * t215 - t204 * t31 - t52 * t190 - t209 * t24 + t67 * t187;
  t255 = (a[3] * t119 + a[4] * t106) * t124;
  jrt[0] = (a[0] * (t19 * t24 - t39 * t9 + t43 * t52 + t48 * t57 - t62 * t37 - t67 * t17 + t72 * t31 + t77 * t13) + a[1] * t89 + a[2] * t99) * t107 - t125 * t99;
  jrt[1] = (a[0] * (t137 * t24 - t39 * t129 + t141 * t52 + t48 * t146 - t151 * t37 - t67 * t135 + t157 * t31 + t77 * t132) + a[1] * t170 + a[2] * t180) * t107 - t125 * t180;
  jrt[2] = (a[0] * (t195 * t24 - t39 * t187 + t199 * t52 + t48 * t204 - t209 * t37 - t67 * t193 + t215 * t31 + t77 * t190) + a[1] * t228 + a[2] * t238) * t107 - t125 * t238;
  jrt[3] = a[0] * t107;
  jrt[4] = a[1] * t107;
  jrt[5] = a[2] * t107 - t125;
  jrt[6] = (a[3] * t89 + a[4] * t99) * t107 - t255 * t99;
  jrt[7] = (a[3] * t170 + a[4] * t180) * t107 - t255 * t180;
  jrt[8] = (a[3] * t228 + a[4] * t238) * t107 - t255 * t238;
  jrt[9] = 0.0e0;
  jrt[10] = a[3] * t107;
  jrt[11] = a[4] * t107 - t255;

  t1 = pow(rt[0], 0.2e1);
  t2 = pow(rt[1], 0.2e1);
  t3 = pow(rt[2], 0.2e1);
  t5 = sqrt(0.1e1 - t1 - t2 - t3);
  t10 = -t5 * qr0[1] - qr0[0] * rt[0] - rt[1] * qr0[3] + rt[2] * qr0[2];
  t11 = t10 * t10;
  t16 = t5 * qr0[0] - rt[0] * qr0[1] - rt[1] * qr0[2] - rt[2] * qr0[3];
  t17 = t16 * t16;
  t22 = t5 * qr0[3] + qr0[0] * rt[2] + rt[0] * qr0[2] - rt[1] * qr0[1];
  t28 = -t5 * qr0[2] - qr0[0] * rt[1] - rt[2] * qr0[1] + rt[0] * qr0[3];
  t29 = t28 * t28;
  t32 = t10 * t28;
  t35 = -t16 * t22;
  t36 = 0.2e1 * t32 + t16 * t22 - t35;
  t38 = -t10 * t22;
  t39 = t16 * t28;
  t42 = t38 + 0.2e1 * t39 - t10 * t22;
  t48 = t10 * xyz[0] + t28 * xyz[1] - t22 * xyz[2];
  t53 = t16 * xyz[2] - t10 * xyz[1] + t28 * xyz[0];
  t58 = t16 * xyz[0] - t28 * xyz[2] - t22 * xyz[1];
  t63 = t16 * xyz[1] + t22 * xyz[0] + t10 * xyz[2];
  t65 = -t48 * t22 + t16 * t53 + t58 * t28 - t63 * t10 + rt[5];
  t66 = 0.1e1 / t65;
  t78 = t48 * t28 + t16 * t63 + t53 * t10 + t58 * t22 + rt[4];
  t82 = t65 * t65;
  t83 = 0.1e1 / t82;
  t84 = (a[0] * (t48 * t10 + t16 * t58 - t63 * t22 - t53 * t28 + rt[3]) + a[1] * t78 + a[2] * t65) * t83;
  t92 = t22 * t22;
  t93 = t29 + t17 - t10 * t10 - t92;
  t95 = -t28 * t22;
  t98 = t16 * t10;
  t99 = 0.2e1 * t95 - t16 * t10 - t98;
  t111 = t95 + 0.2e1 * t98 - t28 * t22;
  t114 = t92 + t17 - t28 * t28 - t11;
  t127 = (a[3] * t78 + a[4] * t65) * t83;
  jst[0] = (a[0] * (t11 + t17 - t22 * t22 - t29) + a[1] * t36 + a[2] * t42) * t66 - t84 * t42;
  jst[1] = (a[0] * (t32 + 0.2e1 * t35 + t10 * t28) + a[1] * t93 + a[2] * t99) * t66 - t84 * t99;
  jst[2] = (a[0] * (0.2e1 * t38 - t16 * t28 - t39) + a[1] * t111 + a[2] * t114) * t66 - t84 * t114;
  jst[3] = (a[3] * t36 + a[4] * t42) * t66 - t127 * t42;
  jst[4] = (a[3] * t93 + a[4] * t99) * t66 - t127 * t99;
  jst[5] = (a[3] * t111 + a[4] * t114) * t66 - t127 * t114;
}


API_MOD void CALL_CONV imgprojac_mot(int j, int i, double *rt, double *xyz, double *jrt, double **adata)
{
double *qr0, *r0=adata[0], *a=adata[1];
double t1, t2, t3, t5, t6, t7, t9, t11, t13, t15, t17, t19, t24, t31, t37,
t39, t41, t43, t48, t52, t57, t62, t67, t72, t77, t89, t99, t106,
t107, t119, t123, t124, t125, t129, t132, t135, t137, t141, t146,
t151, t157, t170, t180, t187, t190, t193, t195, t199, t204, t209, 
t215, t228, t238, t255;

  qr0=r0+j*4;

  t1 = pow(rt[0], 0.2e1);
  t2 = pow(rt[1], 0.2e1);
  t3 = pow(rt[2], 0.2e1);
  t5 = sqrt(0.1e1 - t1 - t2 - t3);
  t6 = 0.1e1 / t5;
  t7 = t6 * qr0[1];
  t9 = -t7 * rt[0] + qr0[0];
  t11 = t6 * qr0[2];
  t13 = -t11 * rt[0] - qr0[3];
  t15 = t6 * qr0[3];
  t17 = -t15 * rt[0] + qr0[2];
  t19 = -t9 * xyz[0] - t13 * xyz[1] - t17 * xyz[2];
  t24 = -t5 * qr0[1] - qr0[0] * rt[0] - rt[1] * qr0[3] + rt[2] * qr0[2];
  t31 = t5 * qr0[2] + qr0[0] * rt[1] + rt[2] * qr0[1] - rt[0] * qr0[3];
  t37 = t5 * qr0[3] + qr0[0] * rt[2] + rt[0] * qr0[2] - rt[1] * qr0[1];
  t39 = t24 * xyz[0] - t31 * xyz[1] - t37 * xyz[2];
  t41 = t6 * qr0[0];
  t43 = -t41 * rt[0] - qr0[1];
  t48 = t5 * qr0[0] - rt[0] * qr0[1] - rt[1] * qr0[2] - rt[2] * qr0[3];
  t52 = t48 * xyz[0] + t31 * xyz[2] - t37 * xyz[1];
  t57 = t43 * xyz[0] + t13 * xyz[2] - t17 * xyz[1];
  t62 = t43 * xyz[1] + t17 * xyz[0] - t9 * xyz[2];
  t67 = t48 * xyz[1] + t37 * xyz[0] + t24 * xyz[2];
  t72 = t43 * xyz[2] + t9 * xyz[1] - t13 * xyz[0];
  t77 = t48 * xyz[2] - t24 * xyz[1] - t31 * xyz[0];
  t89 = -t19 * t31 - t39 * t13 + t43 * t67 + t48 * t62 + t72 * t24 - t77 * t9 + t57 * t37 + t52 * t17;
  t99 = -t19 * t37 - t39 * t17 + t43 * t77 + t48 * t72 - t57 * t31 - t52 * t13 - t62 * t24 + t67 * t9;
  t106 = -t39 * t37 + t48 * t77 - t52 * t31 - t67 * t24 + rt[5];
  t107 = 0.1e1 / t106;
  t119 = -t39 * t31 + t48 * t67 + t77 * t24 + t52 * t37 + rt[4];
  t123 = t106 * t106;
  t124 = 0.1e1 / t123;
  t125 = (a[0] * (t39 * t24 + t48 * t52 - t67 * t37 + t77 * t31 + rt[3]) + a[1] * t119 + a[2] * t106) * t124;
  t129 = -t7 * rt[1] + qr0[3];
  t132 = -t11 * rt[1] + qr0[0];
  t135 = -t15 * rt[1] - qr0[1];
  t137 = -t129 * xyz[0] - t132 * xyz[1] - t135 * xyz[2];
  t141 = -t41 * rt[1] - qr0[2];
  t146 = t141 * xyz[0] + t132 * xyz[2] - t135 * xyz[1];
  t151 = t141 * xyz[1] + t135 * xyz[0] - t129 * xyz[2];
  t157 = t141 * xyz[2] + t129 * xyz[1] - t132 * xyz[0];
  t170 = -t137 * t31 - t39 * t132 + t141 * t67 + t48 * t151 + t157 * t24 - t77 * t129 + t146 * t37 + t52 * t135;
  t180 = -t137 * t37 - t39 * t135 + t141 * t77 + t48 * t157 - t146 * t31 - t52 * t132 - t151 * t24 + t67 * t129;
  t187 = -t7 * rt[2] - qr0[2];
  t190 = -t11 * rt[2] + qr0[1];
  t193 = -t15 * rt[2] + qr0[0];
  t195 = -t187 * xyz[0] - t190 * xyz[1] - t193 * xyz[2];
  t199 = -t41 * rt[2] - qr0[3];
  t204 = t199 * xyz[0] + t190 * xyz[2] - t193 * xyz[1];
  t209 = t199 * xyz[1] + t193 * xyz[0] - t187 * xyz[2];
  t215 = t199 * xyz[2] + t187 * xyz[1] - t190 * xyz[0];
  t228 = -t195 * t31 - t39 * t190 + t199 * t67 + t48 * t209 + t215 * t24 - t77 * t187 + t204 * t37 + t52 * t193;
  t238 = -t195 * t37 - t39 * t193 + t199 * t77 + t48 * t215 - t204 * t31 - t52 * t190 - t209 * t24 + t67 * t187;
  t255 = (a[3] * t119 + a[4] * t106) * t124;
  jrt[0] = (a[0] * (t19 * t24 - t39 * t9 + t43 * t52 + t48 * t57 - t62 * t37 - t67 * t17 + t72 * t31 + t77 * t13) + a[1] * t89 + a[2] * t99) * t107 - t125 * t99;
  jrt[1] = (a[0] * (t137 * t24 - t39 * t129 + t141 * t52 + t48 * t146 - t151 * t37 - t67 * t135 + t157 * t31 + t77 * t132) + a[1] * t170 + a[2] * t180) * t107 - t125 * t180;
  jrt[2] = (a[0] * (t195 * t24 - t39 * t187 + t199 * t52 + t48 * t204 - t209 * t37 - t67 * t193 + t215 * t31 + t77 * t190) + a[1] * t228 + a[2] * t238) * t107 - t125 * t238;
  jrt[3] = a[0] * t107;
  jrt[4] = a[1] * t107;
  jrt[5] = a[2] * t107 - t125;
  jrt[6] = (a[3] * t89 + a[4] * t99) * t107 - t255 * t99;
  jrt[7] = (a[3] * t170 + a[4] * t180) * t107 - t255 * t180;
  jrt[8] = (a[3] * t228 + a[4] * t238) * t107 - t255 * t238;
  jrt[9] = 0.0e0;
  jrt[10] = a[3] * t107;
  jrt[11] = a[4] * t107 - t255;
}


API_MOD void CALL_CONV imgprojac_str(int j, int i, double *rt, double *xyz, double *jst, double **adata)
{
double *qr0, *r0=adata[0], *a=adata[1];
double t1, t2, t3, t5, t10, t11, t16, t17, t22, t28, t29, t32,
t35, t36, t38, t39, t42, t48, t53, t58, t63, t65, t66,
t78, t82, t83, t84, t92, t93, t95, t98, t99, t111, 
t114, t127;

  qr0=r0+j*4;

  t1 = pow(rt[0], 0.2e1);
  t2 = pow(rt[1], 0.2e1);
  t3 = pow(rt[2], 0.2e1);
  t5 = sqrt(0.1e1 - t1 - t2 - t3);
  t10 = -t5 * qr0[1] - qr0[0] * rt[0] - rt[1] * qr0[3] + rt[2] * qr0[2];
  t11 = t10 * t10;
  t16 = t5 * qr0[0] - rt[0] * qr0[1] - rt[1] * qr0[2] - rt[2] * qr0[3];
  t17 = t16 * t16;
  t22 = t5 * qr0[3] + qr0[0] * rt[2] + rt[0] * qr0[2] - rt[1] * qr0[1];
  t28 = -t5 * qr0[2] - qr0[0] * rt[1] - rt[2] * qr0[1] + rt[0] * qr0[3];
  t29 = t28 * t28;
  t32 = t10 * t28;
  t35 = -t16 * t22;
  t36 = 0.2e1 * t32 + t16 * t22 - t35;
  t38 = -t10 * t22;
  t39 = t16 * t28;
  t42 = t38 + 0.2e1 * t39 - t10 * t22;
  t48 = t10 * xyz[0] + t28 * xyz[1] - t22 * xyz[2];
  t53 = t16 * xyz[2] - t10 * xyz[1] + t28 * xyz[0];
  t58 = t16 * xyz[0] - t28 * xyz[2] - t22 * xyz[1];
  t63 = t16 * xyz[1] + t22 * xyz[0] + t10 * xyz[2];
  t65 = -t48 * t22 + t16 * t53 + t58 * t28 - t63 * t10 + rt[5];
  t66 = 0.1e1 / t65;
  t78 = t48 * t28 + t16 * t63 + t53 * t10 + t58 * t22 + rt[4];
  t82 = t65 * t65;
  t83 = 0.1e1 / t82;
  t84 = (a[0] * (t48 * t10 + t16 * t58 - t63 * t22 - t53 * t28 + rt[3]) + a[1] * t78 + a[2] * t65) * t83;
  t92 = t22 * t22;
  t93 = t29 + t17 - t10 * t10 - t92;
  t95 = -t28 * t22;
  t98 = t16 * t10;
  t99 = 0.2e1 * t95 - t16 * t10 - t98;
  t111 = t95 + 0.2e1 * t98 - t28 * t22;
  t114 = t92 + t17 - t28 * t28 - t11;
  t127 = (a[3] * t78 + a[4] * t65) * t83;
  jst[0] = (a[0] * (t11 + t17 - t22 * t22 - t29) + a[1] * t36 + a[2] * t42) * t66 - t84 * t42;
  jst[1] = (a[0] * (t32 + 0.2e1 * t35 + t10 * t28) + a[1] * t93 + a[2] * t99) * t66 - t84 * t99;
  jst[2] = (a[0] * (0.2e1 * t38 - t16 * t28 - t39) + a[1] * t111 + a[2] * t114) * t66 - t84 * t114;
  jst[3] = (a[3] * t36 + a[4] * t42) * t66 - t127 * t42;
  jst[4] = (a[3] * t93 + a[4] * t99) * t66 - t127 * t99;
  jst[5] = (a[3] * t111 + a[4] * t114) * t66 - t127 * t114;
}
