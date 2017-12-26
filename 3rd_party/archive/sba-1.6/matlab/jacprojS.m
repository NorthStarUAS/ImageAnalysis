function jst=jacprojS(j, i, rt, xyz, r0, a)
% symbolic projection function Jacobian
% code automatically generated with maple

  qr0=r0(j*4+1:(j+1)*4);

  t1 = (rt(1) ^ 2);
  t2 = (rt(2) ^ 2);
  t3 = (rt(3) ^ 2);
  t5 = sqrt((1 - t1 - t2 - t3));
  t10 = -t5 * qr0(2) - qr0(1) * rt(1) - rt(2) * qr0(4) + rt(3) * qr0(3);
  t11 = t10 ^ 2;
  t16 = t5 * qr0(1) - rt(1) * qr0(2) - rt(2) * qr0(3) - rt(3) * qr0(4);
  t17 = t16 ^ 2;
  t22 = t5 * qr0(4) + qr0(1) * rt(3) + rt(1) * qr0(3) - rt(2) * qr0(2);
  t28 = -t5 * qr0(3) - qr0(1) * rt(2) - rt(3) * qr0(2) + rt(1) * qr0(4);
  t29 = t28 ^ 2;
  t32 = t10 * t28;
  t35 = -t16 * t22;
  t36 = 0.2e1 * t32 + t16 * t22 - t35;
  t38 = -t10 * t22;
  t39 = t16 * t28;
  t42 = t38 + 0.2e1 * t39 - t10 * t22;
  t48 = t10 * xyz(1) + t28 * xyz(2) - t22 * xyz(3);
  t53 = t16 * xyz(3) - t10 * xyz(2) + t28 * xyz(1);
  t58 = t16 * xyz(1) - t28 * xyz(3) - t22 * xyz(2);
  t63 = t16 * xyz(2) + t22 * xyz(1) + t10 * xyz(3);
  t65 = -t48 * t22 + t16 * t53 + t58 * t28 - t63 * t10 + rt(6);
  t66 = 0.1e1 / t65;
  t78 = t48 * t28 + t16 * t63 + t53 * t10 + t58 * t22 + rt(5);
  t82 = t65 ^ 2;
  t83 = 0.1e1 / t82;
  t84 = (a(1) * (t48 * t10 + t16 * t58 - t63 * t22 - t53 * t28 + rt(4)) + a(2) * t78 + a(3) * t65) * t83;
  t92 = t22 ^ 2;
  t93 = t29 + t17 - t10 ^ 2 - t92;
  t95 = -t28 * t22;
  t98 = t16 * t10;
  t99 = 0.2e1 * t95 - t16 * t10 - t98;
  t111 = t95 + 0.2e1 * t98 - t28 * t22;
  t114 = t92 + t17 - t28 ^ 2 - t11;
  t127 = (a(4) * t78 + a(5) * t65) * t83;
  jst(1) = (a(1) * (t11 + t17 - t22 ^ 2 - t29) + a(2) * t36 + a(3) * t42) * t66 - t84 * t42;
  jst(2) = (a(1) * (t32 + 0.2e1 * t35 + t10 * t28) + a(2) * t93 + a(3) * t99) * t66 - t84 * t99;
  jst(3) = (a(1) * (0.2e1 * t38 - t16 * t28 - t39) + a(2) * t111 + a(3) * t114) * t66 - t84 * t114;
  jst(4) = (a(4) * t36 + a(5) * t42) * t66 - t127 * t42;
  jst(5) = (a(4) * t93 + a(5) * t99) * t66 - t127 * t99;
  jst(6) = (a(4) * t111 + a(5) * t114) * t66 - t127 * t114;
