function m=projRTS(j, i, rt, xyz, r0, a)
% symbolic projection function
% code automatically generated with maple

  qr0=r0(j*4+1:(j+1)*4);

  t1 = (rt(1) ^ 2);
  t2 = (rt(2) ^ 2);
  t3 = (rt(3) ^ 2);
  t5 = sqrt((1 - t1 - t2 - t3));
  t10 = t5 * qr0(2) + qr0(1) * rt(1) + rt(2) * qr0(4) - rt(3) * qr0(3);
  t16 = t5 * qr0(3) + qr0(1) * rt(2) + rt(3) * qr0(2) - rt(1) * qr0(4);
  t22 = t5 * qr0(4) + qr0(1) * rt(3) + rt(1) * qr0(3) - rt(2) * qr0(2);
  t24 = -t10 * xyz(1) - t16 * xyz(2) - t22 * xyz(3);
  t30 = t5 * qr0(1) - rt(1) * qr0(2) - rt(2) * qr0(3) - rt(3) * qr0(4);
  t34 = t30 * xyz(1) + t16 * xyz(3) - t22 * xyz(2);
  t39 = t30 * xyz(2) + t22 * xyz(1) - t10 * xyz(3);
  t44 = t30 * xyz(3) + t10 * xyz(2) - t16 * xyz(1);
  t52 = -t24 * t16 + t30 * t39 - t44 * t10 + t34 * t22 + rt(5);
  t58 = -t24 * t22 + t30 * t44 - t34 * t16 + t39 * t10 + rt(6);
  t61 = 0.1e1 / t58;
  m(1) = (a(1) * (-t24 * t10 + t30 * t34 - t39 * t22 + t44 * t16 + rt(4)) + a(2) * t52 + a(3) * t58) * t61;
  m(2) = (a(4) * t52 + a(5) * t58) * t61;
