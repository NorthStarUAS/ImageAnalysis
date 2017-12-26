#!/usr/bin/perl -w

#################################################################################
## 
##  Perl functions for manipulating quaternions. Currently includes converters
##  between rotation matrices and quaternions.
##  Copyright (C) 2008  Manolis Lourakis (lourakis at ics.forth.gr)
##  Institute of Computer Science, Foundation for Research & Technology - Hellas
##  Heraklion, Crete, Greece.
##
##  This program is free software; you can redistribute it and/or modify
##  it under the terms of the GNU General Public License as published by
##  the Free Software Foundation; either version 2 of the License, or
##  (at your option) any later version.
##
##  This program is distributed in the hope that it will be useful,
##  but WITHOUT ANY WARRANTY; without even the implied warranty of
##  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##  GNU General Public License for more details.
##
#################################################################################


$R[0]=3.2471667291e-01; $R[1]=-1.0372666302e-01; $R[2]=-9.4010630341e-01;
$R[3]=2.7245486596e-01; $R[4]=9.6209316145e-01;  $R[5]=-1.2045526681e-02;
$R[6]=9.0571928783e-01; $R[7]=-2.5222515354e-01; $R[8]=3.4066852448e-01;

@q=rotmat2quat(@R);

printf "Quat from R: %g %g %g %g\n\n", $q[0], $q[1], $q[2], $q[3];
@Rn=quat2rotmat(@q);
printf "R converted back from quat\n";
printf "%g %g %g\n", $Rn[0], $Rn[1], $Rn[2];
printf "%g %g %g\n", $Rn[3], $Rn[4], $Rn[5];
printf "%g %g %g\n", $Rn[6], $Rn[7], $Rn[8];



# compute the quaternion corresponding to a rotation matrix; see A8 in Horn's paper 
sub rotmat2quat{
  my ($R)=@_;
  my ($q, $i, $maxpos, @tmp, $mag);  

	# find the maximum of the 4 quantities
	$tmp[0]=1.0 + $R[0] + $R[4] + $R[8];
	$tmp[1]=1.0 + $R[0] - $R[4] - $R[8];
	$tmp[2]=1.0 - $R[0] + $R[4] - $R[8];
	$tmp[3]=1.0 - $R[0] - $R[4] + $R[8];

	for($i=0, $mag=-1.0; $i<4; $i++){
		if($tmp[$i]>$mag){
			$mag=$tmp[$i];
			$maxpos=$i;
		}
  }

  if($maxpos==0){
		$q[0]=sqrt($tmp[0])*0.5;
		$q[1]=($R[7] - $R[5])/(4.0*$q[0]);
		$q[2]=($R[2] - $R[6])/(4.0*$q[0]);
		$q[3]=($R[3] - $R[1])/(4.0*$q[0]);
  }
  elsif($maxpos==1){
		$q[1]=sqrt($tmp[1])*0.5;
		$q[0]=($R[7] - $R[5])/(4.0*$q[1]);
		$q[2]=($R[3] + $R[1])/(4.0*$q[1]);
		$q[3]=($R[2] + $R[6])/(4.0*$q[1]);
  }
  elsif($maxpos==2){
		$q[2]=sqrt($tmp[2])*0.5;
		$q[0]=($R[2] - $R[6])/(4.0*$q[2]);
		$q[1]=($R[3] + $R[1])/(4.0*$q[2]);
		$q[3]=($R[7] + $R[5])/(4.0*$q[2]);
  }
  elsif($maxpos==3){
		$q[3]=sqrt($tmp[3])*0.5;
		$q[0]=($R[3] - $R[1])/(4.0*$q[3]);
		$q[1]=($R[2] + $R[6])/(4.0*$q[3]);
		$q[2]=($R[7] + $R[5])/(4.0*$q[3]);
  }
  else{ # should not happen
		die "Internal error in rotmat2quat\n";
	}

	# enforce unit length
	$mag=$q[0]*$q[0] + $q[1]*$q[1] + $q[2]*$q[2] + $q[3]*$q[3];

	return @q if($mag==1.0);

	$mag=1.0/sqrt($mag);
	$q[0]*=$mag; $q[1]*=$mag; $q[2]*=$mag; $q[3]*=$mag;

	return @q;
}


# compute the rotation matrix corresponding to a quaternion; see Horn's paper
sub quat2rotmat{
  my ($q)=@_;
  my ($R, $mag);

	# ensure unit length
	$mag=$q[0]*$q[0] + $q[1]*$q[1] + $q[2]*$q[2] + $q[3]*$q[3];
	if($mag!=1.0){
		$mag=1.0/sqrt($mag);
		$q[0]*=$mag; $q[1]*=$mag; $q[2]*=$mag; $q[3]*=$mag;
	}

  $R[0]=$q[0]*$q[0]+$q[1]*$q[1]-$q[2]*$q[2]-$q[3]*$q[3];
	$R[1]=2*($q[1]*$q[2]-$q[0]*$q[3]);
	$R[2]=2*($q[1]*$q[3]+$q[0]*$q[2]);

	$R[3]=2*($q[1]*$q[2]+$q[0]*$q[3]);
	$R[4]=$q[0]*$q[0]+$q[2]*$q[2]-$q[1]*$q[1]-$q[3]*$q[3];
	$R[5]=2*($q[2]*$q[3]-$q[0]*$q[1]);

	$R[6]=2*($q[1]*$q[3]-$q[0]*$q[2]);
	$R[7]=2*($q[2]*$q[3]+$q[0]*$q[1]);
	$R[8]=$q[0]*$q[0]+$q[3]*$q[3]-$q[1]*$q[1]-$q[2]*$q[2];

  return @R;
}
