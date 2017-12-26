#!/usr/bin/perl -w

#################################################################################
## 
##  Perl script for computing the reprojection error corresponding to a
##  given reconstruction. Currently, projective and quaternion-based Euclidean
##  reconstructions are supported. More reconstruction types can be added by
##  supplying appropriate camera matrix generation routines (i.e. CamMat_Generate)
##  Copyright (C) 2005  Manolis Lourakis (lourakis at ics.forth.gr)
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

use lib '/home/lourakis/sba-src/sba-1.6/utils'; # change this!
use SBAio;

#################################################################################
# Initializations

our ($usage, $help);

$usage="Usage is $0 -e|-i|-I|-p [-r,-s,-P,-h] <cams file> <pts file> [<calib file>]";
$help="-e specifies a Euclidean reconstruction with fixed intrinsics,\n-i a Euclidean reconstruction with varying intrinsics,\n-I a Euclidean reconstruction with varying intrinsics and lens distortion and -p a projective one.\n"
."-s computes the average *squared* reprojection error; -P prints camera matrices.";
use constant EUCBA   => 0; # Euclidean BA, fixed intrinsics
use constant EUCIBA  => 1; # Euclidean BA, varying intrinsics
use constant EUCIDBA => 2; # Euclidean BA, varying intrinsics & distortion
use constant PROJBA  => 3; # Projective BA
$cnp=$pnp=0;
$camsfile=$ptsfile=$calfile="";
$CamMat_Generate=\&dont_know;

#################################################################################
# Basic arguments parsing

use Getopt::Std;
getopts("eiIrpsPh", \%opt) or die "$usage\n";
die "$0 help: Compute the average reprojection error for some reconstruction.\n$usage\n$help\n" if($opt{'h'});

$i=(defined($opt{'e'})? 1 : 0) + (defined($opt{'i'})? 1 : 0) +
   (defined($opt{'I'})? 1 : 0) + (defined($opt{'p'})? 1 : 0);
if($i>1){
    die "$0: Only one of -e, -i, -I, -p can be specified!\n";
} elsif($i==0){
    die "$0: One of -e, -i, -I, -p should be specified!\n";
} elsif($opt{'e'}){
    $batype=EUCBA;
} elsif($opt{'i'}){
    $batype=EUCIBA;
} elsif($opt{'I'}){
    $batype=EUCIDBA;
} elsif($opt{'p'}){
    $batype=PROJBA;
}
$squared=$opt{'s'}? 1 : 0;
$printPs=$opt{'P'}? 1 : 0;
$reverse=$opt{'r'}? 1 : 0; # reverse motion: use [R'|-R'*t] instead of [R|t] for the camera matrices
die "$0: -r is meaningful only in combination with one of -e, -i, -I!\n" if($batype!=EUCBA && $batype!=EUCIBA && $batype!=EUCIDBA);

#################################################################################
# Initializations depending on reconstruction type
if($batype==EUCBA){
    $cnp=7; $pnp=3;
    die "$0: Cameras, points, or calibration file is missing!\n$usage" if(@ARGV<3);
    die "$0: Too many arguments!\n$usage" if(@ARGV>3);
    $camsfile=$ARGV[0];
    $ptsfile=$ARGV[1];
    $calfile=$ARGV[2];
    $CamMat_Generate=($reverse==0)? \&PfromRtK : \&PfromRtRevK;
}
elsif($batype==EUCIBA){
    $cnp=7+5; $pnp=3;
    die "$0: Cameras or points file is missing!\n$usage" if(@ARGV<2);
    die "$0: Too many arguments!\n$usage" if(@ARGV>2);
    $camsfile=$ARGV[0];
    $ptsfile=$ARGV[1];
    $CamMat_Generate=($reverse==0)? \&PfromRtVarK : \&PfromRtRevVarK;
}
elsif($batype==EUCIDBA){
    $cnp=7+5+5; $pnp=3;
    die "$0: Cameras or points file is missing!\n$usage" if(@ARGV<2);
    die "$0: Too many arguments!\n$usage" if(@ARGV>2);
    $camsfile=$ARGV[0];
    $ptsfile=$ARGV[1];
    $CamMat_Generate=($reverse==0)? \&PfromRtVarKD : \&PfromRtRevVarKD;
}
elsif($batype==PROJBA){ 
    $cnp=12; $pnp=4;
    die "$0: Cameras or points file is missing!\n$usage" if(@ARGV<2);
    die "$0: Too many arguments!\n$usage" if(@ARGV>2);
    $camsfile=$ARGV[0];
    $ptsfile=$ARGV[1];
    $CamMat_Generate=\&nop;
}
else{
    die "Unknown BA type \"$batype\" specified!\n";
}

die "$0: Do not know how to handle $pnp parameters per point!\n" if($pnp!=3 && $pnp!=4);


#################################################################################
# Main code for computing the reprojection error.

@camPoses=(); # array of arrays storing each camera's pose;
              # Note that in the presence of distortion, camera matrices do not include the intrinsics K!

$camCal=();    # 3x3 array for storing the camera intrinsic calibration (only when identical for all cameras)

@camCalibs=(); # array of arrays storing each camera's intrinsics; only used when $batype==EUCIDBA
@camDistorts=(); # array of arrays storing each camera's distortion parameters; only used when $batype==EUCIDBA


# read calibration file, if there is one
  if(length($calfile)>0){
    $camCal=SBAio::readCalib($calfile);
  }

# read cameras file
  $camParms=SBAio::readCameras($camsfile, $cnp);
  for($i=0; $i<scalar(@$camParms); $i++){
    if($batype!=EUCIDBA){
      push @camPoses, $CamMat_Generate->($i, $camParms->[$i], $camCal);
    }
    else{
      ($camPoses[$i], $camCalibs[$i], $camDistorts[$i])=$CamMat_Generate->($i, $camParms->[$i]);
    }
  }
  @$camParms=(); # not needed anymore

  printf "Read %d cameras\n", scalar(@camPoses);

  if($printPs){ # NOTE: K not included in P's in the presence of distortion!
    for($i=0; $i<scalar(@camPoses); $i++){
      printf "%g %g %g %g\n", $camPoses[$i]->[0], $camPoses[$i]->[1], $camPoses[$i]->[2], $camPoses[$i]->[3];
      printf "%g %g %g %g\n", $camPoses[$i]->[4], $camPoses[$i]->[5], $camPoses[$i]->[6], $camPoses[$i]->[7];
      printf "%g %g %g %g\n\n", $camPoses[$i]->[8], $camPoses[$i]->[9], $camPoses[$i]->[10], $camPoses[$i]->[11];
    }
  }

# read points file
  ($threeDpts, $twoDtrajs, $trajsFrames, $totframes)=SBAio::readPoints($ptsfile, $pnp);

  printf "Read %d 3D points \& trajectories, projecting onto %d image points\n", scalar(@$threeDpts), $totframes;

# Data file has now been read. Following fragment shows how it can be printed
if(0){
  for($i=0; $i<scalar(@$threeDpts); $i++){
    for($j=0; $j<$pnp; $j++){
      printf "%.6g ", $threeDpts->[$i][$j];
    }

    printf "%d ", $trajsFrames->[$i][0];
    for($j=0; $j<$trajsFrames->[$i][0]; $j++){
      $fr=$trajsFrames->[$i][$j+1];
      if(defined($twoDtrajs->[$i]{$fr})){
        printf "%d %.6g %.6g ", $fr, $twoDtrajs->[$i]{$fr}[0], $twoDtrajs->[$i]{$fr}[1];
      }
    }
    print "\n";
  }
}

# compute average, min & max trajectory lengths
  $avlen=0; $maxlen=-1; $minlen=99999999;
  for($i=0; $i<scalar(@$threeDpts); $i++){
    $avlen+=$trajsFrames->[$i][0];
    $maxlen=$trajsFrames->[$i][0] if($trajsFrames->[$i][0]>$maxlen);
    $minlen=$trajsFrames->[$i][0] if($trajsFrames->[$i][0]<$minlen);
  }
  $avlen/=scalar(@$threeDpts);
  printf "Average trajectory length is %g frames [min %d, max %d], S density %.2f%% \n\n", $avlen, $minlen, $maxlen,
              density(scalar(@camPoses), scalar(@$threeDpts), $trajsFrames, $twoDtrajs)*100.0;

# compute reprojection error
  unless ($batype==EUCBA || $batype==EUCIBA || $batype==EUCIDBA || $batype==PROJBA){
    die "current implementations of reprError() cannot handle supplied reconstruction data!\n";
  }

  $toterr=0.0;
  $totsqerr=0.0;
  $totprojs=0;
  @error=();
  @sqerror=();
  for($fr=0; $fr<scalar(@camPoses); $fr++){
    $error[$fr]=0.0;
    $sqerror[$fr]=0.0;
    for($i=$j=0; $i<scalar(@$threeDpts); $i++){
      if(defined($twoDtrajs->[$i]{$fr})){
        if($batype!=EUCIDBA){
          $theerr=&reprErrorNoDistortion($twoDtrajs->[$i]{$fr}, $camPoses[$fr], $threeDpts->[$i], $pnp);
        } else{
          $theerr=&reprErrorWithDistortion($twoDtrajs->[$i]{$fr}, $camPoses[$fr], $threeDpts->[$i], $camCalibs[$fr], $camDistorts[$fr]);
        }
        $theerr=sqrt($theerr) if(!$squared);
        $error[$fr]+=$theerr;
        $sqerror[$fr]+=$theerr*$theerr;
#        printf "@@@ point %d, camera %d: %g\n", $i, $fr, $theerr;
        $j++;
      }
    }
    if($j){
      $mean=$error[$fr]/$j;
      $sdev=sqrt($sqerror[$fr]/$j - $mean*$mean);
      printf "Mean %serror for camera %d [%d projections] is %g, stdev %g\n", $squared? "squared " :"", $fr, $j, $mean, $sdev;
      $toterr+=$error[$fr];
      $totsqerr+=$error[$fr]*$error[$fr];
      $totprojs+=$j;
    } else{
      printf "No projections for camera %d!\n", $fr;
    }
  }

  printf STDERR "\nWarning: total number of image projections does not agree with that read with trajectories! [%d != %d]\n", $totprojs, $totframes if($totframes!=$totprojs);

  $mean=$toterr/$totprojs;
  printf "\nMean %serror for the whole sequence [%d projections] is %g, stdev %g\n", $squared? "squared " :"", $totprojs, $mean, sqrt($totsqerr/$totprojs-$mean*$mean) if($totprojs);



#################################################################################
# Misc routines

# measure the density of the points submatrix S
sub density{
  my ($ncams, $npts, $trjfrms, $trjs)=@_;
  my ($i, $i2, $i3, $j, $k, @S, $nnz);

  @S=(); # array of hashes keeping track of "connected" frames
  for($j=0; $j<$ncams; $j++){
    $S[$j]={};
  }

  for($i=0; $i<$npts; $i++){
    for($i2=1; $i2<=$trjfrms->[$i][0]; $i2++){
      $j=$trjfrms->[$i][$i2];
      for($i3=$i2+1; $i3<=$trjfrms->[$i][0]; $i3++){
        $k=$trjfrms->[$i][$i3];
        $S[$j]{$k}=1;
      }
    }
  }

  for($j=$nnz=0; $j<$ncams; $j++){
    for($k=$j+1; $k<$ncams; $k++){
      $nnz+=2 if(defined($S[$j]{$k}) || defined($S[$k]{$j})); # S is symmetric, both Sjk and Skj are counted
    }
  }
  $nnz+=$ncams; # add diagonal elements (not counted above)

  return $nnz/($ncams*$ncams);
}

#################################################################################
# Reprojection error calculation routines

# compute the SQUARED reprojection error |x-xx|^2 with xx=P*X when no distortion is present
sub reprErrorNoDistortion{
  my ($x, $P, $X, $pnp)=@_;

  # error checking
  unless (@_==4 && ref($x) eq 'ARRAY' && ref($P) eq 'ARRAY' && ref($X) eq 'ARRAY'){
    die "usage: reprErrorNoDistortion ARRAYREF1 ARRAYREF2 ARRAYREF3 pnp";
  }

  my @xx=();
  my $k;

  # compute the projection in xx
  for($k=0; $k<3; $k++){
    $xx[$k]=$P->[$k*4]*$X->[0] + $P->[$k*4+1]*$X->[1] + $P->[$k*4+2]*$X->[2] + $P->[$k*4+3]*(($pnp==4)? $X->[3] : 1.0);
  }
  $xx[0]/=$xx[2];
  $xx[1]/=$xx[2];

# printf "[%g %g -- %g %g]\n", $x->[0], $x->[1], $xx[0], $xx[1];

  return ($x->[0]-$xx[0])*($x->[0]-$xx[0]) + ($x->[1]-$xx[1])*($x->[1]-$xx[1]);
}

# compute the SQUARED reprojection error |x-xx|^2 in the presence of distortion
sub reprErrorWithDistortion{
  my ($x, $P, $X, $K, $kc)=@_;

  # error checking
  unless (@_==5 && ref($x) eq 'ARRAY' && ref($P) eq 'ARRAY' && ref($X) eq 'ARRAY' && ref($K) eq 'ARRAY' && ref($kc) eq 'ARRAY'){
    die "usage: reprErrorWithDistortion ARRAYREF1 ARRAYREF2 ARRAYREF3 ARRAYREF4 ARRAYREF5";
  }

  my @xx=();
  my @yy=();
  my @dxx=();
  my ($k, $s);

  # compute the projection in xx, P is assumed not to include calibration K!
  for($k=0; $k<3; $k++){
    $xx[$k]=$P->[$k*4]*$X->[0] + $P->[$k*4+1]*$X->[1] + $P->[$k*4+2]*$X->[2] + $P->[$k*4+3];
  }
  $xx[0]/=$xx[2];
  $xx[1]/=$xx[2];

  # distort xx into yy
  $rsq=$xx[0]*$xx[0] + $xx[1]*$xx[1];
  $rad=1 + $rsq*($kc->[0] + $rsq*($kc->[1] + $rsq*$kc->[4]));

  # radial distortion
  $yy[0]=$rad*$xx[0];
  $yy[1]=$rad*$xx[1];

  # tangential component
  $yy[0]+=2*$kc->[2]*$xx[0]*$xx[1] + $kc->[3]*($rsq+2*$xx[0]*$xx[0]);
  $yy[1]+=$kc->[2]*($rsq+2*$xx[1]*$xx[1]) + 2*$kc->[3]*$xx[0]*$xx[1];

  # apply K
  $s=1.0/($K->[6]*$yy[0] + $K->[7]*$yy[1] + $K->[8]);
  $dxx[0]=($K->[0]*$yy[0] + $K->[1]*$yy[1] + $K->[2])*$s;
  $dxx[1]=($K->[3]*$yy[0] + $K->[4]*$yy[1] + $K->[5])*$s;

# printf "[%g %g -- %g %g]\n", $x->[0], $x->[1], $dxx[0], $dxx[1];

  return ($x->[0]-$dxx[0])*($x->[0]-$dxx[0]) + ($x->[1]-$dxx[1])*($x->[1]-$dxx[1]);
}


#################################################################################
# Camera matrix generation routines

sub dont_know {
  my ($camid, $camparms)=@_;

  die "Don't know how to generate a projection matrix for camera $camid from the supplied camera parameters!\n";
  return $camparms;
}

# Return as is
sub nop {
  my ($camid, $camparms)=@_;

  return $camparms;
}

# Compute P as K[R|t]. R is specified by the first 4 elements of $camparms, while t corresponds to the last 3 ones
sub PfromRtK {
  my ($camid, $camparms, $calparms)=@_;

  my ($x, $y, $z, $w, $xx, $xy, $xz, $xw, $yy, $yz, $yw, $zz, $zw, $ww, $i, $j, $k);
  my (@R, @P);
  my $mag;

  @R=(); @P=(); # 3x3 & 3x4 resp.
# compute the rotation matrix for q=(x, y, z, w);
# see also http://www.gamedev.net/reference/articles/article1095.asp (but note that q=(w, x, y, z) there!)

  $x=$camparms->[0]; $y=$camparms->[1];
  $z=$camparms->[2]; $w=$camparms->[3];

  # normalize quaternion
  $mag=1.0/sqrt($x*$x + $y*$y + $z*$z + $w*$w);
  $x*=$mag; $y*=$mag; $z*=$mag; $w*=$mag;

  $xx=$x*$x; $xy=$x*$y; $xz=$x*$z; $xw=$x*$w;
  $yy=$y*$y; $yz=$y*$z; $yw=$y*$w;
  $zz=$z*$z; $zw=$z*$w; $ww=$w*$w;
  $R[0]=$xx+$yy - ($zz+$ww); $R[1]=2.0*($yz-$xw);       $R[2]=2.0*($yw+$xz);
  $R[3]=2.0*($yz+$xw);       $R[4]=$xx+$zz - ($yy+$ww); $R[5]=2.0*($zw-$xy);
  $R[6]=2.0*($yw-$xz);       $R[7]=2.0*($zw+$xy);       $R[8]=$xx+$ww - ($yy+$zz);

#print "@R\n\n";
# compute the matrix-matrix & matrix-vector products
  for($i=0; $i<3; $i++){
    for($j=0; $j<3; $j++){
      for($k=0, $sum=0.0; $k<3; $k++){
        $sum+=$calparms->[$i*3+$k]*$R[$k*3+$j];
      }
      $P[$i*4+$j]=$sum;
    }
    for($j=0, $sum=0.0; $j<3; $j++){
      $sum+=$calparms->[$i*3+$j]*$camparms->[4+$j];
    }
    $P[$i*4+3]=$sum;
  }

  return [@P];
}

# As above but compute P as K[R'|-R'*t]
sub PfromRtRevK{
  my ($camid, $camparms, $calparms)=@_;
  my ($mag, $tmp, $q, $t);

  # normalize quaternion
  $mag=1.0/sqrt($camparms->[0]*$camparms->[0] + $camparms->[1]*$camparms->[1] +
                $camparms->[2]*$camparms->[2] + $camparms->[3]*$camparms->[3]);
  $camparms->[0]*=$mag; $camparms->[1]*=$mag; $camparms->[2]*=$mag; $camparms->[3]*=$mag;

  $camparms->[0]=-$camparms->[0]; # opposite angle

  # compute -R'*t using the quaternion: -q*(0, t)*qc, qc is q's conjugate
  # note that the quat multiplications below are adapted to using q & t and are not general-purpose!
  $q[0]=$camparms->[0]; $q[1]=$camparms->[1]; $q[2]=$camparms->[2]; $q[3]=$camparms->[3];
  $t[0]=$camparms->[4]; $t[1]=$camparms->[5]; $t[2]=$camparms->[6];

  # compute tmp as -q*t
  $tmp[0]=-(          - $q[1]*$t[0] - $q[2]*$t[1] - $q[3]*$t[2]);
  $tmp[1]=-($q[0]*$t[0]             + $q[2]*$t[2] - $q[3]*$t[1]);
  $tmp[2]=-($q[0]*$t[1]             + $q[3]*$t[0] - $q[1]*$t[2]);
  $tmp[3]=-($q[0]*$t[2]             + $q[1]*$t[1] - $q[2]*$t[0]);

  # compute tmp*qc
  #always zero:   $tmp[0]*$q[0] + $tmp[1]*$q[1] + $tmp[2]*$q[2] + $tmp[3]*$q[3];
  $camparms->[4]=-$tmp[0]*$q[1] + $tmp[1]*$q[0] - $tmp[2]*$q[3] + $tmp[3]*$q[2];
  $camparms->[5]=-$tmp[0]*$q[2] + $tmp[2]*$q[0] - $tmp[3]*$q[1] + $tmp[1]*$q[3];
  $camparms->[6]=-$tmp[0]*$q[3] + $tmp[3]*$q[0] - $tmp[1]*$q[2] + $tmp[2]*$q[1];
printf "%g %g %g %g %g %g %g\n", $camparms->[0], $camparms->[1], $camparms->[2], $camparms->[3], $camparms->[4], $camparms->[5], $camparms->[6];

  return &PfromRtK($camid, $camparms, $calparms);
}

# Compute P as K[R|t]. K is specified by the first 5 elements of $camparms, while R and t correspond to the next 4 & 3 elements, respectively
sub PfromRtVarK {
  my ($camid, $camparms)=@_;
  my (@K, @poseparms, $size, $i);

  @K=(); @poseparms=();
  # setup the intrinsics matrix from the 5 first elements
  $K[0]=$camparms->[0]; $K[1]=$camparms->[4];                $K[2]=$camparms->[1];
  $K[3]=0.0;            $K[4]=$camparms->[3]*$camparms->[0]; $K[5]=$camparms->[2];
  $K[6]=0.0;            $K[7]=0.0;                           $K[8]=1.0;

  $size=scalar(@$camparms);
  for($i=5; $i<$size; $i++){
    $poseparms[$i-5]=$camparms->[$i];
  }

  return &PfromRtK($camid, [@poseparms], [@K]);
}

# As above but compute P as K[R'|-R'*t]
sub PfromRtRevVarK{
  my ($camid, $camparms)=@_;
  my (@K, @poseparms, $size, $i);

  @K=(); @poseparms=();
  # setup the intrinsics matrix from the 5 first elements
  $K[0]=$camparms->[0]; $K[1]=$camparms->[4];                $K[2]=$camparms->[1];
  $K[3]=0.0;            $K[4]=$camparms->[3]*$camparms->[0]; $K[5]=$camparms->[2];
  $K[6]=0.0;            $K[7]=0.0;                           $K[8]=1.0;

  $size=scalar(@$camparms);
  for($i=5; $i<$size; $i++){
    $poseparms[$i-5]=$camparms->[$i];
  }

  return &PfromRtRevK($camid, [@poseparms], [@K]);
}

# Compute P as [R|t]. K is specified by the first 5 elements of $camparms, distortion from the next 5.
# Parameters for R and t correspond to the next 4 & 3 elements, respectively
sub PfromRtVarKD {
  my ($camid, $camparms)=@_;
  my (@K, @I3, @poseparms, @distparms, $size, $i);

  @K=(); @I3=(); @poseparms=(); @distparms=(); @P=();
  # setup the intrinsics matrix from the 5 first elements
  $K[0]=$camparms->[0]; $K[1]=$camparms->[4];                $K[2]=$camparms->[1];
  $K[3]=0.0;            $K[4]=$camparms->[3]*$camparms->[0]; $K[5]=$camparms->[2];
  $K[6]=0.0;            $K[7]=0.0;                           $K[8]=1.0;

  for($i=5; $i<10; $i++){
    $distparms[$i-5]=$camparms->[$i];
  }
  $size=scalar(@$camparms);
  for($i=10; $i<$size; $i++){
    $poseparms[$i-10]=$camparms->[$i];
  }

  $I3[0]=1.0; $I3[1]=0.0; $I3[2]=0.0;
  $I3[3]=0.0; $I3[4]=1.0; $I3[5]=0.0;
  $I3[6]=0.0; $I3[7]=0.0; $I3[8]=1.0;

  $P=&PfromRtK($camid, [@poseparms], [@I3]);

  return ($P, [@K], [@distparms]);
}

# As above but compute P as K[R'|-R'*t]
sub PfromRtRevVarKD {
  my ($camid, $camparms)=@_;
  my (@K, @I3, @poseparms, @distparms, $size, $i, @P);

  @K=(); @I3=(); @poseparms=(); @distparms=(); @P=();
  # setup the intrinsics matrix from the 5 first elements
  $K[0]=$camparms->[0]; $K[1]=$camparms->[4];                $K[2]=$camparms->[1];
  $K[3]=0.0;            $K[4]=$camparms->[3]*$camparms->[0]; $K[5]=$camparms->[2];
  $K[6]=0.0;            $K[7]=0.0;                           $K[8]=1.0;

  for($i=5; $i<10; $i++){
    $distparms[$i-5]=$camparms->[$i];
  }
  $size=scalar(@$camparms);
  for($i=10; $i<$size; $i++){
    $poseparms[$i-10]=$camparms->[$i];
  }

  $I3[0]=1.0; $I3[1]=0.0; $I3[2]=0.0;
  $I3[3]=0.0; $I3[4]=1.0; $I3[5]=0.0;
  $I3[6]=0.0; $I3[7]=0.0; $I3[8]=1.0;

  $P=&PfromRtRevK($camid, [@poseparms], [@I3]);

  return ($P, [@K], [@distparms]);
}
