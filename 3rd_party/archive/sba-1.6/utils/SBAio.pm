#################################################################################
## 
##  Perl script for reading reconstructions in SBA's file format
##  Copyright (C) 2005-9  Manolis Lourakis (lourakis at ics.forth.gr)
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

package SBAio;

use strict;

# NOTE: all 2D arrays below are stored in row-major order as vectors

# read calibration file containing 3x3 array
sub readCalib{
  my ($calfile)=@_;
  my ($i, $line, @columns, @camCal);

  if(length($calfile)>0){
    if(not open(CAL, $calfile)){
	    print STDERR "cannot open file $calfile: $!\n";
	    exit(1);
    }
    @camCal=();
    for($i=0; $i<3; ){ # $i gets incremented at the bottom of the loop
      $line=<CAL>;
      if($line=~/\r\n$/){ # CR+LF
        chop($line); chop($line);
      }
      else{
        chomp($line);
      }

      next if($line=~/^#.+/); # skip comments

      @columns=split(" ", $line);
      die "line \"$line\" in $calfile does not contain exactly 3 numbers [$#columns+1]!\n" if($#columns+1!=3);
      $camCal[$i*3]=$columns[0]; $camCal[$i*3+1]=$columns[1]; $camCal[$i*3+2]=$columns[2];
      $i++;
    }
    close(CAL);
  }

  return [@camCal];
}

# read cameras file
sub readCameras{
  my ($camsfile, $cnp)=@_;
  my($i, $line, $ncams, @columns, @pose, @camParams);

  if(not open(CAMS, $camsfile)){
	  print STDERR "cannot open file $camsfile: $!\n";
	  exit(1);
  }
  @camParams=(); # array of arrays storing each camera's parameters
  $ncams=0;
  while($line=<CAMS>){
    if($line=~/\r\n$/){ # CR+LF
      chop($line); chop($line);
    }
    else{
      chomp($line);
    }

    next if($line=~/^#.+/); # skip comments

    @columns=split(" ", $line);
    #next if($#columns==-1); # skip empty lines

    die "line \"$line\" in $camsfile does not contain exactly $cnp numbers [$#columns+1]!\n" if($cnp!=$#columns+1);
    @pose=();
    for($i=0; $i<$cnp; $i++){
      $pose[$i]=$columns[$i];
    }
    $camParams[$ncams++]=[@pose];
  }
  close(CAMS);

  return [@camParams];
}

# read points file
sub readPoints{
  my($ptsfile, $pnp)=@_;
  my($i, $j, $line, $npts, $totprojs, @columns, $nframes, $fr, %traj, @recpt, @theframes, @ptParams, @imgTrajs, @trajFrames);

  if(not open(PTS, $ptsfile)){
	  print STDERR "cannot open file $ptsfile: $!\n";
	  exit(1);
  }

  @ptParams=(); # array of arrays storing the reconstructed 3D points; each element is of size $pnp
  @imgTrajs=(); # array of hashes storing the 2D trajectory correponding to reconstructed 3D points.
                # The hash key is the frame number
  @trajFrames=(); # array of arrays storing the frame numbers corresponding to each trajectory.
                  # The first number is the total number of frames, then follow the individual frame
  $npts=0;
  $totprojs=0;
  while($line=<PTS>){
    if($line=~/\r\n$/){ # CR+LF
      chop($line); chop($line);
    }
    else{
      chomp($line);
    }

    next if($line=~/^#.+/); # skip comments
    @columns=split(" ", $line);

    die "line \"$line\" in $ptsfile contains less than $pnp numbers [$#columns+1]!\n" if($pnp>$#columns+1);
    @recpt=();
    for($i=0; $i<$pnp; $i++){
      $recpt[$i]=$columns[$i];
    }

    $nframes=$columns[$pnp];
    $i=$pnp+1+$nframes*3; # 3 numbers per image projection: (i.e. imgid, x, y)
    if($i!=$#columns+1){
      die "line \"$line\" in $ptsfile does not contain exactly the $i numbers required for a 3D point with $nframes 2D projections [$#columns+1]!\n";
    }

    printf STDERR "Warning: point %d (%s) has no image projections!\n", $npts, $line, if($nframes<=0);

    %traj=();
    @theframes=($nframes);
    $totprojs=$totprojs+$nframes;
    for($i=0, $j=$pnp+1; $i<$nframes; $i++, $j+=3){
      $fr=$columns[$j];
#      if($fr>0){
#        $fr =~ s/^0+//; # drop any leading zeros
#      }
#      else{
#        $fr=0; # avoid 00, 000, etc...
#      }
      $fr =~ s/^0+//; # remove any leading zeros
      $fr =~ s/^$/0/; # put one back if they're all zero

      $traj{$fr}=[$columns[$j+1], $columns[$j+2]];
      push @theframes, $fr;

#     printf "%d: %d %.4g %.4g\n", $j, $fr, $columns[$j+1], $columns[$j+2];
    }
    $ptParams[$npts]=[@recpt];
    $imgTrajs[$npts]={%traj};
    $trajFrames[$npts++]=[@theframes];
  }
  close(PTS);

  return ([@ptParams], [@imgTrajs], [@trajFrames], $totprojs);
}


# subroutine showing how the read reconstruction can be printed
sub printRecons{
  my($camParams, $cnp, $ptParams, $imgTrajs, $trajFrames, $pnp)=@_;
  my($i, $j, $fr);

  for($i=0; $i<scalar(@$camParams); $i++){
    for($j=0; $j<$cnp; $j++){
      printf "%.8e ", $camParams->[$i][$j];
    }
    print "\n";
  }

  print "\n\n";

  for($i=0; $i<scalar(@$ptParams); $i++){
    for($j=0; $j<$pnp; $j++){
      printf "%.8e ", $ptParams->[$i][$j];
    }

    printf "%d ", $trajFrames->[$i][0];
    for($j=0; $j<$trajFrames->[$i][0]; $j++){
      $fr=$trajFrames->[$i][$j+1];

      # sanity check
      #die "internal inconsistency for point %d in printRecons()!\n", $i if(!defined($imgTrajs->[$i]{$fr}));

      printf "%d %.4e %.4e ", $fr, $imgTrajs->[$i]{$fr}[0], $imgTrajs->[$i]{$fr}[1];
    }
    print "\n";
  }
}

# find the matches between a pair of images. returns a 2D array whose each line
# is reftoproj1, reftoproj2, ptID
#
# example of use:
#
#$pairs=SBAio::getMatchPairs($imgtrajs, 0, 2);
#for($i=0; $i<scalar(@$pairs); $i++){
# printf "%.4e %.4e  %.4e %.4e  [%d]\n", $pairs->[$i][0]->[0], $pairs->[$i][0]->[1], $pairs->[$i][1]->[0], $pairs->[$i][1]->[1], $pairs->[$i][2];
#}
#
sub getMatchPairs{
  my ($imgTrajs, $fr1, $fr2)=@_;
  my ($i, @matches);

  @matches=();
  for($i=0; $i<scalar(@$imgTrajs); $i++){
    if(defined($imgTrajs->[$i]{$fr1}) && defined($imgTrajs->[$i]{$fr2})){
      push @matches, [$imgTrajs->[$i]{$fr1}, $imgTrajs->[$i]{$fr2}, $i];
    }
  }

  return [@matches];
}

# similar to the above for a triplet of images. returns a 2D array whose each line
# is reftoproj1, reftoproj2, reftoproj3, ptID
sub getMatchTriplets{
  my ($imgTrajs, $fr1, $fr2, $fr3)=@_;
  my ($i, @matches);

  @matches=();
  for($i=0; $i<scalar(@$imgTrajs); $i++){
    if(defined($imgTrajs->[$i]{$fr1}) && defined($imgTrajs->[$i]{$fr2}) && defined($imgTrajs->[$i]{$fr3})){
      push @matches, [$imgTrajs->[$i]{$fr1}, $imgTrajs->[$i]{$fr2}, $imgTrajs->[$i]{$fr3}, $i];
    }
  }

  return [@matches];
}


1; # return true
