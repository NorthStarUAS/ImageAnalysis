# NAME
#
#    Math::Spline  - Cubic Spline Interpolation of data
#
# SYNOPSIS
#    
#    require Math::Spline;
#    $spline=new Math::Spline(\@x,\@y)
#    $y_interp=$spline->evaluate($x);
#
#    use Math::Spline qw(spline linsearch binsearch);
#    use Math::Derivative qw(Derivative2);
#    @y2=Derivative2(\@x,\@y);
#    $index=binsearch(\@x,$x);
#    $index=linsearch(\@x,$x,$index);
#    $y_interp=spline(\@x,\@y,\@y2,$index,$x);
#
# DESCRIPTION
#
# This package provides cubic spline interpolation of numeric data. The
# data is passed as references to two arrays containing the x and y
# ordinates. It may be used as an exporter of the numerical functions
# or, more easily as a class module.
# 
# The B<Math::Spline> class constructor B<new> takes references to the
# arrays of x and y ordinates of the data. An interpolation is performed
# using the B<evaluate> method, which, when given an x ordinate returns
# the interpolate y ordinate at that value.
# 
# The B<spline> function takes as arguments references to the x and y
# ordinate array, a reference to the 2nd derivatives (calculated using
# B<Derivative2>, the low index of the interval in which to interpolate
# and the x ordinate in that interval. Returned is the interpolated y
# ordinate. Two functions are provided to look up the appropriate index
# in the array of x data. For random calls B<binsearch> can be used -
# give a reference to the x ordinates and the x loopup value it returns
# the low index of the interval in the data in which the value
# lies. Where the lookups are strictly in ascending sequence (e.g. if
# interpolating to produce a higher resolution data set to draw a curve)
# the B<linsearch> function may more efficiently be used. It performs
# like B<binsearch>, but requires a third argument being the previous
# index value, which is incremented if necessary.
# 
# NOTE
# 
# requires Math::Derivative module
#
# EXAMPLE
#
#    require Math::Spline;
#    my @x=(1,3,8,10);
#    my @y=(1,2,3,4);						    
#    $spline=new Math::Spline(\@x,\@y);
#    print $spline->evaluate(5)."\n";
#
# produces the output
#
# 2.44    						   
#
# (Perl version) AUTHOR
#
# John A.R. Williams <J.A.R.Williams@aston.ac.uk>
#
# SEE ALSO
#
# "Numerical Recipies: The Art of Scientific Computing"
# W.H. Press, B.P. Flannery, S.A. Teukolsky, W.T. Vetterling.
# Cambridge University Press. ISBN 0 521 30811 9.

# functions for calculating derivatives of data
#
# Math::Derivative - Numeric 1st and 2nd order differentiation
#
#    use Math::Derivative qw(Derivative1 Derivative2);
#    @dydx=Derivative1(\@x,\@y);
#    @d2ydx2=Derivative2(\@x,\@y);
#    @d2ydx2=Derivative2(\@x,\@y,$yp0,$ypn);
#
# DESCRIPTION
#
# This Perl package exports functions for performing numerical first
# (B<Derivative1>) and second B<Derivative2>) order differentiation on
# vectors of data. They both take references to two arrays containing
# the x and y ordinates of the data and return an array of the 1st or
# 2nd derivative at the given x ordinates. B<Derivative2> may optionally
# be given values to use for the first dervivative at the start and end
# points of the data - otherwiswe 'natural' values are used.
#
# (PERL) AUTHOR
#
# John A.R. Williams <J.A.R.Williams@aston.ac.uk>
#

# PYTHON PORT (combines spline and derivative perl modules into a single 
# spline python module)
#
# Curtis L. Olson <curtolson ata flightgear dota org >
#

def derivative1(points):
    n = len(points)-1 # index of last point
    y2 = list(xrange(n+1))
    y2[0] = (points[1][1]-points[0][1]) / (points[1][0]-points[0][0])
    y2[n] = (points[n][1]-points[n-1][1]) / (points[n][0]-points[n-1][0])
    for i in range(1, n):
	y2[i]=(points[i+1][1]-points[i-1][1]) / (points[i+1][0]-points[i-1][0])
    return y2


def derivative2(points, yp1 = "", ypn = ""):
    n = len(points)-1 # index of last point
    y2 = list(xrange(n+1))
    u = list(xrange(n+1))
    if yp1 == "":
	y2[0] = 0
        u[0] = 0
    else:
	y2[0] = -0.5
	u[0] = (3/(points[1][0]-points[0][0]))*((points[1][1]-points[0][1])/(points[1][0]-points[0][0])-float(yp1))
    for i in range(1, n):
	sig = (points[i][0]-points[i-1][0])/(points[i+1][0]-points[i-1][0])
	p = sig * y2[i-1] + 2.0
	y2[i] = (sig-1.0) / p
	u[i] = (6.0*( (points[i+1][1]-points[i][1])/(points[i+1][0]-points[i][0])-(points[i][1]-points[i-1][1])/(points[i][0]-points[i-1][0]))/(points[i+1][0]-points[i-1][0])-sig*u[i-1])/p;

    if ypn == "":
	qn = 0
	un = 0
    else:
	qn = 0.5
	un = (3.0/(points[n][0]-points[n-1][0]))*(float(ypn)-(points[n][1]-points[n-1][1])/(points[n][0]-points[n-1][0]))
    y2[n] = (un-qn*u[n-1])/(qn*y2[n-1]+1.0)
    for i in range(n-1, -1, -1):
	y2[i] = y2[i]*y2[i+1]+u[i]

    return y2

def spline(points, y2, i, v):
    klo = i
    khi = i + 1
    h = points[khi][0] - points[klo][0]
    if h == 0:
        print "Zero interval in spline data."
        return 0;
    a = (points[khi][0] - v) / h
    b = (v - points[klo][0]) / h
    return a*points[klo][1] + b*points[khi][1]+((a*a*a-a)*y2[klo]+(b*b*b-b)*y2[khi])*(h*h)/6.0

# binary search routine finds index just below value
def binsearch(points, v):
    klo = 0
    khi = len(points)-1
    while (khi - klo) > 1:
        k = int((khi+klo)/2)
        if (points[k][0] > v):
            khi = k
        else:
            klo = k
    return klo

# more efficient if repetatively doint it
def linsearch(points, v, khi):
    khi += 1
    n = len(points) - 1
    while v > points[khi][0] and khi < n:
        khi += 1
    return khi - 1

