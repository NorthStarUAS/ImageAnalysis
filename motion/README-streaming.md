# A bug? A feature?

I believe there is an error in the streaming dmd algorithm presented
in the StreamingDMD python code here, as well as in the two papers
that present these ideas.

* Multi-Sensor Scene Segmentation for Unmanned Air and Ground Vehicles
  using Dynamic Mode Decomposition Aniketh Kalur∗, Debraj
  Bhattacharjee†, Youbing Wang‡ and Maziar S. Hemati§ University of
  Minnesota, Minneapolis, MN 55455, USA
* Dynamic Mode Decomposition for Large and Streaming Datasets Maziar
  S. Hemati,1, a) Matthew O. Williams,2, b) and Clarence W. Rowley1,
  c) 1)Mechanical and Aerospace Engineering Department, Princeton
  University, NJ 08544, USA.2)Program in Applied and Computational
  Mathematics, Princeton University, NJ 08544, USA.  (Dated: 30 June
  2014)

## Observation #1

When visualizing the modes produced by the SDMD algorithm in real
time, the information in the modes stops 'updating' (no longer shows
substantial changes) after "max rank" (aka r0) is reached.

## Observation #2

When running the SDMD algorithm, the steps before reaching the defined
max rank produce results similar to non-streaming algorithms
(i.e. pydmd.)

## Observation #3

When running the algorithm with a max rank that is >= to the number of
input time steps, the algorithm continues to update properly and
produces a result that is visually similar (if not identical) to the
traditional full DMD solution.

## Inference #1

The SDMD algorithm produces expected (correct) results for the first
r0 (max rank) iterations but produces unexpected (incorrect) results
after max rank is reached.

The only difference in the flow of the SDMD algorithm relative to max
rank is that once max rank is exceeded, then the POD compression step
is introduced.

The algorithm without POD compression appears to work correctly and
produce expected results.

## Example:

Trace the value of **Gx** through the algorithm when max rank = 2

Note: the variables a1, a2, a3, ...  b1, b2, ..., etc. represent
any number.  The main point is to show what portions of the matrices
hold information, and what portion are padded with zeros, or a 1 as
the expansion and POD compression is performed.

### Initialization:

<pre>
  <b>Gx</b> = np.matrix(normx**2)
</pre>

### Step 1: (Gram-Schmidt reorthonormalization)

<pre>
  <b>Gx</b> is untouched
</pre>

### Step 2: (Expansion)

**Gx** (n x n) is exanded to (n+1 x n+1) the new row, col, and last
position of the diagonal are padded with 0's.

```
  [ a1 a2 ] ===> [ a1 a2  0 ]
  [ a3 a4 ]      [ a3 a4  0 ]
                 [  0  0  0 ]
```

### Step 3: (POD compression if needed)

  eigen values of **Gx** (3x3):

```
    [ b1 b2 0 ]
```

  eigen vectors of **Gx** (3x3)

```
    [ c1 c2  0 ]
    [ c3 c4  0 ]
    [  0  0  1 ]
```

  qx is the leading (sorted) r0 vectors of **Gx** (3 x 2):

```
    [ c1 c2 ]
    [ c3 c4 ]
    [  0  0 ]
```

  **Gx** becomes a 2x2 matrix defined as the diaganal of the leading
  eignen values (sorted)

```
    [ b1  0 ]
    [  0 b2 ]
```

  Now trace **Qx** through the algorithm.  In step 2 (expansion) a new
  column is appended to **Qx**.  In step 3 (POD compression) **Qx** =
  **Qx** * **qx**, but notice that the last row of qx is always 0, so
  this newly appended column is immediately truncated before it is
  used for any other math/processing.

  Because we sort the eigen values during the POD compression step,
  the newly appended eigen value (value is always 0) almost always
  sorts last, and almostly always leads to the last row of qx being
  all zeros which almost always leads to the newly appended column of
  **Qx** being immediately truncated.

  The result when running the algorithm on real input is that the
  system ceases updating in response to new input once it has been
  expanded past max rank.

  