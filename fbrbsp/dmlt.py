import numpy as np

def dmlt(a:np.array, b:np.array) -> np.array:
    """
    Calculate the signed change in MLT given MLT arrays from
    spacecraft a and b.
    
    Sign convention: positive dMLT when a is eastward of b.

    Parameters
    ----------
    a:np.array
        An array of MLTs from spacecraft A.
    b:np.array
        An array of MLTs from spacecraft B.

    Returns
    -------
    np.array
        The signed difference in MLT.

    Raises
    ------
    AssertionError
        If `a` or `b` contain negative MLT values, or if their lengths don't match.
    
    Example
    -------
    >>> import numpy as np
    >>> import mlt_difference
    >>> # Spacecraft A is east of B.
    >>> mlt_difference.dmlt(np.array([4,5,6]), np.array([1,2,3]))
    array([3, 3, 3])
    >>> # Spacecraft A is west of A.
    >>> mlt_difference.dmlt(np.array([1,2,3]), np.array([4,5,6]))
    array([-3, -3, -3])
    >>> # A more involved example.
    >>> a = np.arange(0, 24)
    >>> a
    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 21, 22, 23])
    >>> b = 23*np.ones(24)
    >>> b
    array([23., 23., 23., 23., 23., 23., 23., 23., 23., 23., 23., 23., 23.,
        23., 23., 23., 23., 23., 23., 23., 23., 23., 23., 23.])
    >>> diff = mlt_difference.dmlt(np.arange(0, 24), 23*np.ones(24))
    >>> for a_i, b_i, diff_i in zip(a, b, diff):
    ...    print(f'{a_i=} | {b_i=} | {diff_i=}')
    ...
    a_i=0 | b_i=23.0 | diff_i=1.0
    a_i=1 | b_i=23.0 | diff_i=2.0
    a_i=2 | b_i=23.0 | diff_i=3.0
    a_i=3 | b_i=23.0 | diff_i=4.0
    a_i=5 | b_i=23.0 | diff_i=6.0
    a_i=6 | b_i=23.0 | diff_i=7.0
    a_i=7 | b_i=23.0 | diff_i=8.0
    a_i=8 | b_i=23.0 | diff_i=9.0
    a_i=9 | b_i=23.0 | diff_i=10.0
    a_i=10 | b_i=23.0 | diff_i=11.0
    a_i=11 | b_i=23.0 | diff_i=12.0
    a_i=12 | b_i=23.0 | diff_i=-11.0
    a_i=13 | b_i=23.0 | diff_i=-10.0
    a_i=14 | b_i=23.0 | diff_i=-9.0
    a_i=15 | b_i=23.0 | diff_i=-8.0
    a_i=16 | b_i=23.0 | diff_i=-7.0
    a_i=17 | b_i=23.0 | diff_i=-6.0
    a_i=18 | b_i=23.0 | diff_i=-5.0
    a_i=19 | b_i=23.0 | diff_i=-4.0
    a_i=20 | b_i=23.0 | diff_i=-3.0
    a_i=21 | b_i=23.0 | diff_i=-2.0
    a_i=22 | b_i=23.0 | diff_i=-1.0
    a_i=23 | b_i=23.0 | diff_i=0.0
    """
    # Cast as numpy arrays just in case you passed in a list.
    a = np.array(a)
    b = np.array(b)
    assert np.all(a >= 0), 'Array a contains negative MLT values.'
    assert np.all(b >= 0), 'Array b contains negative MLT values.'
    assert a.shape[0] == b.shape[0], (f'Spacecraft a and b MLT arrays must be'
                                      f' the same shape. Got {a.shape[0]=} and {b.shape[0]=}')
    
    _dmlt = a - b  # We are done if |dMLT| < 12
    # if |_dMLT| > 12, then our convention on which is eastward/westward flips. 
    _dmlt[_dmlt >  12] = -24 + _dmlt[_dmlt >  12]
    # = sign to predictably handle cases when dMLT is exactly 12.
    _dmlt[_dmlt <= -12] = 24 + _dmlt[_dmlt <= -12]
    return _dmlt

def test_dmlt():
    """
    Hand picked dMLTs to test. To test call pytest via
    `pytest .\mlt_difference.py`.
    """
    np.testing.assert_array_equal(
        dmlt(np.array([1,2,3]), np.array([4,5,6])),
        np.array([-3,-3,-3])
    )
    np.testing.assert_array_equal(
        dmlt(np.array([4,5,6]), np.array([1,2,3])),
        np.array([3,3,3])
    )
    np.testing.assert_array_equal(
        dmlt(np.array([1,2,3]), np.array([23,23,23])),
        np.array([2,3,4])
    )
    np.testing.assert_array_equal(
        dmlt(np.array([23,23,23]), np.array([1,2,3])),
        np.array([-2,-3,-4])
    )
    np.testing.assert_array_equal(
        dmlt(np.array([18, 3]), np.array([3, 18])),
        np.array([-9, 9])
    )
    np.testing.assert_array_equal(
        dmlt(np.array([15, 16]), np.array([1, 1])),
        np.array([-10, -9])
    )
    np.testing.assert_array_equal(
        dmlt(np.array([0]), np.array([12])),
        np.array([12])
    )
    return