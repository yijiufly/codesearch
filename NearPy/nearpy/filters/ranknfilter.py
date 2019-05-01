# -*- coding: utf-8 -*-

# Copyright (c) 2013 Ole Krause-Sparmann

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from nearpy.filters.vectorfilter import VectorFilter


class RankNFilter(VectorFilter):
    """
    Sorts vectors with respect to distance and returns the rank N.
    """

    def __init__(self, N):
        """
        Keeps the count threshold.
        """
        self.N = N

    def filter_vectors(self, input_list):
        """
        Returns subset of specified input list.
        """
        try:
            # Return filtered (vector, data, distance )tuple list. Will fail
            # if input is list of (vector, data) tuples.
            sorted_list = sorted(input_list, key=lambda x: x[2])
            index = 0
            val = round(sorted_list[0][2], 9)
            N_index = 1
            for i in sorted_list:
                val_now = round(i[2], 9)
                if val_now > val:
                    N_index += 1
                    if N_index > self.N:
                        break
                    else:
                        val = val_now
                        index += 1
                else:
                    index += 1
            return sorted_list[:index]
        except:
            # Otherwise just return input list
            return input_list
