import ctypes

# Helper classes to be used throughout the project

class UniformArray:
    """Convenience class to create a uniform array.

    Ensure that the order matches structs in the shader code.
    See https://www.w3.org/TR/WGSL/#alignment-and-size for reference on alignment.
    """

    def __init__(self, *args):
        """
        *args is an iterable with (name, format, dim)
        """
        # Analyse incoming fields
        fields = []
        byte_offset = 0
        for name, format, n in args:
            assert format in ("f", "i", "I")
            field = name, format, byte_offset, byte_offset + n * 4
            fields.append(field)
            byte_offset += n * 4
        # Get padding
        nbytes = byte_offset
        while nbytes % 16:
            nbytes += 1
        # Construct memoryview object and a view for each field
        self._mem = memoryview((ctypes.c_uint8 * nbytes)()).cast("B")
        self._views = {}
        for name, format, i1, i2 in fields:
            self._views[name] = self._mem[i1:i2].cast(format)

    @property
    def mem(self):
        return self._mem

    @property
    def nbytes(self):
        return self._mem.nbytes

    def __getitem__(self, key):
        v = self._views[key].tolist()
        return v[0] if len(v) == 1 else v

    def __setitem__(self, key, val):
        m = self._views[key]
        n = m.shape[0]
        if n == 1:
            assert isinstance(val, (float, int))
            m[0] = val
        else:
            assert isinstance(val, (tuple, list))
            for i in range(n):
                m[i] = val[i]
