import numpy

class bcf_store_memory():
    def __init__(self, filename):
        self._filename = filename
        print 'Loading BCF file to memory ... '+filename
        file = open(filename, 'rb')
        size = numpy.fromstring(file.read(8), dtype=numpy.uint64)
        file_sizes = numpy.fromstring(file.read(8*size), dtype=numpy.uint64)
        self._offsets = numpy.append(numpy.uint64(0), numpy.add.accumulate(file_sizes))
        self._memory = file.read()
        file.close()

    def get(self, i):
        return self._memory[self._offsets[i]:self._offsets[i+1]]

    def size(self):
        return len(self._offsets)-1

class bcf_store_file():
    def __init__(self, filename):
        self._filename = filename
        print 'Opening BCF file ... '+filename
        self._file = open(filename, 'rb')
        size = numpy.fromstring(self._file.read(8), dtype=numpy.uint64)
        file_sizes = numpy.fromstring(self._file.read(8*size), dtype=numpy.uint64)
        self._offsets = numpy.append(numpy.uint64(0), numpy.add.accumulate(file_sizes))

    def __del__(self):
        self._file.close()

    def get(self, i):
        self._file.seek(len(self._offsets)*8+self._offsets[i])
        return self._file.read(self._offsets[i+1]-self._offsets[i])

    def size(self):
        return len(self._offsets)-1
