from mpi4py import MPI


# each mpi processor holds one synchronizer
# call synchronizer.sync(data) to synchroning data
#   after this call, all processes have same data
class Synchronizer(object):
    def __init__(self):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

    def sync(self, data):
        return self.comm.allgather(data)
