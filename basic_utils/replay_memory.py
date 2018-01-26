from basic_utils.utils import *
from basic_utils.options import *
from collections import defaultdict
import random
import operator


class SegmentTree(object):
    def __init__(self, capacity, operation, neutral_element):
        assert capacity > 0 and capacity & (capacity - 1) == 0, "capacity must be positive and a power of 2."
        self._capacity = capacity
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
                )

    def reduce(self, start=0, end=None):
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        # index of the leaf
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(
                self._value[2 * idx],
                self._value[2 * idx + 1]
            )
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(
            capacity=capacity,
            operation=operator.add,
            neutral_element=0.0
        )

    def sum(self, start=0, end=None):
        return super(SumSegmentTree, self).reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self._capacity:  # while non-leaf
            if self._value[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._capacity


class ReplayBuffer:
    """
    The replay memory.
    """
    def __init__(self, memory_cap=200000, batch_size_q=64):
        """
        Args:
            memory_cap: the max length of the buffer
            batch_size_q: batch size
        """
        self._storage = []
        self._maxsize = memory_cap
        self.batch_size = batch_size_q
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, data):
        """
        Add new data to the replay memory.

        Args:
            data: new data to add
        """
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        path = defaultdict(list)
        for i in idxes:
            d = self._storage[i]
            path["observation"].append(d[0])
            path["action"].append(d[1])
            path["next_observation"].append(d[2])
            path["reward"].append(d[3])
            path["not_done"].append(d[4])
        for k in path:
            path[k] = np.array(path[k])
        return path

    def sample(self):
        """
        Sample data from the memory.

        Return:
            the sampled path
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(self.batch_size)]
        return self._encode_sample(idxes)

    def update_priorities(self, idxes, priorities):
        pass


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    The prioritized replay memory.
    """
    def __init__(self, memory_cap=200000, batch_size_q=64, alpha=0.8, beta=0.6):
        super(PrioritizedReplayBuffer, self).__init__(memory_cap, batch_size_q)
        self._alpha = alpha
        self._beta = beta

        it_capacity = 1
        while it_capacity < self._maxsize:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, *args, **kwargs):
        """
        Add new data to the replay memory.

        Args:
            data: new data to add
        """
        idx = self._next_idx
        super().add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        for _ in range(batch_size):
            mass = random.random() * self._it_sum.sum(0, len(self._storage) - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self):
        """
        Sample data from the memory.

        Return:
            the sampled path
        """
        idxes = self._sample_proportional(self.batch_size)
        weights = []
        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-self._beta) if p_sample > 1e-12 else 0
            weights.append(weight)
        weights = np.array(weights)/np.sum(weights)
        encoded_sample = self._encode_sample(idxes)
        encoded_sample["weights"] = weights
        encoded_sample["idxes"] = idxes
        return encoded_sample

    def update_priorities(self, idxes, priorities):
        priorities = list(priorities.reshape((-1,)))
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._max_priority = max(self._max_priority, priority)
