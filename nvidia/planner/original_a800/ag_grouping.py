"""Plan grouped contiguous AG GEMMs without changing peer communication.

Pure planning only; plans must not be used as a distributed execution path until
stream ordering, remote buffer lifetime and ACK reuse are validated on TP8.
"""

def contiguous_groups(rank, world, group_size, order='shift'):
    if world < 2 or not 0 <= rank < world or group_size not in (1, 2, 4, 8) or order not in ('fixed', 'shift'):
        raise ValueError('invalid rank, world, group size or order')
    peers = ([p for p in range(world) if p != rank] if order == 'fixed'
             else [(rank + offset) % world for offset in range(1, world)])
    groups = []
    for peer in peers:
        if not groups or len(groups[-1]) >= group_size or peer != groups[-1][-1] + 1:
            groups.append([peer])
        else:
            groups[-1].append(peer)
    return tuple(tuple(group) for group in groups)
