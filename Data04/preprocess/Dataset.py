import numpy as np
import torch
import torch.utils.data

from TDGCN.Data08.transformer import Constants


class EventData(torch.utils.data.Dataset):
    """
    Event stream dataset.
    事件流数据集。
    """

    def __init__(self, data, drop_last=True):
        """
        Data should be a list of event streams; each event stream is a list of dictionaries;
        数据应该是事件流的列表;每个事件流都是一个字典列表;
        each dictionary contains: time_since_start, time_since_last_event, type_event
        """
        self.time = [[elem['time_since_start'] for elem in inst] for inst in data]
        self.time_gap = [[elem['time_since_last_event'] for elem in inst] for inst in data]

        # plus 1 since there could be event type 0, but we use 0 as padding
        # 加上1，因为事件类型可以是0，但我们用0作为填充
        self.event_type = [[elem['type_event'] + 1 for elem in inst] for inst in data]
        self.length = len(data)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Each returned element is a list, which represents an event stream
        每个返回的元素都是一个列表，它代表一个事件流
        """
        return self.time[idx], self.time_gap[idx], self.event_type[idx]


def pad_time(insts):
    """
    Pad the instance to the max seq length in batch.
    在批处理中将实例填充到最大seq长度。
    """
    max_len = max(len(inst) for inst in insts)# max_len=172

    batch_seq = np.array([
        inst + [Constants.PAD] * (max_len - len(inst))
        for inst in insts])

    return torch.tensor(batch_seq, dtype=torch.float32)


def pad_type(insts):
    """
    Pad the instance to the max seq length in batch.
    批量将实例填充到最大seq长度。
    """

    max_len = max(len(inst) for inst in insts)

    batch_seq = np.array([
        inst + [Constants.PAD] * (max_len - len(inst))
        for inst in insts])

    return torch.tensor(batch_seq, dtype=torch.long)


def collate_fn(insts):
    """
    Collate function, as required by PyTorch.
    collate函数，如PyTorch所要求的。
    """

    time, time_gap, event_type = list(zip(*insts))
    time = pad_time(time)
    time_gap = pad_time(time_gap)
    event_type = pad_type(event_type)
    return time, time_gap, event_type


def get_dataloader(data, batch_size, shuffle=False, drop_last=True):
    """ Prepare dataloader. """

    ds = EventData(data)
    dl = torch.utils.data.DataLoader(
        ds,
        num_workers=2,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle,
        drop_last=True
    )
    return dl
