import torch


def aug_near_mix(index, index_list, data, k=10, random_t=0.1, device="cuda"):
    r = (
        torch.arange(start=0, end=index.shape[0]) * k
        + torch.randint(low=1, high=k, size=(index.shape[0],))
    ).to(device)
    random_select_near_index = (
        index_list[index][:, :k].reshape((-1,))[r].long()
    )
    random_select_near_data2 = data[random_select_near_index]
    random_rate = torch.rand(size=(index.shape[0], 1, 1)).to(device) * random_t
    # import pdb; pdb.set_trace()
    data_origin = data[index]
    
    return (
        random_rate * random_select_near_data2 + (1 - random_rate) * data_origin
    )


def aug_near_feautee_change(index, index_list, data, k=10, t=0.99, device="cuda"):
    r = (
        torch.arange(start=0, end=index.shape[0]) * k
        + torch.randint(low=1, high=k, size=(index.shape[0],))
    ).to(device)
    random_select_near_index = (
        index_list[index][:, :k].reshape((-1,))[r].long()
    )
    random_select_near_data2 = data[random_select_near_index]
    data_origin = data[index]
    random_rate = torch.rand(size=(data_origin.shape[1], data_origin.shape[2]), device=device)
    random_mask = (random_rate > t).float()
    out = random_select_near_data2 * random_mask + data_origin * (1 - random_mask)
    return out

def aug_randn(index, data, index_list, k=None, normal_t=0.01, device="cuda"):
    data_origin = data[index]
    return (
        data_origin
        + torch.randn(data_origin.shape, device=data_origin.device) * 0.1 * normal_t
    )
