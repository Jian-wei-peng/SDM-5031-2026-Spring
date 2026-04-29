import torch


def get_random_problems(batch_size, problem_size, distribution="uniform"):
    if distribution != "uniform":
        raise ValueError("POMO_model is the model-only ablation and supports only uniform training data.")
    return torch.rand(size=(batch_size, problem_size, 2))


def augment_xy_data_by_8_fold(problems):
    # problems.shape: (batch, problem, 2)
    x = problems[:, :, [0]]
    y = problems[:, :, [1]]

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    return torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
