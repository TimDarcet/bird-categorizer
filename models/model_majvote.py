import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import importlib

nclasses = 20

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model_name_list = [
            # ("model_resnet50", "experiment/model_1.pth"),
            # ("model_resnet50", "experiment/model_2.pth"),
            # ("model_resnet50", "experiment/model_3.pth"),
            ("model_resnet50", "experiment/model_9.pth"),
            ("model_resnet50", "experiment/model_10.pth"),
        ]
        self.models = nn.ModuleList()
        for m, m_w in self.model_name_list:
            state_dict = torch.load(m_w, map_location=torch.device("cpu"))
            # from collections import OrderedDict
            # new_state_dict = OrderedDict()
            # for k, v in state_dict.items():
            #     name = k[7:] # remove `module.`
            #     new_state_dict[name] = v
            # state_dict = new_state_dict
            # load params
            spec = importlib.util.spec_from_file_location(m, f"./models/{m}.py")
            foo = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(foo)
            model = foo.Net()
            model.load_state_dict(state_dict)
            model.eval()
            for param in model.parameters():
                param.require_grad = False
            self.models.append(model)
        self.sm = torch.nn.Softmax(2)

    def forward(self, x):
        # for model in self.models:
        #     print(model(x)[0])
        outputs = torch.stack(tuple(model(x) for model in self.models))
        # print(outputs.shape)
        outputs = self.sm(outputs)
        output = torch.sum(outputs, 0)
        # print(output.shape)
        return output

