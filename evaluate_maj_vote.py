import argparse
from tqdm import tqdm
import os
import PIL.Image as Image
import importlib.util
import torch

parser = argparse.ArgumentParser(description='RecVis A3 evaluation script')
parser.add_argument('--data', type=str, default='bird_dataset', metavar='D',
                    help="folder where data is located. test_images/ need to be found in the folder")
# parser.add_argument('--model', type=str, metavar='M',
#                     help="the model file to be evaluated. Usually it is of the form model_X.pth")
parser.add_argument('--outfile', type=str, default='experiment/kaggle.csv', metavar='D',
                    help="name of the output csv file")

nClasses = 20
model_list = [
    ("model_resnet50", "experiment/model1.pth"),
    ("model_resnet50", "experiment/model2.pth"),
    ("model_resnet50", "experiment/model3.pth"),
    ("model_resnet50", "experiment/model4.pth"),
    ("model_resnet50", "experiment/model5.pth"),
]

args = parser.parse_args()
use_cuda = torch.cuda.is_available()
models = []
for m, m_w in model_list:
    state_dict = torch.load(m_w, map_location=torch.device("cpu"))
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    state_dict = new_state_dict
    # load params
    spec = importlib.util.spec_from_file_location(m, f"./models/{m}.py")
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    model = foo.Net()
    model.load_state_dict(new_state_dict)
    model.eval()
    if use_cuda:
        model.cuda()
    models.append(model)

if use_cuda:
    print('Using GPU')
else:
    print('Using CPU')

from data import eval_transforms

test_dir = args.data + '/test_images/mistery_category'

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


output_file = open(args.outfile, "w")
output_file.write("Id,Category\n")
for f in tqdm(os.listdir(test_dir)):
    if 'jpg' in f:
        data = eval_transforms(pil_loader(test_dir + '/' + f))
        data = data.view(1, data.size(0), data.size(1), data.size(2))
        if use_cuda:
            data = data.cuda()
        outputs = torch.zeros((len(model_list), nClasses))
        for i, model in enumerate(models):
            outputs[i] = model(data)
            outputs[i] = torch.nn.Softmax(outputs[i])
        output = torch.sum(outputs, 0)
        pred = output.data.max(1, keepdim=True)[1]
        output_file.write("%s,%d\n" % (f[:-4], pred))

output_file.close()

print("Succesfully wrote " + args.outfile + ', you can upload this file to the kaggle competition website')
        


