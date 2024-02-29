from einops import rearrange
from PIL import Image
import numpy as np
import torch
import glob
import json
torch.set_grad_enabled(False)
import cv2
import torchvision.transforms as transforms
from torch.autograd import Variable
#from MobileNet import mobilenet_pretrained
from models.MobileNet_lce import mobilenet_pretrained
# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]


def readjson(path):
    with open(path, 'r', encoding='utf8') as fp:
    # with open(path, 'r') as fp:
        json_data = json.load(fp)
        fh = json_data["annotations"]

        return fh

# 输入文件夹，从json中读取非标准的数据占比
def count_nostd(imgpath, jsonpath, notstdpath):
    js = readjson(jsonpath)

    notstdimg = glob.glob(notstdpath+"\\*")
    bb = []
    for itm in notstdimg:
        imgname = itm[itm.rindex("\\") + 1:]
        bb.append(imgname)

    imgs = glob.glob(imgpath+"\\*")
    count_std = 0
    count_not_std = 0
    count_other = 0
    nmjs = {}
    std = "其它"
    for itm in imgs:
        imgname = itm[itm.rindex("\\")+1:]
        if imgname not in js:
            if imgname in bb:
                count_not_std +=1
                std = "非标准"
            else:
                count_other +=1
        else:
            if "standard" in js[imgname]:
                std = js[imgname]["standard"]
            else:
                std = js[imgname]["annosets"][0]["standard"]
            if std =="":
                std ="其它"
            # print(imgname,",",std)
            if std == "标准" or std == "基本标准":
                count_std +=1
            else:
                count_not_std += 1
        # nmjs[imgname] = std
        print(imgname, ",", std)
    # print(nmjs)
    print("总数：",len(imgs),"标准：",count_std,"非标准：",count_not_std,"其他：",count_other)


def default_loader(path):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)/255.
    image = torch.Tensor(image).float()
    image = rearrange(image, 'h w -> 1 1 h w')
    return image

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
transform_test = transforms.Compose([
    transforms.Resize((416, 416)),
    #transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(0.4),
    transforms.ToTensor(),
    #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
])

def load_image(imgpth):

    img_x = Image.open(imgpth)
    roi_img = img_x.convert("RGB")
    roi_img = transform_test(roi_img)
    roi_img.unsqueeze_(0)
    roi_img = Variable(roi_img).to(device)
    return roi_img

def load_checkpoint_model(model, ckpt_best, device):
    state_dict = torch.load(ckpt_best, map_location=device)
    model.load_state_dict(state_dict['state_dict'])
    #model.load_state_dict(state_dict)
    return model

def visuals(model, imgpth, savepath):
    model.eval()
    pv0, pv1, pv2, pv3 = [], [], [], []

    # 注册hook
    hooks = [
        model.sa.sigmoid.register_forward_hook(
            lambda self, input, output: pv0.append(output)
        )
        # ,model.att1.sigmoid.register_forward_hook(
        #     lambda self, input, output: pv1.append(output)
        # ),
        # model.att2.sigmoid.register_forward_hook(
        #     lambda self, input, output: pv2.append(output)
        # ),
        # model.att3.sigmoid.register_forward_hook(
        #     lambda self, input, output: pv3.append(output)
        # )
    ]

    imgname = imgpth[imgpth.rindex('/')+1:]
    # propagate through the model
    img = default_loader(imgpth)
    outputs,_ = model(img)

    # 用完的hook后删除
    for hook in hooks:
        hook.remove()

    out = pv0[0]
    #print(out.max(), out.min())
    print(out.shape)  # [1, 128, 26, 26]

    ott = np.array(out[0][0:60, :, :])  # [128, 26, 26]

    np.seterr(divide='ignore', invalid='ignore')
    print("shape=", ott.shape)
    heatmap = np.mean(ott, axis=0)

    heatmap = np.maximum(heatmap, 0)
    # heatmap /= np.max(heatmap)

    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    print("shape=", heatmap.shape, "max=", ott.max(), "min=", ott.min())

    image = cv2.imread(imgpth, 0)
    x = image.shape[1]  ##宽度
    y = image.shape[0]  ##高度
    target = max(x, y)
    BLACK = [0, 0, 0]
    a = (target - x) / 2
    b = (target - y) / 2
    orgimg = cv2.copyMakeBorder(image, int(b), int(b), int(a), int(a), cv2.BORDER_CONSTANT, value=BLACK)

    orgimg = cv2.cvtColor(orgimg, cv2.COLOR_GRAY2BGR)

    heatmap = cv2.resize(heatmap, (orgimg.shape[1], orgimg.shape[0]))

    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = heatmap * 0.4 + np.array(orgimg)
    cv2.imwrite(savepath+imgname, superimposed_img)

if __name__ == '__main__':


    phi = 0.75

    low_phi_count = 0

    high_phi_count = 0

    high_std = 0
    high_none_std = 0

    high_and_true = 0

    classes = {"chiraogu":0, "foot":1, "jingfeigu":5, "gugu":3, "hand":4, "gongu":2}

    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available()
                              else "cpu")
    #model = GEEL_Net(num_classes=10).to(device)
    model = mobilenet_pretrained(num_classes=10, pretrained=False).to(device)
    
    #pth = 'output/train/20231109-114245-mobilenet-256/model_best.pth.tar'
    # 置信度
    pth = 'output/train/20231114-165422-mobilenet-256/model_best.pth.tar'
    
    model = load_checkpoint_model(model, pth, device)
    model.eval()
    # imgpth = r"E:\gu_classification\val\chiraogu\000abd963e7b41c9b0b849fb2cc865e0.jpg"
    # imgpth = r"E:\gu_classification\val\chiraogu\00b0e2ab8ace4c5eb731f06c2465e837.jpg"
    imgpth = "data/dataset_none_std/val"
    for imgfile in glob.glob(imgpth+'/*'):
        class_file = imgfile[imgfile.rindex("/")+1:]
        class_true = classes[class_file]
        for imgs in glob.glob(imgfile+'/*'):

            imgname = imgs[imgs.rindex("/")+1:]
            with open("nmjs2.json", 'r', encoding='utf8') as fp:
                json_data = json.load(fp)
            std = json_data[imgname]

            img = load_image(imgs)
            out, conf = model(img.to(device))

            predicted_index = torch.argmax(out)

            cf = conf
            #cf = out.softmax(dim=1).max()
            #print(cf)
            if cf > phi:
                high_phi_count +=1
                if std =="标准" or (std =="基本标准" and predicted_index==4):
                    high_std +=1
                else:
                    #print(imgname, class_file, std, class_true)
                    high_none_std +=1

                if predicted_index==class_true:  # 预测正确
                    high_and_true +=1

            else:
                low_phi_count +=1

    print("Discarded:",low_phi_count/(low_phi_count+high_phi_count),"N/T:", high_none_std/high_phi_count, "ACC:", high_and_true/high_phi_count)

