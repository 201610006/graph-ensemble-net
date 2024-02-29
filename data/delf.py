import glob
import os
imgfile = glob.glob("val/*")
for fl in imgfile:
    imgname = glob.glob(fl+'/*')
    for itm in imgname:
        path = itm.replace('val','dataset/train')
        isExists = os.path.exists(path)

        if isExists:
            os.remove(path)
