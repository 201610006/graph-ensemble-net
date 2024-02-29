import torch
from thop import profile

from Inception_v4 import InceptionV4



num_classes = 4

#model = resnet50(num_classes)
#model = ViT(num_classes=num_classes)
#model = t2t_vit_7()
#model = SwimTransformer(num_classes=num_classes)
#model = efficientnetv2_s(num_classes=num_classes)
model = InceptionV4(n_classes=num_classes)

dummy_input = torch.randn(1, 3, 256, 256) #flops: 67916.60 M, params: 69.16 M

flops, params = profile(model, (dummy_input,))
print('flops: ', flops, 'params: ', params)
print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))




