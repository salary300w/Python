# pytorch相关
## 模型的保存
```python
import torch
import torchvision

vgg16 = torchvision.models.vgg16()

# 保存方式一
torch.save(vgg16, "file_path")
# 保存网络模型的结构及其参数
# 加载方法一
model = torch.load("file_path")


# 保存方式二（推荐）
torch.save(vgg16.state_dict(), "file_path")
# 仅保存网络模型的参数
# 加载方法二

# 需新建网络模型的结构
vgg16 = torchvision.models.vgg16()
model = torch.load("file_path")
vgg16.load_state_dict(model)
```