import torch
from byol_pytorch import BYOL
from torchvision import models

vision_transformer = models.vit_b_16(pretrained=True)

learner = BYOL(
    vision_transformer,
    image_size = 256,
    hidden_layer = 'avgpool'# what's this for?
)

opt = torch.optim.Adam(learner.parameters(), lr=3e-4)

def sample_unlabelled_images():
    return torch.randn(20, 3, 256, 256)

for _ in range(100):
    images = sample_unlabelled_images()
    loss = learner(images)
    opt.zero_grad()
    loss.backward()
    opt.step()
    learner.update_moving_average() # update moving average of target encoder

# save your improved network
torch.save(vision_transformer.state_dict(), './pretrained-net.pth')