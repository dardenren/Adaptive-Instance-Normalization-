import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image

from config import Config_Args
from config import *

class Encoder(nn.Module):
  def __init__(self):
    super(Encoder, self).__init__()

    self.chosen_layers = ['2', '9', '16', '29']
    self.vggmodel = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features[:29].eval()
    self.module_list = nn.ModuleList()
    # self.modified_model = self.modified_model() #To add reflection padding and other layers
    # self.decoder = self.generate_decoder()

    for name, module in self.vggmodel.named_children():
      if isinstance(module, nn.Conv2d):
        # Add padding layer to the Sequential container
        self.module_list.append(nn.ReflectionPad2d(padding=1))
      # Add the original convolutional layer directly
      self.module_list.append(module)

    self.modules = nn.Sequential(*self.module_list)


    # for name,module in self.model.named_modules():
    # #Add reflection padding before conv layers
    #   if isinstance(module, nn.Conv2d):
    #     padding_layer = nn.ReflectionPad2d(padding=1)
    #     self.module_list.append(padding_layer)
    #   self.module_list.append(module)


  #Feature extraction of selected layers, basically functions as the encoder
  def forward(self,x):  #encoding process
    features = []
    if x.dtype != torch.float:
      x = x.float()
    for layer_num, layer in enumerate(self.module_list):
      if layer_num == 30:
        break

      x = layer(x)

      if str(layer_num) in self.chosen_layers:
        features.append(x)

    return features  #This list is accessible after passing thru the model, e.g. list = model(image)
                     #This represents the raw features after passing thru the layers


  def adain(self,content_features, style_features,batches=True,epsilon=1e-7):
    #features is the output from self.forward, shape is B*C*H*W or C*H*W
    epsilon = epsilon
    if len(content_features) != len(style_features):
        raise ValueError("Content and style features lists must have the same length.")

    # Calculate channel-wise mean and standard deviation
    encoded_features = []
    normalized_content_list = []
    normalized_style_list = []

    for content_feature, style_feature in zip(content_features, style_features):

      if content_feature.shape[0] != style_feature.shape[0]:
        raise ValueError("Content and style features in each pair must have the same number of channels.")

      if batches:
        content_mean = torch.mean(content_feature, dim=[2, 3], keepdim=True)
        content_std = torch.std(content_feature, dim=[2, 3], keepdim=True, unbiased=False) + epsilon

        style_mean = torch.mean(style_feature, dim=[2, 3], keepdim=True)
        style_std = torch.std(style_feature, dim=[2, 3], keepdim=True, unbiased=False) + epsilon

      else:
        content_mean = torch.mean(content_feature, dim=[1, 2], keepdim=True)
        content_std = torch.std(content_feature, dim=[1, 2], keepdim=True, unbiased=False) + epsilon

        style_mean = torch.mean(style_feature, dim=[1, 2], keepdim=True)
        style_std = torch.std(style_feature, dim=[1, 2], keepdim=True, unbiased=False) + epsilon

      # Normalize features
      normalized_content = (content_feature - content_mean) / content_std
      normalized_style = (style_feature - style_mean) / style_std
      normalized_content_list.append(normalized_content)
      normalized_style_list.append(normalized_style)

      encoded_feature = style_std * normalized_content + style_mean
      encoded_features.append(encoded_feature)

    return encoded_features,normalized_content_list,normalized_style_list
    #It is not a concern if the encoded_features have high values such as 800, especially
    #in the later layers the feature values gets multiplied by the filters and weights.






  # def generate_decoder(self):
  #   features = nn.ModuleList()
  #   for name, module in self.modified_model.named_modules()[::-1]:
  #     if isinstance(module, nn.MaxPool2d):
  #       features.append(nn.Upsample(scale_factor=2, mode='nearest'))
  #     else:
  #       features.append(module)

  #   return features




  #1)Calculate mean and s.d content and style features channel-wise, and normalize channel-wise
  #2)Apply style features scaling and shifting to content features
  def transform(self, content_features, style_features):
    for content_feature, style_feature in zip(content_features,style_features):
        pass



  #Get finalized image
  def decode(self,input):
    for layer in self.decoder:
      input = layer(input)

    return input

#MultiScaleDecoder
class Decoder(nn.Module):
  def __init__(self):
    super(Decoder, self).__init__()
    self.mse_loss = nn.MSELoss()
    self.resize = transforms.Resize((256,256))

    # 516 to 256 channels
    self.branch1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                 nn.ReflectionPad2d((1, 1, 1, 1)),
                                 nn.Upsample(scale_factor=2, mode='nearest'),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                 nn.ReflectionPad2d((1, 1, 1, 1)),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                 nn.ReflectionPad2d((1, 1, 1, 1)),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                 nn.ReflectionPad2d((1, 1, 1, 1)),
                                 nn.ReLU(inplace=True),
                                 )
    #256 to 128
    self.branch2 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                 nn.ReflectionPad2d((1, 1, 1, 1)),
                                 nn.Upsample(scale_factor=2, mode='nearest'),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                 nn.ReflectionPad2d((1, 1, 1, 1)),
                                 nn.ReLU(inplace=True),
                                 )
    #128 to 64
    self.branch3 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                nn.ReflectionPad2d((1, 1, 1, 1)),
                                nn.Upsample(scale_factor=2, mode='nearest'),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                nn.ReflectionPad2d((1, 1, 1, 1)),
                                nn.ReLU(inplace=True),
   )

    #64 to 3
    self.branch4 = nn.Conv2d(64, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    self.conv_final = nn.Conv2d(12, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.relu_final = nn.ReLU(inplace=True)


    # (0): ReflectionPad2d((1, 1, 1, 1))
    # (1): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # (2): ReLU(inplace=True)
    # (3): ReflectionPad2d((1, 1, 1, 1))
    # (4): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # (5): ReLU(inplace=True)
    # (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    # (7): ReflectionPad2d((1, 1, 1, 1))
    # (8): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # (9): ReLU(inplace=True)
    # (10): ReflectionPad2d((1, 1, 1, 1))
    # (11): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # (12): ReLU(inplace=True)
    # (13): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    # (14): ReflectionPad2d((1, 1, 1, 1))
    # (15): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # (16): ReLU(inplace=True)
    # (17): ReflectionPad2d((1, 1, 1, 1))
    # (18): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # (19): ReLU(inplace=True)
    # (20): ReflectionPad2d((1, 1, 1, 1))
    # (21): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # (22): ReLU(inplace=True)
    # (23): ReflectionPad2d((1, 1, 1, 1))
    # (24): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # (25): ReLU(inplace=True)
    # (26): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    # (27): ReflectionPad2d((1, 1, 1, 1))
    # (28): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

  # x is a list consisting of the below tensors
  # torch.Size([64, 258, 258])
  # torch.Size([128, 132, 132])
  # torch.Size([256, 69, 69])
  # torch.Size([512, 39, 39])


  def forward(self,x):
    final_conv = []
    for tensor in x:
      if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(0)
      if tensor.shape[1] == 512:
        tensor = self.branch1(tensor)
        tensor = self.branch2(tensor)
        tensor = self.branch3(tensor)
        tensor = self.branch4(tensor)
      elif tensor.shape[1] == 256:
        tensor = self.branch2(tensor)
        tensor = self.branch3(tensor)
        tensor = self.branch4(tensor)
      elif tensor.shape[1] == 128:
        tensor = self.branch3(tensor)
        tensor = self.branch4(tensor)
      elif tensor.shape[1] == 64:
        tensor = self.branch4(tensor)

      tensor = self.resize(tensor)
      final_conv.append(tensor) #concatenate?

    final_catted = torch.cat((final_conv),1)
    output = self.conv_final(final_catted)
    output = self.relu_final(output)

    return output
    #output is Bx3x256x256


    #Firstly, decoder is trained to turn the adain-transformed features into a picture
    #Secondly, the output of this decoder is passed through the 'frozen' encoder to get adain-ed features
    #Thirdly, this output is used to be compared with the adain-ed content and style for loss and backprop happens.

    #'content' is adain-ed output(modified_features), generated is output of this model forward-prop.
    #style_list and generated_list are lists of outputs from encoder(vgg net)
  def loss(self,generated,generated_list,content,style_list,style_loss_weight = 1,epsilon=1e-7,batches=False):
    content_loss = self.mse_loss(content,generated)

    #style loss is the sum of all layers, generated style image - referenced style image at each layer
    style_loss = 0
    for generated,style in zip(generated_list,style_list):
      if batches:
        style_mean = torch.mean(style, dim=[2, 3], keepdim=True)
        style_std = torch.std(style, dim=[2, 3], keepdim=True, unbiased=False) + epsilon

        generated_mean = torch.mean(generated, dim=[2, 3], keepdim=True)
        generated_std = torch.std(generated, dim=[2, 3], keepdim=True, unbiased=False) + epsilon

      else:
        style_mean = torch.mean(style, dim=[1, 2], keepdim=True)
        style_std = torch.std(style, dim=[1, 2], keepdim=True, unbiased=False) + epsilon

        generated_mean = torch.mean(generated, dim=[1, 2], keepdim=True)
        generated_std = torch.std(generated, dim=[1, 2], keepdim=True, unbiased=False) + epsilon

      mean_loss = self.mse_loss(generated_mean,style_mean)
      std_loss = self.mse_loss(generated_std,style_std)
      style_loss += mean_loss + std_loss

    total_loss = content_loss + style_loss
    return total_loss

      # for i in modified_features:
      #   print(i.shape)
      # torch.Size([64, 258, 258])
      # torch.Size([128, 132, 132])
      # torch.Size([256, 69, 69])
      # torch.Size([512, 39, 39])




    def backprop(self):
      pass




class StyleAware():
   def __init__(self):
    self.mse_loss = nn.MSELoss()

    def gram_matrix(self,feature):
      assert feature.dim() == 1
      C,H,W= feature.shape
      self.resize = self.transforms.Resize((C,H,W))
      resized_feature = self.resize(feature)
      return feature * torch.transpose(feature)

    def loss(self,target_list,generated_list): #AdaIN.get_style_features() outputs are passed through here
      assert len(target_list) == len(generated_list)

      losses = []
      for i in range(len(target_list)):
        #Spatial dimensions (PyTorch default: C*H*W )
        assert target.shape == generated.shape
        spatial_dim = target.shape[1] * target.shape[2]
        target = target_list[i]
        generated = generated_list[i]
        supremum = (self.mse_loss(target) + self.mse_loss(generated)) / spatial_dim



class SlicedWasserstein():
  def __init__(self):
    self.mse_loss = nn.MSELoss()

  def random_directions(self,dim=3,device='cpu'):
    directions = torch.randn(dim, 1).to(device)
    directions_unit_vector = directions / torch.sqrt(torch.sum(directions**2))

    return directions_unit_vector

  #F is the image features, F.shape = C*H*W and C == 3
  def loss(self, F, F_, directions=5):
    losses = []
    for i in range(directions):
      Vs = self.random_directions(dim=len(F.shape))

      # flatten pixel indices to [M,N]
      assert F.shape == F_.shape
      C, H, W = F.shape
      proj_flatten = torch.reshape(F,(C,H*W))
      proj_flatten = torch.transpose(proj_flatten)

      proj_flatten_ = torch.reshape(F_,(C,H*W))
      proj_flatten_ = torch.transpose(proj_flatten_)

      # project each pixel feature onto directions
      projection = torch.zeros_like(F)  #F.shape = (pixels x channels)
      for i in range(F.shape[0]): #1 x 3
        projection[i] = torch.dot(F[i], Vs)

      projection_ = torch.zeros_like(F_)
      for i in range(F_.shape[0]):
        projection_[i] = torch.dot(F_[i], Vs)

      # sort projections for each direction
      torch.sort(projection, axis=0)
      torch.sort(projection_, axis=0)

      losses.append(projection - projection_)

    return self.mse_loss(losses)



#Global functions
def load_image(tensor):
  transform = transforms.ToPILImage()
  image = transform(tensor)
  return image

def save_checkpoint(epoch, model, optimizer, filename="checkpoint.pth"):
  checkpoint = {
      'epoch': epoch,
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict()
  }
  torch.save(checkpoint, filename)


class VGGFeat(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features[:29].eval()
        for p in vgg.parameters():
            p.requires_grad_(False)
        self.vgg = vgg
        self.keep = {1, 6, 11, 20}  # relu1_1..relu4_1

    def forward(self, x):
        feats = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.keep:
                feats.append(x)
            if i == 20:
                break
        return feats
