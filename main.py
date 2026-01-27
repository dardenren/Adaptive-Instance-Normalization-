import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image

from model import Encoder, Decoder, VGGFeat
from config import Config_Args
from config import *
from utils import to_float_4d

# Add GPU support with DEVICE

def main(args):
    try:
        style_1 = Image.open('style/style_1.jpg')
        content_1 = Image.open('content/content_1.jpg')
        content_2 = Image.open('content/content_2.jpg')
        content_3 = Image.open('content/content_3.jpg')

        image_size = 256
        transform = transforms.Compose(
            [
                # transforms.PILToTensor() ,
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

        style_1_tensor = transform(style_1)
        content_1_tensor = transform(content_1)
        # content_2_tensor = transform(content_2)
        # content_3_tensor = transform(content_3)

        style_1_tensor   = style_1_tensor.unsqueeze(0).to(DEVICE)    
        content_1_tensor = content_1_tensor.unsqueeze(0).to(DEVICE)


        logger.info("Loaded image tensors successfully")

        encoder = Encoder().to(DEVICE).eval()
        decoder = Decoder().to(DEVICE).train()

        content_features = encoder(content_1_tensor)
        # style_features = encoder(style_1_tensor)
        # style_1_tensor_u = style_1_tensor.unsqueeze(0)
        style_features = encoder(style_1_tensor)

        for p in encoder.parameters():
            p.requires_grad_(False)

        modified_features,normalized_content,normalized_style = encoder.adain(content_features, style_features, batches=True)
        generate = decoder(modified_features)

        vgg = VGGFeat().to(DEVICE)

        generated_list = vgg(generate)
        # content_1_tensor = to_float_4d(content_1_tensor, DEVICE)

        loss = decoder.loss(generate, generated_list, content_1_tensor, style_features, batches=True)
        logger.info(f"Loss before training: {loss}")

        generate = to_float_4d(generate, DEVICE)

        optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)

        logger.info("Starting training")
        for epoch in range(args.epochs):
            optimizer.zero_grad()

            content_feats = encoder(content_1_tensor)
            style_feats = encoder(style_1_tensor)

            t_feats, _, _ = encoder.adain(content_feats, style_feats, batches=True)

            generated = decoder(t_feats)               

            # gen_feats   = vgg(generated)
            # style_feats = vgg(style_1_tensor)
            generated_feats = encoder(generated)

            total_loss = decoder.loss(generated, generated_feats, content_1_tensor, style_feats, batches=True)

            total_loss.backward()
            optimizer.step()
            
            if epoch % 2:
                logger.info(f"Epoch: {epoch}, Loss: {total_loss} \n" )

        logger.info("Training done, saving model weights.")
        torch.save(decoder.state_dict(), "output/decoder_weights.pth")



    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #  parser.add_argument("--model_name", required=True, help="Model name (e.g., bert-base-uncased)")
    # parser.add_argument("--train_batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=1, help="Seed for randomization")
    args = parser.parse_args()
    Config_Args.update_args(args)
    main(args)