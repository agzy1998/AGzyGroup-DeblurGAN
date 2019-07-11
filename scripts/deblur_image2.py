import numpy as np
from PIL import Image
import click
import os

from deblurgan.model import generator_model
from deblurgan.utils import load_image, deprocess_image, preprocess_image

class myDeblur:
    def deblur(self,image_path,model_path):
        data = {
            'A_paths': [image_path],
            'A': np.array([preprocess_image(load_image(image_path))])
        }
        x_test = data['A']
        g = generator_model()
        g.load_weights(model_path)
        generated_images = g.predict(x=x_test)
        generated = np.array([deprocess_image(img) for img in generated_images])
        x_test = deprocess_image(x_test)

        image_path2=image_path
        (filepath, tempfilename) = os.path.split(image_path2)
        print(tempfilename)
        for i in range(generated_images.shape[0]):
            x = x_test[i, :, :, :]
            img = generated[i, :, :, :]
            output = np.concatenate((x, img), axis=1)
            im = Image.fromarray(output.astype(np.uint8))
            im.save('../outputs/'+'deblur'+tempfilename)

        return '../outputs/'+'deblur'+tempfilename


@click.command()
@click.option('--image_path', help='Image to deblur')
def deblur_command(image_path):
    return myDeblur().deblur(image_path)


if __name__ == "__main__":
    deblur_command()
