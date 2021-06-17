import os
from collections import OrderedDict
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer

opt = TestOptions().parse(save=False)
opt.display_id = 0
opt.nThreads = 1
opt.batchSize = 1
opt.serial_batches = True
opt.no_flip = True
opt.in_the_wild = True
opt.traverse = True
opt.interp_step = 0.1

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
visualizer = Visualizer(opt)

opt.name = 'males_model' # change to 'females_model' if you're trying the code on a female image
model = create_model(opt)
model.eval()

img_path = 'test.jpg'
data = dataset.dataset.get_item_from_path(img_path)
visuals = model.inference(data)

os.makedirs('results', exist_ok=True)
out_path = os.path.join('results', os.path.splitext(img_path)[0].replace(' ', '_') + '.png')
visualizer.save_row_image(visuals, out_path)
print(out_path)
