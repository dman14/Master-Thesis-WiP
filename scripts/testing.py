from skimage.metrics import structural_similarity as ssim
from scripts.img_helper import *

class Tester:
  def __init__(self):
  
  def setup_dataloader(self, test_path, scale = 4, 
                       reupscale = None,batch_size = 1, single = None, size = 64,
                       shuffle = False, num_workers = 1):
  
    self.dataloader_test = SRDataLoader(test_path, scale, reupscale,
                                        single, size, shuffle, num_workers, batch_size)
    
  def load_model(self, model):
    self.model = model
  
  def run_test(self, model, device, test_loader, test_step, device, loss_func):
    self.psnr_list = []
    self.ssim_list = []
    model.eval()
    with torch.no_grad()
      for data, target in test_loader:
        data = data.to_device()
        target = target.to_device()
        output, loss = test_step(model, device, test_loader, loss_func)
        #loss?

        psnr, ssimScore = quality_measure_YCbCr(target, output)

        self.psnr_list.append(psnr)
        self.ssim_list.append(ssimScore)
    
 
  def mean_metrics(psnr_list, ssim_list):
    self.mean_psnr = np.mean(psnr_list)
    self.ssim_list = np.mean(ssim_list)
 
  def example_image():
 
  def main(self, model, test_path, scale = 4,
           reupscale = None, single = None,
           size = 64, shuffle = False, num_workers = 0,
           test_step, device, loss_func):
    
    self.setup_dataloader(test_path, scale, reupscale,
                          single, size, batchsize,
                          shuffle, num_workers)
    self.load_model(model)
    self.run_test(self.model, device, self.dataloader_test,test_step, device,
                  loss_func)
    self.mean_metrics(self.psnr_list, self.ssim_list)
    return self.mean_psnr, self.ssim_list