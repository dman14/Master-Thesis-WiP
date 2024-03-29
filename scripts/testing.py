from skimage.metrics import structural_similarity as ssim
from scripts.img_helper import *
from scripts.dataloader import *

class Tester:
  def __init__(self):
    a = 1
  def setup_dataloader(self, test_path, scale = 4, 
                       reupscale = None,batch_size = 1, single = None, size = 64,
                       shuffle = False, num_workers = 0,repatch_data=True):
    self.size = size
    self.dataloader_main = SRDataLoader_patches(test_path , scale,
                                        reupscale, single,
                                        batch_size, size,
                                        shuffle, num_workers, repatch_data)
    self.test_dataloader = self.dataloader_main.get_dataloader_patches()
    
  def load_model(self, model):
    self.model = model
  
  def run_test(self, model, device, test_loader, test_step, loss_func):
    self.psnr_list = []
    self.ssim_list = []
    model.eval()
    with torch.no_grad():
      #for lr_batch, ref_batch,lr_patch_data,hr_patch_data in test_loader:
      for a in test_loader:
        (batch,lr_patch_data,hr_patch_data) = a[0]
        (lr_batch, ref_batch) = batch
        output = test_step(model, device, lr_batch,lr_patch_data,self.size)
        (mask_t, base_tensor, t_size, padding) = hr_patch_data
        ref = image_from_patches(ref_batch,mask_t, base_tensor, t_size,self.size*4 ,padding)
        for i in range (0,1):
          ref = ref.squeeze(0).permute(1,2,0).cpu().detach().numpy()
          aux= output[i].permute(1,2,0)
          aux = aux.cpu().detach().numpy()
          ref = ref*255
          #aux = aux/255
          
          #imshow(ref)
          #imshow(aux)

          psnr, ssimScore = quality_measure_YCbCr(ref, aux)
          print(psnr)
          self.psnr_list.append(psnr)
          self.ssim_list.append(ssimScore)
    
 
  def mean_metrics(self, psnr_list, ssim_list):
    self.mean_psnr = np.mean(psnr_list)
    self.ssim_list = np.mean(ssim_list)
 
  def example_image():
    a = 1
  def main(self, test_step, device, loss_func, model, test_path,
           scale = 4, reupscale = None, single = None,
           size = 64, shuffle = False, num_workers = 0, batchsize = 1):
    
    self.setup_dataloader(test_path=test_path, scale= scale, reupscale= reupscale,
                          batch_size=batchsize, single=single, size=size,
                          shuffle=shuffle, num_workers=num_workers)

    self.load_model(model)
    self.run_test(self.model, device, self.test_dataloader,test_step, loss_func)
    self.mean_metrics(self.psnr_list, self.ssim_list)
    return self.mean_psnr, self.ssim_list