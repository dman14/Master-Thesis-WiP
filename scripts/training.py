import torch.optim as optim

class Trainer:
  """
  We can add some comment here for later, could be really useful
  """
  def __init__(self):
    """

    """
    # Set cuda or cpu based on config and availability
    self.use_cuda = not self.config.no_cuda and torch.cuda.is_available()
    self.device = torch.device("cuda" if self.use_cuda else "cpu")
    #kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Set random seeds and deterministic pytorch for reproducibility
    random.seed(self.config.seed)       # python random seed
    torch.manual_seed(self.config.seed) # pytorch random seed
    numpy.random.seed(self.config.seed) # numpy random seed
    torch.backends.cudnn.deterministic = True

  def setup_hyperparams(self, batch_size = 30, test_batch_size = 10,
                        epochs = 10, lr = 0.01, momentum = 0.5,
                        no_cuda = False, seed = 42, log_interval = 10):
    """
    function to set-up hyperparameters for the training.
    uses the wandb.config in order to be able to save them in the
    WandB session for later

    Config is a variable that holds and saves hyperparameters and inputs
    config.batch_size = input batch size for training
    config.test_batch_size  = input batch size for testing
    config.epochs = number of epochs to train
    config.lr = learning rate
    config.momentum = SGD momentum
    config.no_cuda = disables CUDA training
    config.seed = random seed
    config.log_interval = how many batches to wait before
                          logging training status
    """
    self.config = wandb.config  # Initialize config
    self.config.batch_size = batch_size
    self.config.test_batch_size = test_batch_size
    self.config.epochs = epochs
    self.config.lr = lr
    self.config.momentum = momentum
    self.config.no_cuda = no_cuda
    self.config.seed = seed
    self.config.log_interval = log_interval

  def setup_wandb(self, project_name="UNSET"):
    """
    Initializes WandB for a new run

    project_name= name of the new run; each run is a single execution of the
                  training script
    """
    self.project_name = project_name
    wandb.init(project= project_name) #need to be run before a new session
    wandb.watch_called = False # Re-run the model without restarting
                               #the runtime, unnecessary after the next release

  def model_save(self, save_path = "/"):
    """
    Saves the model in the current state to the specified path, and with 
    the specified name
    """
    torch.save(model.state_dict(), save_path + self.project_name + ".h5")
    wandb.save(self.project_name + ".h5")

  def setup_train_step(self, training_step):
    """
    A training step function should be passed here to incrorporate it in the
    main training module
    """
    self.training_step = training_step
  
  def setup_test_step(self, test_step):
    """
    A Evaluation step function should be passed here to incrorporate 
    it in the main training module
    """
    self.test_step = test_step

  def setup_dataloaders(self, train_path, val_path,
                        scale = 4, reupscale = None,
                        single = None, size = 64,
                        shuffle = True, num_workers = 0):
    """
    Set-up and load the dataloders for the training
    using the SRDataLoader class
    """
    self.dataloader_main = SRDataLoader(train_path , scale,
                                        reupscale, single,
                                        size, self.config.batch_size,
                                        shuffle, num_workers)
    self.train_dataloader = dataloader_main.get_dataloader()

    self.dataloader_main = SRDataLoader(val_path , scale,
                                        reupscale, single,
                                        size, self.config.test_batch_size,
                                        shuffle, num_workers)
    self.test_dataloader = test_dataloader

  def load_model(self, model):
    self.model = model

  def setup_optimizer(self, optimizer = optim.SGD, optim_kwargs = None):
    """
    Set-up of the optimizer to be used for the training of the model.
    the arguments that need to be supplied are optimizer, and args containing
    extra arguments for the specific type of chosen optimizer
    """
    optim_kwargs["lr"] = self.config.lr
    optim_kwargsm = optim_kwargs
    optim_kwargsm["momentum"] = self.config.momentum
    try:
      self.optimizer = optimizer(model.parameters(), optim_kwargsm)
    except:
      self.optimizer = optimizer(model.parameters(), optim_kwargs)
      
  def setup_loss(self, loss_func):
    """
    
    """
    self.loss_func = loss_func

  def warmup(self):
    """
    https://www.reddit.com/r/MachineLearning/comments/es9qv7/d_warmup_vs_initially_high_learning_rate/
    this might be the momentum in the hyperparameters tho
    """
    #############################################
    a = 1
  
  def setup_grad_skip(self):
    """
    we should set-up here the gradient skipping like in the VDVAE paper
    """
    #############################################
    a = 1

  def resume_training(self, resume_path):
    self.model.load_state_dict(torch.load(resume_path))
  def start_training(self):
    """
    Main function for starting the training proccess after all the other
    parth have been initialized
    """
    # Starting the watch in order to log all layer dimensions, gradients and
    # model parameters to the dashboard
    # Using log="all" log histograms of parameter values in addition to gradients
    wandb.watch(model, log="all")
    model = model.to(self.device)

    for epoch in range(1, config.epochs + 1):

        loss = self.train_step(self.model, self.device,
                               self.train_dataloader, self.optimizer,
                               self.loss_func)
        wanb.log(loss)

        loss = self.test_step(self.model, self.device,
                              self.test_dataloader, self.loss_func)
        wanb.log(loss)

  def Main_start(self, training_step, test_step, model, train_path,
                 val_path, loss_func, batch_size = 30, 
                 test_batch_size = 10, epochs = 10, lr = 0.01,
                 momentum = 0.5, no_cuda = False, seed = 42,
                 log_interval = 10, project_name="UNSET", 
                 save_path = "/", scale = 4, reupscale = None,
                 single = None, size = 64, shuffle = True,
                 num_workers = 0, optimizer = optim.SGD,
                 optim_kwargs = None, resume = False, 
                 resume_path = None):
    """
    Function that receives all arguments, initializes every module, 
    and starts the training
    """
    self.setup_wandb(project_name)
    self.setup_hyperparams(batch_size, test_batch_size)

    self.setup_dataloaders(train_path, val_path, scale, reupscale, 
                          single, size, shuffle, num_workers)
    
    self.load_model(model)
    self.setup_optimizer(optimizer, optim_kwargs)
    self.setup_train_step(training_step)
    self.setup_test_step(test_step)
    self.setup_loss = loss_func
    if resume == True:
      self.resume_training(resume_path)

    #starting the training with all the parameters and settings provided
    self.start_training()

    #Save the model after the training is finished
    self.model_save(save_path)


def train_step(args, model, device, train_loader, optimizer, loss_func):
    model.train()
    total_loss = 0
    for data, target in train_loader:
        
        optimizer.zero_grad()
        data = data.to_device(data)
        target = target.to_device(target)
        
        output = model.forward(data,target)

        loss = loss_func(output, target)

        total_loss += loss
        
        loss.backward()
        optimizer.step()
        
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                              params.grad_clip).item()
        
        if args.skip_threshold == -1 or grad_norm < args.skip_threshold:
            optimizer.step()

    return {"Training Loss": total_loss}

 

def test_step(args, model, device, test_loader, loss_func):
    model.eval()
    
    example_images = []
    total_loss = 0
    with torch.no_grad():
    
        for data,target in test_loader:
            
            data = data.to_device(data)
            target = target.to_device(target)
            
            output = model(data)
            
            test_loss = loss_func(output, target)
            total_loss += test_loss
            
    example_images.append(wandb.Image(data[0],
                                      caption="Pred: {} Truth: {}".format(output[0].item(),
                                                                          target[0])))
    
    return {"Examples": example_images,
            "Test Loss": total_loss}