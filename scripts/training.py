class Trainer:
  """
  We can add some comment here for later, could be really useful
  """
  def __init__(self):
    """

    """
    #################################################################
    continue

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

  def model_save(self, path):
    """
    Saves the model in the current state to the specified path, and with 
    the specified name
    """
    torch.save(model.state_dict(), path + self.project_name + ".h5")
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
                        batch_size = 4, shuffle = True,
                        num_workers = 0):
    """
    Set-up and load the dataloders for the training
    using the SRDataLoader class
    """
    self.dataloader_main = SEDataLoader(train_path , scale,
                                        reupscale, single,
                                        size, batch_size,
                                        shuffle, num_workers)
    self.train_dataloader = dataloader_main.get_dataloader()

    self.dataloader_main = SEDataLoader(val_path , scale,
                                        reupscale, single,
                                        size, batch_size,
                                        shuffle, num_workers)
    self.test_dataloader = test_dataloader

  def load_model(self, model)
    self.model = model

  def setup_optimizer(self):
    """
    
    """
    #############################################
    continue

  def setup_loss(self):
    """
    
    """
    #############################################
    continue

  def warmup(self):
    """
    https://www.reddit.com/r/MachineLearning/comments/es9qv7/d_warmup_vs_initially_high_learning_rate/
    this might be the momentum in the hyperparameters tho
    """
    #############################################
    continue
  
  def setup_grad_skip(self):
    """
    we should set-up here the gradient skipping like in the VDVAE paper
    """
    #############################################
    continue

  def start_training(self):
    """
    Main function for starting the training proccess after all the other
    parth have been initialized
    """
    # Set cuda or cpu based on config and availability
    use_cuda = not config.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    #kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Set random seeds and deterministic pytorch for reproducibility
    random.seed(config.seed)       # python random seed
    torch.manual_seed(config.seed) # pytorch random seed
    numpy.random.seed(config.seed) # numpy random seed
    torch.backends.cudnn.deterministic = True

    ###
    # Set Model
    ###

    ###
    # Set optimizer
    ###

    ###
    # Set evaluator
    ###

    # Starting the watch in order to log all layer dimensions, gradients and
    # model parameters to the dashboard
    # Using log="all" log histograms of parameter values in addition to gradients
    wandb.watch(model, log="all")

    for epoch in range(1, config.epochs + 1):

      for batch_idx, (data, target) in enumerate(self.train_dataloader):
        self.train_step(self.model, device, data, target, optimizer)
        #SOME OTHER STUFF NEED TO HAPPEN HERE

      for batch_idx, (data, target) in enumerate(self.val_dataloader):
        self.test_step(self.model, device, data, target)
        #SOME OTHER STUFF HERE TOO

    #Save the model after the training is finished
    self.model_save()

  def Main_start(self)
  """
  Function that receives all arguments, initializes every module, and starts
  the training
  """