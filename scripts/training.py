class Trainer:
  """
  We can add some comment here for later, could be really useful
  """
  def __init__(self):
    """

    """
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

  def setup_dataloaders(self, train_dataloader, test_dataloader):
    """
    Set-up and load the dataloders for the training
    """
    self.train_dataloader = train_dataloader
    self.test_dataloader = test_dataloader

  def setup_optimizer(self):
    """
    
    """
    continue

  def setup_loss(self):
    """
    
    """
    continue

  def warmup(self):
    """
    https://www.reddit.com/r/MachineLearning/comments/es9qv7/d_warmup_vs_initially_high_learning_rate/
    this might be the momentum in the hyperparameters tho
    """
    continue
  
  def setup_grad_skip(self):
    """
    we should set-up here the gradient skipping like in the VDVAE paper
    """
    continue

  def start_training():
    """

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
    # Set Dataloaders
    ###

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
      self.train_step() #MAKE TRAIN FUNCTION
      self.test_step()  #MAKE TEST FUNCTION

    #Save the model after the training is finished
    self.model_save()