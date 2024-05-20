import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    config.study_name = 'force_prediction_full_data_big_model_relative_error'
    config.dataset = '/local/disk1/ebeqa/coefficient_predictor/data/naca_foil_data_full'
    config.output_dir = '/local/disk1/ebeqa/coefficient_predictor/Outputs/encoder_input'
    config.checkpoint_dir = '/local/disk1/ebeqa/coefficient_predictor/Outputs/force_prediction_full_data_big_model/checkpoints/Final'  # this is also for fine tunning
    config.trainer = 'inference'  # 'train' or 'inference'
    config.train_parallel = True
    config.load_train_state = False  # in case you stopped training and want to continue
    config.num_epochs = 200
    config.batch_size = 1
    config.shuffle_buffer_size = 1024
    config.loss_function = 'MSE'  # MSE or MAE or Relative_error or Huber
    config.learning_rate_scheduler = "sgdr"
    config.learning_rate_end_value = 2.0973550568070488e-05
    config.sgdr_restarts = 1 #int(config.num_epochs / 50)
    config.warmup_fraction = 0
    config.weight_decay = 0
    config.output_frequency = 10

    config.vit = ml_collections.ConfigDict()
    config.vit.img_size = (200, 200)
    config.vit.patch_size = (10, 10)  # num_patches = (img_size / patch_size)
    config.vit.hidden_size = (config.vit.patch_size[0] ** 2) *2   #len(config.channels)  # patch_size^2 * num_channels
    config.vit.num_layers = 3
    config.vit.num_heads = 1 #len(config.channels)  # every aerodynamic property(pressure, ux, uy) gets a head,  might need to change this
    config.vit.dim_mlp = config.vit.hidden_size
    config.vit.dropout_rate = 0.0
    config.vit.att_dropout_rate = 0.0

    config.head = ml_collections.ConfigDict()
    config.head.hidden_sizes = [1000, 500, 50]
    config.head.num_targets = 3

    config.fine_tune = ml_collections.ConfigDict()
    config.fine_tune.enable = False
    config.fine_tune.dataset = ''
    config.fine_tune.checkpoint_dir = ''
    config.fine_tune.load_train_state = False #in case you stopped training and want to continue
    config.fine_tune.layers_to_train = ('Head')  # while fine tuning
    
    return config