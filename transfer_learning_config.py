class Configuration:
    def __init__(self):
        self.feature_extraction_epochs = 10
        self.fine_tuning_epochs = 20
        self.epochs_without_transfer_learning = 100
        self.batch_size = 30
        self.data_dir = "data/10-monkey-species/training"
        self.val_dir = "data/10-monkey-species/validation"

