class Callback:
    """Base class for all callbacks. Provides a common interface for handling
    the start and end of training/validation batches and epochs.
    """

    def on_train_start(self, **kwargs):
        """Called at the start of training."""
        pass

    def on_train_end(self, **kwargs):
        """Called at the end of training."""
        pass

    def on_train_epoch_start(self, epoch, **kwargs):
        """Called at the start of each training epoch."""
        pass

    def on_train_epoch_end(self, epoch, **kwargs):
        """Called at the end of each training epoch."""
        pass

    def on_train_batch_start(self, batch, **kwargs):
        """Called at the start of each training batch."""
        pass

    def on_train_batch_end(self, batch, **kwargs):
        """Called at the end of each training batch."""
        pass

    def on_valid_start(self, **kwargs):
        """Called at the start of validation."""
        pass

    def on_valid_end(self, **kwargs):
        """Called at the end of validation."""
        pass

    def on_valid_epoch_start(self, epoch, **kwargs):
        """Called at the start of each validation epoch."""
        pass

    def on_valid_epoch_end(self, epoch, **kwargs):
        """Called at the end of each validation epoch."""
        pass

    def on_valid_batch_start(self, batch, **kwargs):
        """Called at the start of each validation batch."""
        pass

    def on_valid_batch_end(self, batch, **kwargs):
        """Called at the end of each validation batch."""
        pass
