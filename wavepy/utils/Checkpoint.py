import os
import time
import shutil
import logging

import torch

class Checkpoint(object):
    """
    The Checkpoint class manages the saving and loading of a model during training. It allows training to be suspended
    and resumed at a later time (e.g. when running on a cluster using sequential jobs).

    To make a checkpoint, initialize a Checkpoint object with the following args; then call that object's save() method
    to write parameters to disk.

    Args:
        model (seq2seq): seq2seq model being trained
        optimizer (Optimizer): stores the state of the optimizer
        epoch (int): current epoch (an epoch is a loop through the full training data)
        step (int): number of examples seen within the current epoch

    Attributes:
        CHECKPOINT_DIR_NAME (str): name of the checkpoint directory
        TRAINER_STATE_NAME (str): name of the file storing trainer states
        G_MODEL_NAME (str): name of the file storing model
    """

    CHECKPOINT_DIR_NAME = 'checkpoints_backup/checkpoints'
    TRAINER_STATE_NAME = 'trainer_states.pt'
    G_MODEL_NAME = 'G.pt'

    def __init__(self, G, Goptimizer, epoch, step, config, Doptimizer=None, path=None, overlap=-1e10):
        self.G = G
        self.Goptimizer = Goptimizer
        self.epoch = epoch
        self.step = step
        self.config = config
        self._path = path
        self.overlap = overlap
        self.logger = logging.getLogger(__name__)

        if config['bidirectional']:
            network_direc = 'bidirectional'
        else:
            network_direc = 'unidirectional'
        strain = config['strain'].upper()
        rcell = config['rnn_cell'].upper()
        data_token = config['path'].split('_')[-1]
        if not 'bias' in data_token:
            data_token = 'bias_'+data_token
        alpha_token = 'alpha_'+str(config['alpha'])
        isize_token = 'Isize_'+str(config['inp_size'])
        hiddens = str(config['num_stacked_layer'])+'n'+str(config['hidden_size'])
        tfr = 'TFR'+str(config['teacher_forcing_ratio']).replace('.', '')
        self.G_model_structure = strain+'_'+network_direc \
                                +'_'+rcell+'_'+hiddens+'_'+tfr \
                                +'_'+data_token+'_'+alpha_token+'_'+isize_token
        if config['start_token'] == 'input':
            self.G_model_structure += '_ST_input'

    @property
    def path(self):
        if self._path is None:
            raise LookupError("The checkpoint has not been saved.")
        return self._path

    def save(self):
        """
        Saves the current model and related training parameters into a subdirectory of the checkpoint directory.
        The name of the subdirectory is the current local time in Y_M_D_H_M_S format.
        Args:
            experiment_dir (str): path to the experiment root directory
        Returns:
             str: path to the saved checkpoint subdirectory
        """
        date_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())

        self._path = os.path.join(self.CHECKPOINT_DIR_NAME, self.G_model_structure, date_time)
        path = self._path

        od_path = os.path.join(self.CHECKPOINT_DIR_NAME, self.G_model_structure)
        if os.path.exists(od_path):
            outdated = os.listdir(od_path)
            for od in outdated:
                self.logger.info("Remove outdated checkpoint %s" % od)
                shutil.rmtree(os.path.join(od_path, od))

        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
        torch.save({'epoch': self.epoch,
                    'step': self.step,
                    'Goptimizer': self.Goptimizer.state_dict()
                   },
                   os.path.join(path, self.TRAINER_STATE_NAME))
        torch.save(self.G.state_dict(), os.path.join(path, self.G_MODEL_NAME))

        with open(os.path.join(path, 'overlap'), 'w') as f:
            f.write('%.18f' % self.overlap)

        return path

    @classmethod
    def load(cls, G, Goptimizer, path, config, device=0):
        """
        Loads a Checkpoint object that was previously saved to disk.
        Args:
            path (str): path to the checkpoint subdirectory
        Returns:
            checkpoint (Checkpoint): checkpoint object with fields copied from those stored on disk
        """
        if torch.cuda.is_available():
            resume_checkpoint = torch.load(os.path.join(path, cls.TRAINER_STATE_NAME), map_location='cuda:%s'%device)
            G.load_state_dict(torch.load(os.path.join(path, cls.G_MODEL_NAME), map_location='cuda:%s'%device))
        else:
            resume_checkpoint = torch.load(os.path.join(path, cls.TRAINER_STATE_NAME), map_location=lambda storage, loc: storage)
            G.load_state_dict(torch.load(os.path.join(path, cls.G_MODEL_NAME), map_location=lambda storage, loc: storage))

        G.flatten_parameters() # make RNN parameters contiguous
        Goptimizer.load_state_dict(resume_checkpoint['Goptimizer'])

        with open(os.path.join(path, 'overlap'), 'r') as f:
            overlap = float(f.readline())
        return Checkpoint(G=G,
                          Goptimizer=Goptimizer,
                          epoch=resume_checkpoint['epoch'],
                          step=resume_checkpoint['step'],
                          config=config,
                          path=path), overlap

    @classmethod
    def get_latest_checkpoint(cls, config):
        """
        Given the path to an experiment directory, returns the path to the last saved checkpoint's subdirectory.

        Precondition: at least one checkpoint has been made (i.e., latest checkpoint subdirectory exists).
        Args:
            experiment_path (str): path to the experiment directory
        Returns:
             str: path to the last saved checkpoint's subdirectory
        """

        if config['bidirectional']:
            network_direc = 'bidirectional'
        else:
            network_direc = 'unidirectional'
        strain = config['strain'].upper()
        rcell = config['rnn_cell'].upper()
        data_token = config['path'].split('_')[-1]
        if not 'bias' in data_token:
            data_token = 'bias_'+data_token
        alpha_token = 'alpha_'+str(config['alpha'])
        isize_token = 'Isize_'+str(config['inp_size'])
        hiddens = str(config['num_stacked_layer'])+'n'+str(config['hidden_size'])
        tfr = 'TFR'+str(config['teacher_forcing_ratio']).replace('.', '')
        G_model_structure = strain+'_'+network_direc+'_'+rcell+'_'+hiddens+'_'+tfr+'_'+data_token+'_'+alpha_token+'_'+isize_token
        if config['start_token'] == 'input':
            G_model_structure += '_ST_input'

        checkpoints_path = os.path.join(cls.CHECKPOINT_DIR_NAME, G_model_structure)
        all_times = sorted(os.listdir(checkpoints_path), reverse=True)
        return os.path.join(checkpoints_path, all_times[0])

    @classmethod
    def get_every_checkpoint(cls, config):
        if config['bidirectional']:
            network_direc = 'bidirectional'
        else:
            network_direc = 'unidirectional'
        strain = config['strain'].upper()
        rcell = config['rnn_cell'].upper()
        data_token = config['path'].split('_')[-1]
        if not 'bias' in data_token:
            data_token = 'bias_'+data_token
        alpha_token = 'alpha_'+str(config['alpha'])
        hiddens = str(config['num_stacked_layer'])+'n'+str(config['hidden_size'])
        tfr = 'TFR'+str(config['teacher_forcing_ratio']).replace('.', '')
        G_model_structure = strain+'_'+network_direc+'_'+rcell+'_'+hiddens+'_'+tfr+'_'+data_token+'_'+alpha_token
        if config['start_token'] == 'input':
            G_model_structure += '_ST_input'

        checkpoints_path = os.path.join(cls.CHECKPOINT_DIR_NAME, G_model_structure)
        all_times = sorted(os.listdir(checkpoints_path), reverse=True)
        return [os.path.join(checkpoints_path, atime) for atime in all_times]

















