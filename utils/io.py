import os
import time
import random
import string
from typing import Dict, List

def rand_str(length: int=4) -> str:
    """ Generate a random string with the given length

    Args:
        length: length of the random string
    
    Return:
        random string
    """
    return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(length))

def time_str(Y: bool=True, m: bool=True, d: bool=True, H: bool=True, M: bool=True, S: bool=True) -> str:
    """ Generate a time string
    
    Args:
        Y, m, d, H, M, S: create year or not, default is True
    
    Return:
        time string
    """
    f = ""
    f += "%Y" if Y else ""
    f += "%m" if m else ""
    f += "%d" if d else ""
    f += "-" if (Y or m or d) else ""
    f += "%H" if H else ""
    f += "%M" if M else ""
    f += "%S" if S else ""

    return time.strftime(f, time.localtime(time.time()))

def mkdir_if_not_exists(dir_name: str, recursive: bool=False) -> None:
    """ Make directory with the given dir_name
    Args:
        dir_name: input directory name that can be a path
        recursive: recursive directory creation
    """
    if os.path.exists(dir_name):
        return 
    
    if recursive:
        os.makedirs(dir_name)
    else:
        os.mkdir(dir_name)

def singleton(cls):
    _instance = {}

    def inner(*args, **kwargs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kwargs)
        return _instance[cls]
    return inner

class WanDBoard():
    def __init__(self, project: str='SceneDiffuser++', log_dir: str='./tmp/', **kwargs) -> None:
        """ Init a board for visualizing the curves

        Args:
            project: the project name
            log_dir: save path of board log
        """
        self.project = project
        self.log_dir = log_dir

        import wandb
        self.wandb = wandb
        output_dir, exp_name, = '/'.join(log_dir.split('/')[:-2]), log_dir.split('/')[-2]
        ## WanDB log is save in '${output_dir}/wandb/'
        ## Please see the curve on the website: https://wandb.ai/
        self.wandb.init(project=project, dir=output_dir, name=exp_name)
    
    def close(self) -> None:
        """ Close the board and flush the log """
        self.wandb.finish()

    def write(self, write_dict: Dict) -> None:
        """ Write data to board 
        
        Args:
            write_dict: the data need to be write, which must be a dict, e.g., {'step': int, ...}
        """
        assert 'step' in write_dict, "You much specify the `step` value."
        step = write_dict.pop('step')
        self.wandb.log(write_dict, step=step)

class TensorBoard():

    def __init__(self, log_dir: str='./tmp/', **kwargs) -> None:
        """ Init a board for visualizing the curves
        
        Args:
            log_dir: save path of board log
        """
        assert log_dir is not None, "None value of log_dir!"
        self.log_dir = log_dir

        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=log_dir)
    
    def close(self) -> None:
        """ Close the board and flush the log """
        self.writer.close()
    
    def write(self, write_dict: Dict) -> None:
        """ Write data to board

        Args:
            write_dict: the data need to be write, which must be a dict, e.g., {'step': int, ...}
        """
        assert 'step' in write_dict, "You much specify the `step` value."
        step = write_dict.pop('step')
        for key in write_dict.keys():
            self.writer.add_scalar(key, write_dict[key], step)

@singleton
class Board():

    def __init__(self) -> None:
        self.board = None

    def create_board(self, platform: str, **kwargs: Dict) -> None:
        """ Create a board for visualizing the curves
        
        Args:
            platform: the platform of board, e.g., 'TensorBoard', 'WanDB'
        """
        assert platform in ['TensorBoard', 'WanDB'], f"Unsupported board platform! The value is '{platform}'."

        self.board = {
            'TensorBoard': TensorBoard,
            'WanDB': WanDBoard,
        }[platform](**kwargs)

    def close(self) -> None:
        """ Close the board and flush the log """
        self.board.close()

    def write(self, write_dict: Dict) -> None:
        """ Write data to board

        Args:
            write_dict: the data need to be write, which must be a dict, e.g., {'step': int, ...}
        """
        self.board.write(write_dict)
    
    