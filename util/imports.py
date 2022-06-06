import torch
import torch.backends.cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import pandas as pd 
from torch import Tensor
from numpy import ndarray
import random
import logging
from typing import Optional, Union, List, Dict, Tuple, Any, Callable, Iterable, Literal, Iterator
import itertools
import math
from datetime import datetime, date, timedelta
from collections import defaultdict, namedtuple
from tqdm import tqdm
from pprint import pprint
import pickle
import os
import dataclasses
from dataclasses import dataclass
import pymongo
import argparse 
import json
import yaml 
import copy 
import csv 

# ========== DGL ==========
import dgl
import dgl.function as dglfn
import dgl.nn.pytorch as dglnn
import dgl.nn.functional as dglF
import dgl.data.utils as dglutil
# =========================

# ========== PyG ==========
import torch_geometric as pyg
import torch_geometric.data as pygdata 
import torch_geometric.nn as pygnn 
import torch_geometric.nn.conv as pygconv 
import torch_geometric.loader as pygloader 
# =========================

IntTensor = FloatTensor = BoolTensor = FloatScalarTensor = Tensor
IntArray = FloatArray = BoolArray = ndarray
