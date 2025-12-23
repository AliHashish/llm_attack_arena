import gc
import numpy as np
import torch
import torch.nn as nn
import time
import argparse
import os
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter
import subprocess
from prompt_toolkit.shortcuts import ProgressBar
import re
import gc
import logging
fro

logging.basicConfig(level=logging.INFO)
os.environ["MKL_THREADING_LAYER"] = "GNU"


def _clear_memo():
    tensors_to_delete = []
    
    for obj in gc.get_objects():
        
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(obj)
                tensors_to_delete.append(obj)
        except:
            pass
    
    for tensor in tensors_to_delete:
        # print("delete",tensor)
        del tensor
    
    gc.collect()
    torch.cuda.empty_cache()
    

def args_to_cmd(args):
    cmd = []
    for key, value in args.items():
        cmd.append(f"--{key}")
        cmd.append(str(value))
    return cmd

class BaseAttackModel:
    def __init__(self):
        self.parameters = {}

    def collect_parameters(self):
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def Execute(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

class DeepInception(BaseAttackModel):
    def __init__(self,model):
        super().__init__()
        self.parameters = {
            'target-model': model,
            'target-max-n-tokens': 128,
            'exp_name':"main",
            'defense': 'none'
        }



    def run(self):
        print("Running main.py with specified parameters for DeepInception")
        command = "python"
        script = "main.py"
        args = args_to_cmd(self.parameters)

        cmd = ['stdbuf', '-oL', command, '-u', script] + args

        # Start the subprocess and capture its output
        try:
            with subprocess.Popen(cmd, cwd="./Attacks/DeepInception", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as sp:
                for line in sp.stdout:
                    print(line)
                    logging.info(line)
        finally:
            sp.terminate()
            sp.wait()

class TemplateJailbreak(BaseAttackModel):
    def __init__(self,model):
        # Set up default parameters
        super().__init__()
        self.parameters = {
            'model':model,
            'question_count':100,
        }


    def run(self):
        print("Running main.py with specified parameters for TemplateJailbreak")
        command = "python"
        script = "main.py"
        args = args_to_cmd(self.parameters)

        cmd = ['stdbuf', '-oL', command, '-u', script] + args

        # Start the subprocess and capture its output
        try:
            with subprocess.Popen(cmd, cwd="./Attacks/TemplateJailbreak/", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as sp:
                for line in sp.stdout:
                    print(line)
                    logging.info(line)
        finally:
            sp.terminate()
            sp.wait()

class Parameters(BaseAttackModel):
    def __init__(self,model):
        # Set up default parameters
        super().__init__()
        self.parameters = {
            'model':model,          
        }


    def run(self):
        print("Running main.py with specified parameters for Parameters")
        command = "python"
        script = "attack.py"
        args = args_to_cmd(self.parameters)

        cmd = ['stdbuf', '-oL', command, '-u', script] + args

        # Start the subprocess and capture its output
        try:
            with subprocess.Popen(cmd, cwd="./Attacks/Parameter/", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as sp:
                for line in sp.stdout:
                    print(line)
                    logging.info(line)
        finally:
            sp.terminate()
            sp.wait()

class FFA(BaseAttackModel):
    def __init__(self, model):
        # Set up default parameters
        super().__init__()
        self.parameters = {
            'model': model,
        }

    def run(self):
        print("Running main.py with specified parameters for FFA")
        command = "python"
        script = "main.py"
        args = args_to_cmd(self.parameters)

        cmd = ['stdbuf', '-oL', command, '-u', script] + args

        # Start the subprocess and capture its output
        try:
            with subprocess.Popen(cmd, cwd="./Attacks/FFA/", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as sp:
                for line in sp.stdout:
                    print(line)
                    logging.info(line)
        finally:
            sp.terminate()
            sp.wait()