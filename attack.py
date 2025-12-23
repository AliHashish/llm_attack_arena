import numpy as np
import torch.nn as nn
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter
import subprocess
from prompt_toolkit.shortcuts import ProgressBar

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
            'model': model,
        }

    def run(self):
        print("Running main.py with specified parameters for DeepInception")
        command = "python"
        script = "main.py"
        args = args_to_cmd(self.parameters)

        cmd = ['stdbuf', '-oL', command, '-u', script] + args

        try:
            with subprocess.Popen(cmd, cwd="./Attacks/DeepInception", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as sp:
                for line in sp.stdout:
                    print(line)
        finally:
            sp.terminate()
            sp.wait()

class TemplateJailbreak(BaseAttackModel):
    def __init__(self,model):
        super().__init__()
        self.parameters = {
            'model':model,
        }

    def run(self):
        print("Running main.py with specified parameters for TemplateJailbreak")
        command = "python"
        script = "main.py"
        args = args_to_cmd(self.parameters)

        cmd = ['stdbuf', '-oL', command, '-u', script] + args

        try:
            with subprocess.Popen(cmd, cwd="./Attacks/TemplateJailbreak/", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as sp:
                for line in sp.stdout:
                    print(line)
        finally:
            sp.terminate()
            sp.wait()

class Parameters(BaseAttackModel):
    def __init__(self,model):
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

        try:
            with subprocess.Popen(cmd, cwd="./Attacks/Parameter/", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as sp:
                for line in sp.stdout:
                    print(line)
        finally:
            sp.terminate()
            sp.wait()

class FFA(BaseAttackModel):
    def __init__(self, model):
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

        try:
            with subprocess.Popen(cmd, cwd="./Attacks/FFA/", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as sp:
                for line in sp.stdout:
                    print(line)
        finally:
            sp.terminate()
            sp.wait()