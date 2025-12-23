import argparse
import sys
from prompt_process import load_json
import attack
from utils import model_names_list

ATTACK_REGISTRY = {
    "DeepInception": attack.DeepInception,
    "Jailbroken": attack.Jailbroken,
    "TemplateJailbreak": attack.TemplateJailbreak,
    "Parameters": attack.Parameters,
    "FFA": attack.FFA,
}


def run_attack(model_name, attack_type):
    """Execute an attack on a specified model."""
    if attack_type not in ATTACK_REGISTRY:
        raise ValueError(f"Attack type '{attack_type}' not recognized. Available: {', '.join(ATTACK_REGISTRY.keys())}")
    
    print(f"Applying {attack_type} attack to {model_name}")
    attack_instance = ATTACK_REGISTRY[attack_type](model=model_name)
    attack_instance.run()


def main():
    parser = argparse.ArgumentParser(description="Run attack mechanisms on AI models")
    parser.add_argument('--model', choices=list(model_names_list.keys()), required=True, 
                        help='Target model for attack execution or result processing')
    parser.add_argument('--attack-model', help='Attack model identifier for generative attack methods')
    parser.add_argument('--mode', choices=['attack', 'process'], required=True,
                        help='Choose to execute an attack or process existing results')
    parser.add_argument('--type', choices=list(ATTACK_REGISTRY.keys()),
                        help='Attack method to execute (mandatory for attack mode)')
    
    args = parser.parse_args()
    
    # Validate attack mode requirements
    if args.mode == 'attack' and not args.type:
        parser.error("--type is required when mode is 'attack'")
    
    try:
        # Execute operation
        if args.mode == 'attack':
            run_attack(args.model, args.type)
        elif args.mode == 'process':
            print(f"Processing results from ./Results/{args.model}")
            load_json(f'./Results/{args.model}')
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
