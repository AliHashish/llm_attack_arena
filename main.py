import argparse
import sys
import attack
from utils import model_names_list

ATTACK_REGISTRY = {
    "DeepInception": attack.DeepInception,
    "TemplateJailbreak": attack.TemplateJailbreak,
    "Parameters": attack.Parameters,
    "FFA": attack.FFA,
}


# Running the specified attack
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
    parser.add_argument('--type', choices=list(ATTACK_REGISTRY.keys()), required=True,
                        help='Attack method to execute (mandatory for attack mode)')
    
    args = parser.parse_args()
    
    try:
        run_attack(args.model, args.type)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
