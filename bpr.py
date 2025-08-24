from utils import ban_list, attacks_list, developers
import os
import json

def is_banned_response(response):
    for phrase in ban_list:
        if phrase in response:
            return True
    return False

def get_developer_from_file_name(file_name):
    for developer in developers.values():
        if developer in file_name:
            return developer
    return None

if __name__ == "__main__":
    
    '''
    dictionary:{
        attack_1: {
            "developer_name_value_1": "value",
            "developer_name_value_2": "value",
            "developer_name_value_3": "value"
        },
        attack_2: {
            "developer_name_value_1": "value",
            "developer_name_value_2": "value",
            "developer_name_value_3": "value"
        } ....
    }
    
    '''
    results = dict()
    for attack in attacks_list:
        print(f"Evaluating attack: {attack}")
        results[attack] = dict()
        # for each attack, open folder Attacks/{attack}/Results
        attack_folder = os.path.join("Attacks", attack, "Results")
        if os.path.exists(attack_folder):
            print(f"Found attack folder: {attack_folder}")
        else:
            print(f"Attack folder not found: {attack_folder}")

        # For each attack, check for result files
        for result_file in os.listdir(attack_folder):
            if result_file.endswith(".json"):
                print(f"Found result file: {result_file}")
                developer = get_developer_from_file_name(result_file)
                if developer:
                    print(f"Detected developer: {developer}")
                else:
                    print("Developer not found.")

                # Read the result file
                with open(os.path.join(attack_folder, result_file), "r") as f:
                    result_data = json.load(f)
                    total = len(result_data)
                    count = 0
                    for entry in result_data:
                        if not is_banned_response(entry.get("response", "")):
                            count += 1

                    # Store the result in the dictionary
                    results[attack][f"{developer}"] = {
                        "total": total,
                        "passed": count,
                        "fraction": (count / total if total > 0 else 0) * 100
                    }

    # Save the results to a summary file
    summary_file = os.path.join("bpr_results.json")
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=4)
