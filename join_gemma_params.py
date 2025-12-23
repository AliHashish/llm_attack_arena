import json

path = 'Attacks\\Parameter\\Results\\Parameters_google_gemma-3n-E4B-it.json'

with open('Attacks\\Parameter\\Results\\Parameters_google_gemma-3n-E4B-it_1.json', 'r') as file:
    data1 = json.load(file)

with open('Attacks\\Parameter\\Results\\Parameters_google_gemma-3n-E4B-it_2.json', 'r') as file:
    data2 = json.load(file)

with open('Attacks\\Parameter\\Results\\Parameters_google_gemma-3n-E4B-it_3.json', 'r') as file:
    data3 = json.load(file)


combined_data = data1 + data2 + data3

with open(path, 'w') as file:
    json.dump(combined_data, file, indent=4)