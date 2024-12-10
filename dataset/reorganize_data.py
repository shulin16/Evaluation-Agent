import json
import os

input_filename = 'dataset/open_ended_user_questions.json'

base_name, ext = os.path.splitext(input_filename)
output_filename = f"{base_name}_summary{ext}"

with open(input_filename, 'r', encoding='utf-8') as file:
    data = json.load(file)

ability_dict = {}
general_specific_dict = {}
category_dict = {}

for item in data['questions']:
    question = item['question']
    ability = item['ability']
    general_or_specific = item['general_or_specific']
    category = item['category'] if item['category'] else 'No Category'

    ability_dict.setdefault(ability, []).append(question)
    general_specific_dict.setdefault(general_or_specific, []).append(question)
    category_dict.setdefault(category, []).append(question)


output_data = {
    "Ability": ability_dict,
    "General/Specific": general_specific_dict,
    "Category": category_dict
}

with open(output_filename, 'w', encoding='utf-8') as outfile:
    json.dump(output_data, outfile, indent=2, ensure_ascii=False)

print(f"Output JSON file '{output_filename}' has been created successfully.")
