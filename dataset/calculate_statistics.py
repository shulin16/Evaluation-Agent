import json
from collections import Counter

with open('dataset/open_ended_user_questions.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

ability_counter = Counter()
general_specific_counter = Counter()
category_counter = Counter()


for item in data['questions']:
    ability = item['ability']
    general_or_specific = item['general_or_specific']
    category = item['category']

    ability_counter[ability] += 1
    general_specific_counter[general_or_specific] += 1
    category_counter[category if category else 'No Category'] += 1 

def print_counts(counter, title):
    print(f"\n{title}:")
    for key, count in counter.items():
        print(f"  {key}: {count}")

print_counts(ability_counter, "Ability Counts")
print_counts(general_specific_counter, "General/Specific Counts")
print_counts(category_counter, "Category Counts")
