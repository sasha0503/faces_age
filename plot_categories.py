import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

if __name__ == '__main__':
    with open("ages.txt") as f:
        ages = list(map(float, f.read().split()))
    with open("categories.txt") as f:
        categories = list(f.read().split())
    cat2age = {}
    for age, cat in zip(ages, categories):
        if cat not in cat2age:
            cat2age[cat] = []
        cat2age[cat].append(int(age))
    child_ages = {}
    for age in cat2age['child']:
        child_ages[age] = child_ages.get(age, 0) + 1
    adult_ages = {}
    for age in cat2age['adult']:
        adult_ages[age] = adult_ages.get(age, 0) + 1
    min_age = min(min(child_ages.keys()), min(adult_ages.keys()))
    max_age = max(max(child_ages.keys()), max(adult_ages.keys()))
    groups = [str(i) for i in list(range(min_age, max_age + 1))]
    values_1 = [child_ages.get(int(i), 0) for i in groups]  # children
    values_2 = [adult_ages.get(int(i), 0) for i in groups]  # adults

    fig, ax = plt.subplots()

    ax.bar(groups, values_1)
    ax.bar(groups, values_2, bottom=values_1)

    ax.set_ylabel('Number of people')
    ax.set_xlabel('Age')
    ax.set_title('Age distribution by category')
    ax.legend(['children', 'adults'])

    for label_i, label in enumerate(ax.get_xticklabels()):
        if label_i % 5 != 0:
            label.set_visible(False)

    plt.savefig('age_distribution.png')
