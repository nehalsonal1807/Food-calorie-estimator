import pandas as pd
import numpy as np
from collections import deque

# food dataset (manually typed)
food_items = [
    # name, calories, protein(g), carbs(g), fat(g), category
    ("Chicken Breast", 165, 31.0, 0.0, 3.6, "Protein"),
    ("Egg", 78, 6.0, 0.6, 5.0, "Protein"),
    ("Tuna", 132, 28.0, 0.0, 1.0, "Protein"),
    ("Paneer", 296, 18.0, 3.4, 22.0, "Protein"),
    ("Lentils", 116, 9.0, 20.0, 0.4, "Protein"),
    ("Chickpeas", 164, 9.0, 27.0, 2.6, "Protein"),
    ("Tofu", 76, 8.0, 2.0, 4.2, "Protein"),

    ("White Rice", 206, 4.0, 45.0, 0.4, "Carbs"),
    ("Brown Rice", 215, 5.0, 45.0, 1.6, "Carbs"),
    ("Oats", 154, 5.0, 28.0, 2.5, "Carbs"),
    ("Wheat Bread", 79, 3.0, 15.0, 1.0, "Carbs"),
    ("Potato", 90, 2.0, 21.0, 0.1, "Carbs"),
    ("Pasta", 220, 8.0, 43.0, 1.3, "Carbs"),
    ("Corn", 132, 5.0, 29.0, 1.8, "Carbs"),

    ("Banana", 89, 1.1, 23.0, 0.3, "Fruit"),
    ("Apple", 52, 0.3, 14.0, 0.2, "Fruit"),
    ("Mango", 60, 0.8, 15.0, 0.4, "Fruit"),
    ("Orange", 47, 0.9, 12.0, 0.1, "Fruit"),
    ("Grapes", 67, 0.6, 17.0, 0.4, "Fruit"),
    ("Watermelon", 30, 0.6, 8.0, 0.2, "Fruit"),

    ("Broccoli", 55, 3.7, 11.0, 0.6, "Vegetable"),
    ("Spinach", 23, 2.9, 3.6, 0.4, "Vegetable"),
    ("Carrot", 41, 0.9, 10.0, 0.2, "Vegetable"),
    ("Tomato", 18, 0.9, 3.9, 0.2, "Vegetable"),
    ("Cucumber", 16, 0.7, 3.6, 0.1, "Vegetable"),

    ("Milk", 42, 3.4, 5.0, 1.0, "Dairy"),
    ("Yogurt", 59, 10.0, 3.6, 0.4, "Dairy"),
    ("Cheese", 402, 25.0, 1.3, 33.0, "Dairy"),

    ("Butter", 717, 0.9, 0.1, 81.0, "Fats"),
    ("Almonds", 579, 21.0, 22.0, 50.0, "Fats"),
    ("Peanut Butter", 588, 25.0, 20.0, 50.0, "Fats"),
    ("Olive Oil", 884, 0.0, 0.0, 100.0, "Fats"),
]

cols = ["food", "cal", "proteinin_g", "carbsin_g", "fatin_g", "category"]
df = pd.DataFrame(food_items, columns=cols)
# Simple calorie calculation (no ML, no numpy, no sklearn)
def prediction(p, c, f):
    calories = (p * 4) + (c * 4) + (f * 9)
    return round(calories, 2)

# building of graph for BFS and DFS - each food is a node, edges connect foods in the same category
# grouping of foods by category using adjacency list
food_graph = {}

for f in df["food"]:
    food_graph[f] = []   # initialize empty list

# connecting foods in same category (probably not the most efficient way)
for i, rowA in df.iterrows():
    for j, rowB in df.iterrows():
        if i != j:
            if rowA["category"] == rowB["category"]:
                a = rowA["food"]
                b = rowB["food"]

                if b not in food_graph[a]:
                    food_graph[a].append(b)

                # kinda redundant but safer
                if a not in food_graph[b]:
                    food_graph[b].append(a)

 # Using BFS we Find similar foods level by level
def bfs_search(start_food):
    visited_nodes = []
    q = deque()

    q.append(start_food)
    visited_nodes.append(start_food)

    while len(q) > 0:
        current = q.popleft()

        for neigh in food_graph[current]:
            if neigh not in visited_nodes:
                visited_nodes.append(neigh)
                q.append(neigh)

    # remove the starting food (feels cleaner in output)
    if start_food in visited_nodes:
        visited_nodes.remove(start_food)

    return visited_nodes


# exploring the category of a food recursively through DFS
def dfs_search(start_food, visited=None):
    if visited is None:
        visited = []

    visited.append(start_food)

    for neigh in food_graph[start_food]:
        if neigh not in visited:
            dfs_search(neigh, visited)   # recursive call

    return visited


# getting food info
def get_food_details(name):
    row = df[df["food"].str.lower() == name.lower().strip()]
    if len(row) == 0:
        return None
    return row.iloc[0]


# calorie prediction
def prediction(p, c, f):
    calories = (p * 4) + (c * 4) + (f * 9)
    return round(calories, 2)


def main():
    print("==== Simple Food Calorie Tool ====")

    while True:
        print("\n1. Search food")
        print("2. Predict calories")
        print("3. Similar foods (BFS)")
        print("4. Explore category (DFS)")
        print("5. Show all data")
        print("6. Exit")
        print("-------------------------------")

        choice = input("Choice: ").strip()

        if choice == "1":
            name = input("Enter food: ")
            info = get_food_details(name)

            if info is None:
                print("Food not found... maybe typo?")
            else:
                print("\n--- Details ---")
                print("Name:", info["food"])
                print("Calories:", info["cal"], "kcal")
                print("Protein:", info["proteinin_g"], "g")
                print("Carbs:", info["carbsin_g"], "g")
                print("Fat:", info["fatin_g"], "g")
                print("Category:", info["category"])

        elif choice == "2":
            print("\nEnter the nutritional values of your food:")
            try:
                p = float(input("Protein: "))
                c = float(input("Carbs: "))
                f = float(input("Fat: "))

                res = prediction(p, c, f)
                print("Estimated Calories:", res, "kcal")

            except:
                print("Invalid input... numbers only pls")

        elif choice == "3":
            name = input("Enter food name to find similar categories: ").strip().title()

            if name not in food_graph:
                print("Not found in graph.")
            else:
                result = bfs_search(name)
                print(f"\n Foods similar to '{name}':")
                for idx, item in enumerate(result):
                    print(idx+1, "-", item)

        elif choice == "4":
            name = input("Enter food name to explore its category: ").title()

            if name not in food_graph:
                print("Not found.")
            else:
                res = dfs_search(name)

                if name in res:
                    res.remove(name)

                print(f"\n Full category explored from '{name}':")
                for i, val in enumerate(res):
                    print(i+1, val)

        elif choice == "5":
            print("\n--- Full Dataset ---")
            for i, row in df.iterrows():
                print(i+1, row["food"], "-", row["cal"], "kcal")

        elif choice == "6":
            print("\nBye Bye ! Thank you for reaching out to us\n ")
            break

        else:
            print("Invalid choice... try again")


if __name__ == "__main__":
    main()