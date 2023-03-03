# --------------------------------------------------------
#           PYTHON PROGRAM
# Here is where we are going to define our set of...
# - Imports
# - Global Variables
# - Functions
# ...to achieve the functionality required.
# When executing > python 'this_file'.py in a terminal,
# the Python interpreter will load our program,
# but it will execute nothing yet.
# --------------------------------------------------------

import json
import pymongo

# --------------------------------------------------------
# FUNCTION load_json_file
# --------------------------------------------------------
def load_json_file(file_path):
    my_file = open(file_path, 'r')
    content = json.load(my_file)
    my_file.close()
    return content

# --------------------------------------------------------
# FUNCTION store_json_file
# --------------------------------------------------------
def store_json_file(file_path, d):
    my_file = open(file_path, 'w')
    json.dump(d, my_file)
    my_file.close()

# ------------------------------------------
# FUNCTION Query1
# ------------------------------------------
def query1(c):

    # 1. We delete the database
    c.drop_database('bank')

    # 2. We create the database
    db = c.bank

    # 3. We create the collection
    collection = db.staff

    # 5. We read from file
    content = load_json_file("bank_data.json")
    documents = content["documents"]

    # 6. Insert the new elements modified
    collection.insert(documents)

# ------------------------------------------
# FUNCTION Query2
# ------------------------------------------
def query2(c):
    # 1. We get the result of the query
    res = []
    for i in c.find({"eyeColor" : "brown" , "isActive" : True}):
        res.append(i)

    # 2. We return this result
    return res

# ------------------------------------------
# FUNCTION Query3
# ------------------------------------------
def query3(c):
    # 1. We get the result of the query
    res = []


    # 2. We return this result
    return res

# ------------------------------------------
# FUNCTION Query4
# ------------------------------------------
def query4(c):
    # 1. We get the result of the query
    res = []

    # 2. We return this result
    return res

# ------------------------------------------
# FUNCTION Query5
# ------------------------------------------
def query5(c):
    # 1. We get the result of the query
    res = []
    res_updated = []

    # 2. We filter the collection to get just the desired documents

    # 3. We filter the fields of the documents

    # 3. We return this result
    return res_updated

# ------------------------------------------
# FUNCTION Query6a
# ------------------------------------------
def query6a(c):
    # 1. We get the result of the query
    res = []

    # 2. We restrict it to the first five
    updated_res = []

    # 2.1. If the length of res is greater than 5

        # 2.1.1. We use a loop to collect the first 5 documents

    # 2.2. If the length is smaller or equal than five we can just assign update_res to res

    # 3. We return this result
    return updated_res

# ------------------------------------------
# FUNCTION Query6b
# ------------------------------------------
def query6b(c):
    # 1. We get the result of the query
    res = []

    # 2. We remove the first two

    # 3. We return this result
    return res

# ------------------------------------------
# FUNCTION Query7
# ------------------------------------------
def query7(c):
    # 1. We get the result of the query. We need to use sort()
    res = []

    # 2. Similar Query6a. We collect by the property and then we take as much as the 10 first.
    updated_res = []

    # 2. We return this result
    return updated_res

# ------------------------------------------
# FUNCTION Query8
# ------------------------------------------
def query8(c):
    # 1. We get the result of the query. We need to use aggregate()
    res = []

    # 2. We return this result
    return res

# ------------------------------------------
# FUNCTION Query9
# ------------------------------------------
def query9(c):
    # 1. We get the result of the query. We need to use aggregate()
    res = []

    # 2. We return this result
    return res

# ------------------------------------------
# FUNCTION Main
# ------------------------------------------
def my_main():
    # 1. We set up the connection to mongod.exe
    client = pymongo.MongoClient()

    # 2. We restart the database to its initial status
    query1(client)

    # 3. We get the bank database
    db = client.bank

    # 4. Get the staff collection
    collection = db.staff

    # 5. We set up the main loop
    print("---------  MENU ----------")
    print("2. Query 2")
    print("3. Query 3")
    print("4. Query 4")
    print("5. Query 5")
    print("6. Query 6a")
    print("16. Query 6b")
    print("7. Query 7")
    print("8. Query 8")
    print("9. Query 9")

    # 5.1. Ask the user for an option
    option = int(input("Select Option:"))

    # 5.2. Trigger the associated query asked by the user
    res = []

    if option == 2:
        res = query2(collection)
    if option == 3:
        res = query3(collection)
    if option == 4:
        res = query4(collection)
    if option == 5:
        res = query5(collection)
    if option == 6:
        res = query6a(collection)
    if option == 16:
        res = query6b(collection)
    if option == 7:
        res = query7(collection)
    if option == 8:
        res = query8(collection)
    if option == 9:
        res = query9(collection)

    # 5.3. Print the result of the query
    # We traverse the documents of the result
    for i in range(0, len(res)):
        # 5.3.1. We get the i-est document
        document = res[i]
        print("---------------------------------")
        print("--- DOCUMENT", (i+1), "INFO   ---")
        print("---------------------------------")
        # 5.3.2. We traverse all the fields of the document
        for j in document:
            print(j, " : ", document[j])

# ---------------------------------------------------------------
#           PYTHON EXECUTION
# This is the main entry point to the execution of our program.
# It provides a call to the 'main function' defined in our
# Python program, making the Python interpreter to trigger
# its execution.
# ---------------------------------------------------------------
if __name__ == '__main__':
    my_main()

